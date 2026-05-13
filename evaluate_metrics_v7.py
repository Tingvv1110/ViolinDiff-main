import os
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
import resampy
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, mean_absolute_error
from torchvision.models import inception_v3
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from harritaylor_torchvggish_master.torchvggish import vggish_input
from harritaylor_torchvggish_master.torchvggish.vggish import VGGish
from panns_inference import AudioTagging
import warnings
import json

warnings.filterwarnings("ignore")


class Config:
    real_audio_dir = "/root/autodl-tmp/ViolinDiff/true_wav_dir_"  # 真实音频目录
    checkpoint_dir = "./checkpoints"  # 模型权重目录
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16  # 批大小
    sample_rate = 16000  # 音频采样率
    segment_duration = 3.0  # 音频分段时长（秒）
    target_dim = 128  # 统一的目标特征维度

    # 配置参数
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    vggish_sample_rate = 16000
    f1_threshold = 0.01
    frechet_eps = 1e-6
    kl_eps = 1e-10
    max_files = 1000
    vggish_batch_size = 256


# 音频处理函数保持不变
def load_audio(file_path, target_sr=None):
    try:
        audio, sr = sf.read(file_path, always_2d=True)
        if audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio.flatten()

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        if target_sr and sr != target_sr:
            audio = resampy.resample(audio, sr, target_sr)
        elif not target_sr and sr != Config.sample_rate:
            audio = resampy.resample(audio, sr, Config.sample_rate)

        return audio.astype(np.float32)
    except Exception as e:
        print(f"加载音频错误: {file_path} - {str(e)}")
        return np.zeros(0, dtype=np.float32)


def compute_mel(audio, sr=Config.sample_rate):
    if len(audio) < Config.hop_length:
        return np.zeros((128, 10))

    try:
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=Config.n_fft,
            hop_length=Config.hop_length,
            win_length=Config.win_length,
            n_mels=128,
            power=1.0
        )
        return librosa.power_to_db(mel, ref=np.max)
    except Exception as e:
        print(f"Mel计算错误: {str(e)}")
        return np.zeros((128, 10))


def compute_stft(audio):
    if len(audio) < Config.hop_length:
        return np.zeros((Config.n_fft // 2 + 1, 10))

    stft = librosa.stft(
        audio,
        n_fft=Config.n_fft,
        hop_length=Config.hop_length,
        win_length=Config.win_length
    )
    magnitude = np.abs(stft)
    return magnitude


# 音频级指标计算函数保持不变
def calculate_mae(audio_real, audio_gen):
    min_len = min(len(audio_real), len(audio_gen))
    if min_len == 0:
        return 0.0

    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]
    return mean_absolute_error(audio_real, audio_gen)


def calculate_f1_score(audio_real, audio_gen, threshold=Config.f1_threshold):
    min_len = min(len(audio_real), len(audio_gen))
    if min_len == 0:
        return 0.0

    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]

    bin_real = (np.abs(audio_real) > threshold).astype(int)
    bin_gen = (np.abs(audio_gen) > threshold).astype(int)

    return f1_score(bin_real, bin_gen)


def calculate_lsd(audio_real, audio_gen):
    min_len = min(len(audio_real), len(audio_gen))
    if min_len == 0:
        return 0.0

    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]

    stft_real = compute_stft(audio_real)
    stft_gen = compute_stft(audio_gen)

    log_real = np.log10(np.maximum(stft_real ** 2, 1e-10))
    log_gen = np.log10(np.maximum(stft_gen ** 2, 1e-10))

    lsd_per_frame = np.sqrt(np.mean((log_real - log_gen) ** 2, axis=0))
    return np.mean(lsd_per_frame)


def calculate_psnr(audio_real, audio_gen):
    min_len = min(len(audio_real), len(audio_gen))
    if min_len == 0:
        return 0.0

    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]

    max_val = np.max(np.abs(audio_real))
    mse = np.mean((audio_real - audio_gen) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def calculate_ssim(audio_real, audio_gen, window_size=11):
    min_len = min(len(audio_real), len(audio_gen))
    if min_len == 0:
        return 0.0

    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]

    audio_real_tensor = torch.FloatTensor(audio_real).unsqueeze(0).unsqueeze(0)
    audio_gen_tensor = torch.FloatTensor(audio_gen).unsqueeze(0).unsqueeze(0)

    window = torch.ones(1, 1, window_size) / window_size

    mu_real = F.conv1d(audio_real_tensor, window, padding=window_size // 2)
    mu_gen = F.conv1d(audio_gen_tensor, window, padding=window_size // 2)

    mu_real_sq = mu_real.pow(2)
    mu_gen_sq = mu_gen.pow(2)
    mu_real_gen = mu_real * mu_gen

    sigma_real_sq = F.conv1d(audio_real_tensor * audio_real_tensor, window, padding=window_size // 2) - mu_real_sq
    sigma_gen_sq = F.conv1d(audio_gen_tensor * audio_gen_tensor, window, padding=window_size // 2) - mu_gen_sq
    sigma_real_gen = F.conv1d(audio_real_tensor * audio_gen_tensor, window, padding=window_size // 2) - mu_real_gen

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_real_gen + C1) * (2 * sigma_real_gen + C2)) / \
               ((mu_real_sq + mu_gen_sq + C1) * (sigma_real_sq + sigma_gen_sq + C2))

    return ssim_map.mean().item()


# 特征级指标计算函数保持不变
def calculate_kid(real_activations, gen_activations, max_size=100):
    n_samples = min(len(real_activations), len(gen_activations), max_size)
    if n_samples < 2:
        return 0.0

    real_sub = real_activations[np.random.choice(len(real_activations), n_samples, replace=False)]
    gen_sub = gen_activations[np.random.choice(len(gen_activations), n_samples, replace=False)]

    real_sub = (real_sub - real_sub.mean(0)) / (real_sub.std(0) + 1e-10)
    gen_sub = (gen_sub - gen_sub.mean(0)) / (gen_sub.std(0) + 1e-10)

    kernel_real = np.dot(real_sub, real_sub.T)
    kernel_gen = np.dot(gen_sub, gen_sub.T)
    kernel_cross = np.dot(real_sub, gen_sub.T)

    kid = np.mean(kernel_real) + np.mean(kernel_gen) - 2 * np.mean(kernel_cross)
    return max(kid, 0)


def calculate_kl_divergence(P, Q):
    eps = Config.kl_eps

    P = np.clip(P, eps, 1 - eps)
    Q = np.clip(Q, eps, 1 - eps)

    log_diff = np.log(P) - np.log(Q)
    kl_values = P * log_diff

    return np.nanmean(kl_values)


def calculate_isc(features):
    if features.size == 0:
        return 0.0

    features_tensor = torch.tensor(features).float()
    features_tensor = features_tensor - features_tensor.max(dim=1, keepdim=True)[0]
    py = F.softmax(features_tensor, dim=1).mean(dim=0)

    scores = []
    for i in range(features_tensor.shape[0]):
        pyx = F.softmax(features_tensor[i], dim=0)
        kl = F.kl_div((py + 1e-10).log(), pyx + 1e-10, reduction='sum').item()
        scores.append(kl)

    mean_score = np.mean(scores) if scores else 0.0
    return np.exp(mean_score)


# 模型加载和特征提取函数保持不变
def load_vggish_model():
    urls = {
        "vggish": f"file://{os.path.abspath(os.path.join(Config.checkpoint_dir, 'vggish-10086976.pth'))}",
        "pca": f"file://{os.path.abspath(os.path.join(Config.checkpoint_dir, 'vggish_pca_params-970ea276.pth'))}"
    }
    model = VGGish(
        urls=urls,
        pretrained=True,
        preprocess=False,
        postprocess=False,
        device=Config.device
    )
    model.eval()
    return model


def load_panns_model():
    os.makedirs("/root/panns_data", exist_ok=True)
    model = AudioTagging(checkpoint_path=None, device=str(Config.device))
    return model


def extract_vggish_embeddings(audio_dir, model):
    all_embeddings = []
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    for file in audio_files:
        audio_path = os.path.join(audio_dir, file)
        examples = vggish_input.wavfile_to_examples(audio_path)

        for i in range(0, len(examples), Config.batch_size):
            batch = examples[i:i + Config.batch_size].to(Config.device)
            with torch.no_grad():
                embeddings = model(batch)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def extract_panns_embeddings(audio_dir, model):
    all_embeddings = []
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    segment_samples = int(Config.sample_rate * Config.segment_duration)

    for file in audio_files:
        audio_path = os.path.join(audio_dir, file)

        try:
            waveform, sr = torchaudio.load(audio_path)

            if sr != Config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, Config.sample_rate)
                waveform = resampler(waveform)

            min_length = int(Config.sample_rate * 1.0)
            if waveform.shape[1] < min_length:
                print(f"Skipping {audio_path}: too short ({waveform.shape[1] / sr:.2f}s)")
                continue

            segments = []
            for start in range(0, waveform.shape[1], segment_samples):
                end = start + segment_samples
                if end > waveform.shape[1]:
                    break
                segment = waveform[:, start:end]

                if segment.dim() == 2 and segment.shape[0] > 1:
                    segment = segment.mean(dim=0)
                segment = segment.numpy().astype(np.float32)
                segment /= np.max(np.abs(segment)) + 1e-8

                if segment.ndim == 1:
                    segment = segment[np.newaxis, :]

                try:
                    _, embedding = model.inference(segment)
                    all_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error extracting features from {audio_path}: {str(e)}")

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")

    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def apply_pca(embeddings, n_components, fit_pca=None):
    if embeddings.size == 0:
        return np.array([]), None

    if fit_pca is not None:
        return fit_pca.transform(embeddings), fit_pca

    pca = PCA(n_components=n_components, svd_solver='full')
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings, pca


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2

    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    distance = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return max(distance, 0)


def calculate_fad(real_emb, gen_emb):
    if real_emb.size == 0 or gen_emb.size == 0:
        return float('inf')

    mu_real = np.mean(real_emb, axis=0)
    sigma_real = np.cov(real_emb, rowvar=False)

    mu_gen = np.mean(gen_emb, axis=0)
    sigma_gen = np.cov(gen_emb, rowvar=False)

    min_eig_real = np.min(np.real(np.linalg.eigvals(sigma_real)))
    min_eig_gen = np.min(np.real(np.linalg.eigvals(sigma_gen)))

    if min_eig_real < 1e-8:
        print("Warning: Real embeddings covariance matrix not positive definite. Adding regularization.")
        sigma_real += np.eye(sigma_real.shape[0]) * 1e-6

    if min_eig_gen < 1e-8:
        print("Warning: Generated embeddings covariance matrix not positive definite. Adding regularization.")
        sigma_gen += np.eye(sigma_gen.shape[0]) * 1e-6

    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def extract_inception_features(mels, model):
    all_features = []

    for mel in mels:
        mel_img = np.stack([mel] * 3, axis=0)

        mel_tensor = torch.FloatTensor(mel_img).unsqueeze(0)
        mel_tensor = F.interpolate(mel_tensor, size=(299, 299), mode='bilinear', align_corners=False)
        mel_tensor = mel_tensor.to(Config.device)

        with torch.no_grad():
            features = model(mel_tensor)
            all_features.append(features.cpu().numpy())

    return np.vstack(all_features) if all_features else np.array([])


# 修改main函数，使其接受gen_dir和save_path参数
def process_epoch(gen_audio_dir, result_save_path):
    results = {
        'FAD': {'L-AUDIO': 0.0, 'L-MUSIC': 0.0},
        'ISC': {'L-AUDIO': 0.0, 'L-MUSIC': 0.0},
        'KID': {'L-AUDIO': 0.0, 'L-MUSIC': 0.0},
        'KL': {'L-AUDIO': {'KL(sigmoid)': 0.0, 'KL(softmax)': 0.0},
               'L-MUSIC': {'KL(sigmoid)': 0.0, 'KL(softmax)': 0.0}},
        'LSD': 0.0,
        'PSNR': 0.0,
        'SSIM': 0.0,
        'MAE': 0.0,
        'F1': 0.0
    }

    print("Loading VGGish model for L-AUDIO...")
    vggish_model = load_vggish_model()

    print("Loading PANNs model for L-MUSIC...")
    panns_model = load_panns_model()
    print("PANNs model loaded successfully")

    print("Loading Inception model for L-MUSIC features...")
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()
    inception_model.eval().to(Config.device)

    real_files = sorted([f for f in os.listdir(Config.real_audio_dir) if f.endswith(".wav")])
    gen_files = sorted([f for f in os.listdir(gen_audio_dir) if f.endswith(".wav")])

    min_files = min(len(real_files), len(gen_files), Config.max_files)
    if min_files == 0:
        print("错误：没有找到音频文件！")
        return

    real_files = real_files[:min_files]
    gen_files = gen_files[:min_files]

    print(f"Processing {min_files} paired audio files...")

    lsd_scores, psnr_scores, ssim_scores, mae_scores, f1_scores = [], [], [], [], []
    real_mels, gen_mels = [], []
    real_vggish_features, gen_vggish_features = [], []

    for i in tqdm(range(min_files), desc="Processing audio pairs"):
        real_audio = load_audio(os.path.join(Config.real_audio_dir, real_files[i]))
        gen_audio = load_audio(os.path.join(gen_audio_dir, gen_files[i]))

        lsd_scores.append(calculate_lsd(real_audio, gen_audio))
        psnr_scores.append(calculate_psnr(real_audio, gen_audio))
        ssim_scores.append(calculate_ssim(real_audio, gen_audio))
        mae_scores.append(calculate_mae(real_audio, gen_audio))
        f1_scores.append(calculate_f1_score(real_audio, gen_audio))

        real_mel = compute_mel(real_audio)
        gen_mel = compute_mel(gen_audio)

        min_frames = min(real_mel.shape[1], gen_mel.shape[1])
        if min_frames > 0:
            real_mels.append(real_mel[:, :min_frames])
            gen_mels.append(gen_mel[:, :min_frames])

        real_examples = vggish_input.waveform_to_examples(real_audio, Config.sample_rate)
        gen_examples = vggish_input.waveform_to_examples(gen_audio, Config.sample_rate)

        if len(real_examples) > 0:
            real_tensor = torch.tensor(real_examples).to(Config.device)
            with torch.no_grad():
                real_features = vggish_model(real_tensor).cpu().numpy()
            real_vggish_features.append(real_features)

        if len(gen_examples) > 0:
            gen_tensor = torch.tensor(gen_examples).to(Config.device)
            with torch.no_grad():
                gen_features = vggish_model(gen_tensor).cpu().numpy()
            gen_vggish_features.append(gen_features)

    results['LSD'] = np.mean(lsd_scores) if lsd_scores else 0.0
    results['PSNR'] = np.mean(psnr_scores) if psnr_scores else 0.0
    results['SSIM'] = np.mean(ssim_scores) if ssim_scores else 0.0
    results['MAE'] = np.mean(mae_scores) if mae_scores else 0.0
    results['F1'] = np.mean(f1_scores) if f1_scores else 0.0

    if real_vggish_features and gen_vggish_features:
        real_vggish = np.vstack(real_vggish_features)
        gen_vggish = np.vstack(gen_vggish_features)

        results['FAD']['L-AUDIO'] = calculate_fad(real_vggish, gen_vggish)
        results['ISC']['L-AUDIO'] = calculate_isc(gen_vggish)
        results['KID']['L-AUDIO'] = calculate_kid(real_vggish, gen_vggish)

        min_samples = min(real_vggish.shape[0], gen_vggish.shape[0])
        if min_samples > 0:
            real_idx = np.random.choice(real_vggish.shape[0], min_samples, replace=False)
            gen_idx = np.random.choice(gen_vggish.shape[0], min_samples, replace=False)

            real_sub = real_vggish[real_idx]
            gen_sub = gen_vggish[gen_idx]

            real_sigmoid = torch.sigmoid(torch.tensor(real_sub)).numpy()
            gen_sigmoid = torch.sigmoid(torch.tensor(gen_sub)).numpy()
            results['KL']['L-AUDIO']['KL(sigmoid)'] = calculate_kl_divergence(real_sigmoid, gen_sigmoid)

            real_softmax = F.softmax(torch.tensor(real_sub), dim=1).numpy()
            gen_softmax = F.softmax(torch.tensor(gen_sub), dim=1).numpy()
            results['KL']['L-AUDIO']['KL(softmax)'] = calculate_kl_divergence(real_softmax, gen_softmax)
        else:
            results['KL']['L-AUDIO']['KL(sigmoid)'] = 0.0
            results['KL']['L-AUDIO']['KL(softmax)'] = 0.0

    if real_mels and gen_mels:
        real_inception_features = extract_inception_features(real_mels, inception_model)
        gen_inception_features = extract_inception_features(gen_mels, inception_model)

        results['ISC']['L-MUSIC'] = calculate_isc(gen_inception_features)
        results['KID']['L-MUSIC'] = calculate_kid(real_inception_features, gen_inception_features)

        min_samples = min(real_inception_features.shape[0], gen_inception_features.shape[0])
        if min_samples > 0:
            real_idx = np.random.choice(real_inception_features.shape[0], min_samples, replace=False)
            gen_idx = np.random.choice(gen_inception_features.shape[0], min_samples, replace=False)

            real_sub = real_inception_features[real_idx]
            gen_sub = gen_inception_features[gen_idx]

            real_sigmoid = torch.sigmoid(torch.tensor(real_sub)).numpy()
            gen_sigmoid = torch.sigmoid(torch.tensor(gen_sub)).numpy()
            results['KL']['L-MUSIC']['KL(sigmoid)'] = calculate_kl_divergence(real_sigmoid, gen_sigmoid)

            real_softmax = F.softmax(torch.tensor(real_sub), dim=1).numpy()
            gen_softmax = F.softmax(torch.tensor(gen_sub), dim=1).numpy()
            results['KL']['L-MUSIC']['KL(softmax)'] = calculate_kl_divergence(real_softmax, gen_softmax)
        else:
            results['KL']['L-MUSIC']['KL(sigmoid)'] = 0.0
            results['KL']['L-MUSIC']['KL(softmax)'] = 0.0

    print("\nExtracting L-MUSIC features for real audio...")
    real_music_emb = extract_panns_embeddings(Config.real_audio_dir, panns_model)
    print(f"Real L-MUSIC embeddings shape: {real_music_emb.shape}")

    print("Extracting L-MUSIC features for generated audio...")
    gen_music_emb = extract_panns_embeddings(gen_audio_dir, panns_model)
    print(f"Generated L-MUSIC embeddings shape: {gen_music_emb.shape}")

    print("\nApplying PCA to L-MUSIC features...")
    real_music_reduced, pca_model = apply_pca(real_music_emb, Config.target_dim)
    gen_music_reduced, _ = apply_pca(gen_music_emb, Config.target_dim, pca_model)

    print(f"Real L-MUSIC after PCA: {real_music_reduced.shape}")
    print(f"Generated L-MUSIC after PCA: {gen_music_reduced.shape}")

    results['FAD']['L-MUSIC'] = calculate_fad(real_music_reduced, gen_music_reduced)

    print("\n" + "=" * 50)
    print(f"Epoch results:")
    print(f"FAD:{{'L-AUDIO':{results['FAD']['L-AUDIO']:.4f}, 'L-MUSIC':{results['FAD']['L-MUSIC']:.4f}}}")
    print(f"ISC:{{'L-AUDIO': {results['ISC']['L-AUDIO']:.4f}, 'L-MUSIC': {results['ISC']['L-MUSIC']:.4f}}}")
    print(
        f"KL:{{'L-AUDIO': {{'KL(sigmoid)': {results['KL']['L-AUDIO']['KL(sigmoid)']:.4f}, 'KL(softmax)': {results['KL']['L-AUDIO']['KL(softmax)']:.4f}}}, 'L-MUSIC': {{'KL(sigmoid)': {results['KL']['L-MUSIC']['KL(sigmoid)']:.4f}, 'KL(softmax)': {results['KL']['L-MUSIC']['KL(softmax)']:.4f}}}}}")
    print(f"LSD:{results['LSD']:.4f}")
    print(f"PSNR:{results['PSNR']:.4f}")
    print(f"SSIM:{results['SSIM']:.4f}")
    print(f"MAE:{results['MAE']:.4f}")
    print(f"F1:{results['F1']:.4f}")
    print("\n" + "=" * 50)

    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    results_converted = convert_numpy_types(results)

    try:
        save_dir = os.path.dirname(result_save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        with open(result_save_path, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, ensure_ascii=False, indent=4)
        print(f"评估结果已成功保存到: {result_save_path}")
    except Exception as e:
        print(f"保存评估结果失败: {str(e)}")


def main():
    # 定义需要处理的epoch列表，这里设置为1000、2000、3000，可以根据需要修改
    epochs = range(1000, 10000, 1000)  # 生成1000,2000,3000

    for epoch in epochs:
        print(f"\n{'=' * 20} 开始处理 {epoch} epoch {'=' * 20}")
        # 动态生成路径
        gen_audio_dir = f"./inference_wav/{epoch}_epoch"
        result_save_path = f"./evaluation_results/evaluation_results_{epoch}.json"

        # 处理当前epoch
        process_epoch(gen_audio_dir, result_save_path)
        print(f"{'=' * 20} {epoch} epoch 处理完成 {'=' * 20}\n")


if __name__ == "__main__":
    main()