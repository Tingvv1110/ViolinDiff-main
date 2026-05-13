import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3
from sklearn.metrics import f1_score, mean_absolute_error
import soundfile as sf
from tqdm import tqdm
import argparse
import warnings
import resampy
import torchaudio
import torchaudio.transforms as T
from torchvggish import vggish, vggish_input
import sys

warnings.filterwarnings("ignore")


class Config:
    sample_rate = 22050
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vggish_sample_rate = 16000
    # 添加F1-score的阈值参数
    f1_threshold = 0.01  # 用于二值化音频信号的阈值


config = Config()


def load_audio(file_path, target_sr=None):
    audio, sr = sf.read(file_path, always_2d=True)
    if audio.shape[1] > 1:
        audio = np.mean(audio, axis=1)
    else:
        audio = audio.flatten()

    if target_sr and sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
    elif not target_sr and sr != config.sample_rate:
        audio = resampy.resample(audio, sr, config.sample_rate)

    return audio.astype(np.float32)


def compute_mel(audio, sr=config.sample_rate):
    if len(audio) == 0:
        return np.zeros((128, 10))

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=128,
        power=1.0
    )
    return librosa.power_to_db(mel, ref=np.max)


def compute_stft(audio):
    if len(audio) == 0:
        return np.zeros((config.n_fft // 2 + 1, 10))

    stft = librosa.stft(
        audio,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length
    )
    magnitude = np.abs(stft)
    return magnitude


def calculate_mae(audio_real, audio_gen):
    min_len = min(len(audio_real), len(audio_gen))
    if min_len == 0:
        return 0.0

    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]
    return mean_absolute_error(audio_real, audio_gen)


def calculate_f1_score(audio_real, audio_gen, threshold=config.f1_threshold):
    min_len = min(len(audio_real), len(audio_gen))
    if min_len == 0:
        return 0.0

    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]

    # 将音频信号二值化（大于阈值为1，小于等于阈值为0）
    bin_real = (np.abs(audio_real) > threshold).astype(int)
    bin_gen = (np.abs(audio_gen) > threshold).astype(int)

    # 计算F1-score
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


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_kid(real_activations, gen_activations, max_size=100):
    n_samples = min(len(real_activations), len(gen_activations), max_size)
    if n_samples < 2:
        return 0.0

    real_sub = real_activations[np.random.choice(len(real_activations), n_samples, replace=False)]
    gen_sub = gen_activations[np.random.choice(len(gen_activations), n_samples, replace=False)]

    kernel_real = np.dot(real_sub, real_sub.T)
    kernel_gen = np.dot(gen_sub, gen_sub.T)
    kernel_cross = np.dot(real_sub, gen_sub.T)

    kid = np.mean(kernel_real) + np.mean(kernel_gen) - 2 * np.mean(kernel_cross)
    return kid


def calculate_kl_divergence(real_activations, gen_activations):
    P = real_activations + 1e-10
    Q = gen_activations + 1e-10
    kl = np.sum(P * np.log(P / Q))
    return kl if np.isfinite(kl) and kl >= 0 else 0.0


def calculate_isc(features):
    features_tensor = torch.tensor(features).float()
    py = F.softmax(features_tensor, dim=1).mean(dim=0)

    scores = []
    for i in range(features_tensor.shape[0]):
        pyx = F.softmax(features_tensor[i], dim=0)
        kl = F.kl_div(py.log(), pyx, reduction='sum').item()
        scores.append(kl)

    return np.exp(np.mean(scores))


class VGGishWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vggish()
        self.model.eval()

        # 将PCA参数转换为张量并保持在同一设备
        if hasattr(self.model, 'pproc') and self.model.pproc is not None:
            self.model.pproc._pca_means = torch.tensor(
                self.model.pproc._pca_means,
                dtype=torch.float32
            )
            self.model.pproc._pca_matrix = torch.tensor(
                self.model.pproc._pca_matrix,
                dtype=torch.float32
            )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 确保PCA参数与输入在同一设备
        if hasattr(self.model, 'pproc') and self.model.pproc is not None:
            device = x.device
            self.model.pproc._pca_means = self.model.pproc._pca_means.to(device)
            self.model.pproc._pca_matrix = self.model.pproc._pca_matrix.to(device)

        return self.model(x)


def extract_vggish_features(audio, sr=config.sample_rate):
    if sr != config.vggish_sample_rate:
        audio = resampy.resample(audio, sr, config.vggish_sample_rate)

    log_mel = vggish_input.waveform_to_examples(audio, config.vggish_sample_rate)

    # 确保设备一致性
    device = next(vggish_model.parameters()).device
    log_mel_tensor = torch.tensor(log_mel).float().to(device)

    # 确保4D输入 [batch, channel, height, width]
    if log_mel_tensor.dim() == 3:
        log_mel_tensor = log_mel_tensor.unsqueeze(1)

    features_list = []
    with torch.no_grad():
        batch_size = 32
        for i in range(0, len(log_mel_tensor), batch_size):
            batch = log_mel_tensor[i:i + batch_size]

            # 确保输入在模型设备上
            batch = batch.to(device)
            features = vggish_model(batch)
            features_list.append(features.cpu())

    return torch.cat(features_list, dim=0).numpy()


def evaluate_metrics(real_audio_dir, gen_audio_dir):
    results = {
        'FAD': {'L-AUDIO': 0.0, 'L-MUSIC': 0.0},
        'FID': {'L-AUDIO': 0.0, 'L-MUSIC': 0.0},
        'ISC': {'L-AUDIO': 0.0, 'L-MUSIC': 0.0},
        'KID': {'L-AUDIO': 0.0, 'L-MUSIC': 0.0},
        'KL': {'L-AUDIO': {'KL(sigmoid)': 0.0, 'KL(softmax)': 0.0},
               'L-MUSIC': {'KL(sigmoid)': 0.0, 'KL(softmax)': 0.0}},
        'LSD': 0.0,
        'PSNR': 0.0,
        'SSIM': 0.0,
        # 添加MAE和F1-score
        'MAE': 0.0,
        'F1': 0.0
    }

    real_files = sorted([os.path.join(real_audio_dir, f) for f in os.listdir(real_audio_dir) if f.endswith('.wav')])
    gen_files = sorted([os.path.join(gen_audio_dir, f) for f in os.listdir(gen_audio_dir) if f.endswith('.wav')])

    min_files = min(len(real_files), len(gen_files))
    if min_files == 0:
        print("错误：没有找到音频文件！")
        return results

    real_files = real_files[:min_files]
    gen_files = gen_files[:min_files]

    real_audios = []
    gen_audios = []
    real_mels = []
    gen_mels = []

    print(f"找到 {min_files} 个音频文件对")
    print("提取特征...")

    for i in tqdm(range(min_files), desc="处理音频"):
        real_audio = load_audio(real_files[i])
        gen_audio = load_audio(gen_files[i])

        min_len = min(len(real_audio), len(gen_audio))
        real_audio = real_audio[:min_len]
        gen_audio = gen_audio[:min_len]

        real_audios.append(real_audio)
        gen_audios.append(gen_audio)

        real_mel = compute_mel(real_audio)
        gen_mel = compute_mel(gen_audio)

        min_frames = min(real_mel.shape[1], gen_mel.shape[1])
        real_mel = real_mel[:, :min_frames]
        gen_mel = gen_mel[:, :min_frames]

        real_mels.append(real_mel)
        gen_mels.append(gen_mel)

    if len(real_audios) == 0:
        print("错误：没有成功处理任何音频文件")
        return results

    print("\n计算音频级指标...")
    lsd_scores = []
    psnr_scores = []
    ssim_scores = []
    mae_scores = []  # 新增MAE分数列表
    f1_scores = []  # 新增F1分数列表

    for i in tqdm(range(len(real_audios)), desc="音频指标"):
        lsd_scores.append(calculate_lsd(real_audios[i], gen_audios[i]))
        psnr_scores.append(calculate_psnr(real_audios[i], gen_audios[i]))
        ssim_scores.append(calculate_ssim(real_audios[i], gen_audios[i]))
        # 新增MAE和F1-score计算
        mae_scores.append(calculate_mae(real_audios[i], gen_audios[i]))
        f1_scores.append(calculate_f1_score(real_audios[i], gen_audios[i]))

    results['LSD'] = np.mean(lsd_scores) if lsd_scores else 0.0
    results['PSNR'] = np.mean(psnr_scores) if psnr_scores else 0.0
    results['SSIM'] = np.mean(ssim_scores) if ssim_scores else 0.0
    # 添加MAE和F1-score结果
    results['MAE'] = np.mean(mae_scores) if mae_scores else 0.0
    results['F1'] = np.mean(f1_scores) if f1_scores else 0.0

    print("\n提取VGGish特征 (L-AUDIO)...")
    real_features_audio = []
    gen_features_audio = []

    for i in tqdm(range(len(real_audios)), desc="VGGish特征"):
        real_feat = extract_vggish_features(real_audios[i])
        gen_feat = extract_vggish_features(gen_audios[i])

        if real_feat.size > 0 and gen_feat.size > 0:
            real_features_audio.append(real_feat)
            gen_features_audio.append(gen_feat)

    real_audio_feats = np.concatenate(real_features_audio)
    gen_audio_feats = np.concatenate(gen_features_audio)

    if real_audio_feats.ndim == 1:
        real_audio_feats = real_audio_feats.reshape(-1, 1)
    if gen_audio_feats.ndim == 1:
        gen_audio_feats = gen_audio_feats.reshape(-1, 1)

    if real_audio_feats.shape[0] > 1 and gen_audio_feats.shape[0] > 1:
        mu_real_audio = np.mean(real_audio_feats, axis=0)
        sigma_real_audio = np.cov(real_audio_feats, rowvar=False)
        mu_gen_audio = np.mean(gen_audio_feats, axis=0)
        sigma_gen_audio = np.cov(gen_audio_feats, rowvar=False)

        results['FAD']['L-AUDIO'] = calculate_frechet_distance(
            mu_real_audio, sigma_real_audio,
            mu_gen_audio, sigma_gen_audio
        )
        results['FID']['L-AUDIO'] = results['FAD']['L-AUDIO']
        results['ISC']['L-AUDIO'] = calculate_isc(gen_audio_feats)
        results['KID']['L-AUDIO'] = calculate_kid(real_audio_feats, gen_audio_feats)

        # 简化KL计算
        real_sigmoid = torch.sigmoid(torch.tensor(real_audio_feats)).numpy()
        gen_sigmoid = torch.sigmoid(torch.tensor(gen_audio_feats)).numpy()
        real_softmax = F.softmax(torch.tensor(real_audio_feats), dim=1).numpy()
        gen_softmax = F.softmax(torch.tensor(gen_audio_feats), dim=1).numpy()

        results['KL']['L-AUDIO'] = {
            'KL(sigmoid)': calculate_kl_divergence(real_sigmoid, gen_sigmoid),
            'KL(softmax)': calculate_kl_divergence(real_softmax, gen_softmax)
        }

    print("\n提取Inception概率分布 (L-MUSIC)...")
    real_probs = []
    gen_probs = []

    inception_model_full = inception_v3(pretrained=True, transform_input=False)
    inception_model_full.eval().to(config.device)

    for i in tqdm(range(len(real_mels)), desc="Inception概率"):
        real_mel = real_mels[i]
        gen_mel = gen_mels[i]

        real_img = np.stack([real_mel] * 3, axis=0)
        gen_img = np.stack([gen_mel] * 3, axis=0)

        real_img = torch.FloatTensor(real_img).unsqueeze(0)
        gen_img = torch.FloatTensor(gen_img).unsqueeze(0)

        real_img = F.interpolate(real_img, size=(299, 299), mode='bilinear', align_corners=False)
        gen_img = F.interpolate(gen_img, size=(299, 299), mode='bilinear', align_corners=False)

        # 移动到设备
        real_img = real_img.to(config.device)
        gen_img = gen_img.to(config.device)

        with torch.no_grad():
            real_out = inception_model_full(real_img)
            gen_out = inception_model_full(gen_img)

            real_prob = F.softmax(real_out, dim=1).cpu().numpy()
            gen_prob = F.softmax(gen_out, dim=1).cpu().numpy()

        real_probs.append(real_prob)
        gen_probs.append(gen_prob)

    gen_all_probs = np.concatenate(gen_probs)
    py = np.mean(gen_all_probs, axis=0) + 1e-12
    scores = []
    for i in range(gen_all_probs.shape[0]):
        pyx = gen_all_probs[i] + 1e-12
        kl = np.sum(py * (np.log(py) - np.log(pyx)))
        scores.append(kl)
    results['ISC']['L-MUSIC'] = np.exp(np.mean(scores))

    print("\n提取Inception特征 (L-MUSIC)...")
    real_features_music = []
    gen_features_music = []

    inception_model_feat = inception_v3(pretrained=True, transform_input=False)
    inception_model_feat.fc = torch.nn.Identity()
    inception_model_feat.eval().to(config.device)

    for i in tqdm(range(len(real_mels)), desc="Inception特征"):
        real_mel = real_mels[i]
        gen_mel = gen_mels[i]

        real_img = np.stack([real_mel] * 3, axis=0)
        gen_img = np.stack([gen_mel] * 3, axis=0)

        real_img = torch.FloatTensor(real_img).unsqueeze(0)
        gen_img = torch.FloatTensor(gen_img).unsqueeze(0)

        real_img = F.interpolate(real_img, size=(299, 299), mode='bilinear', align_corners=False)
        gen_img = F.interpolate(gen_img, size=(299, 299), mode='bilinear', align_corners=False)

        # 移动到设备
        real_img = real_img.to(config.device)
        gen_img = gen_img.to(config.device)

        with torch.no_grad():
            real_feat = inception_model_feat(real_img).cpu().numpy()
            gen_feat = inception_model_feat(gen_img).cpu().numpy()

        real_features_music.append(real_feat)
        gen_features_music.append(gen_feat)

    real_music_feats = np.concatenate(real_features_music)
    gen_music_feats = np.concatenate(gen_features_music)

    if real_music_feats.ndim == 1:
        real_music_feats = real_music_feats.reshape(-1, 1)
    if gen_music_feats.ndim == 1:
        gen_music_feats = gen_music_feats.reshape(-1, 1)

    if real_music_feats.shape[0] > 1 and gen_music_feats.shape[0] > 1:
        mu_real_music = np.mean(real_music_feats, axis=0)
        sigma_real_music = np.cov(real_music_feats, rowvar=False)
        mu_gen_music = np.mean(gen_music_feats, axis=0)
        sigma_gen_music = np.cov(gen_music_feats, rowvar=False)

        results['FAD']['L-MUSIC'] = calculate_frechet_distance(
            mu_real_music, sigma_real_music,
            mu_gen_music, sigma_gen_music
        )
        results['FID']['L-MUSIC'] = results['FAD']['L-MUSIC']
        results['KID']['L-MUSIC'] = calculate_kid(real_music_feats, gen_music_feats)

        # 简化KL计算
        real_sigmoid = torch.sigmoid(torch.tensor(real_music_feats)).numpy()
        gen_sigmoid = torch.sigmoid(torch.tensor(gen_music_feats)).numpy()
        real_softmax = F.softmax(torch.tensor(real_music_feats), dim=1).numpy()
        gen_softmax = F.softmax(torch.tensor(gen_music_feats), dim=1).numpy()

        results['KL']['L-MUSIC'] = {
            'KL(sigmoid)': calculate_kl_divergence(real_sigmoid, gen_sigmoid),
            'KL(softmax)': calculate_kl_divergence(real_softmax, gen_softmax)
        }

    return results


if __name__ == "__main__":
    global vggish_model
    vggish_model = VGGishWrapper().to(config.device)
    vggish_model.eval()
    print("VGGish模型加载成功")

    parser = argparse.ArgumentParser(description='音频生成质量评估')
    parser.add_argument('--real_dir', type=str, required=True, help='真实音频目录')
    parser.add_argument('--gen_dir', type=str, required=True, help='生成音频目录')
    parser.add_argument('--output', type=str, default='evaluation_results.txt', help='输出结果文件')
    # 添加F1-score阈值参数
    parser.add_argument('--f1_threshold', type=float, default=0.01, help='F1-score计算使用的阈值')

    args = parser.parse_args()

    # 更新F1-score阈值
    config.f1_threshold = args.f1_threshold

    print(f"评估设置:")
    print(f"  真实音频目录: {args.real_dir}")
    print(f"  生成音频目录: {args.gen_dir}")
    print(f"  输出文件: {args.output}")
    print(f"  F1-score阈值: {config.f1_threshold}")

    metrics = evaluate_metrics(args.real_dir, args.gen_dir)

    print("\n评估结果:")
    print("FAD:", metrics['FAD'])
    print("FID:", metrics['FID'])
    print("ISC:", metrics['ISC'])
    print("KID:", metrics['KID'])
    print("KL:", metrics['KL'])
    print(f"LSD: {metrics['LSD']:.4f}")
    print(f"PSNR: {metrics['PSNR']:.4f}")
    print(f"SSIM: {metrics['SSIM']:.4f}")
    # 输出新增的MAE和F1-score
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"F1-score: {metrics['F1']:.4f}")

    with open(args.output, 'w') as f:
        f.write("Experimental Results\n\n")
        f.write("| Key | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| FAD | {metrics['FAD']} |\n")
        f.write(f"| FID | {metrics['FID']} |\n")
        f.write(f"| ISC | {metrics['ISC']} |\n")
        f.write(f"| KID | {metrics['KID']} |\n")
        f.write(f"| KL | {metrics['KL']} |\n")
        f.write(f"| LSD | {metrics['LSD']:.4f} |\n")
        f.write(f"| PSNR | {metrics['PSNR']:.4f} |\n")
        f.write(f"| SSIM | {metrics['SSIM']:.4f} |\n")
        # 保存新增的MAE和F1-score
        f.write(f"| MAE | {metrics['MAE']:.4f} |\n")
        f.write(f"| F1-score | {metrics['F1']:.4f} |\n")

    print(f"\n结果已保存到 {args.output}")