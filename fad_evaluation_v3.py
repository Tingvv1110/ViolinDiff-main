import os
import numpy as np
import torch
import torchaudio
from scipy import linalg
from sklearn.decomposition import PCA
from harritaylor_torchvggish_master.torchvggish import vggish_input
from harritaylor_torchvggish_master.torchvggish.vggish import VGGish
from panns_inference import AudioTagging


# 配置参数
class Config:
    real_audio_dir = "/root/autodl-tmp/ViolinDiff/true_wav_dir"  # 真实音频目录
    gen_audio_dir = "/root/autodl-tmp/ViolinDiff/test_wav_dir"  # 生成音频目录
    checkpoint_dir = "./checkpoints"  # 模型权重目录
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16  # 批大小
    sample_rate = 16000  # 音频采样率
    segment_duration = 3.0  # 音频分段时长（秒）
    target_dim = 128  # 统一的目标特征维度


def load_vggish_model():
    """加载VGGish模型用于L-AUDIO"""
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
    """加载PANNs模型用于L-MUSIC"""
    os.makedirs("/root/panns_data", exist_ok=True)
    model = AudioTagging(checkpoint_path=None, device=str(Config.device))
    return model


def extract_vggish_embeddings(audio_dir, model):
    """使用VGGish提取L-AUDIO特征"""
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
    """使用PANNs模型提取L-MUSIC特征"""
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

            # 分割音频片段
            segments = []
            for start in range(0, waveform.shape[1], segment_samples):
                end = start + segment_samples
                if end > waveform.shape[1]:
                    break
                segment = waveform[:, start:end]

                # 确保单声道并转换为numpy
                if segment.dim() == 2 and segment.shape[0] > 1:
                    segment = segment.mean(dim=0)
                segment = segment.numpy().astype(np.float32)
                segment /= np.max(np.abs(segment)) + 1e-8

                # 确保输入是二维数组 [batch_size, audio_length]
                if segment.ndim == 1:
                    segment = segment[np.newaxis, :]  # 添加批处理维度

                # 提取特征
                try:
                    _, embedding = model.inference(segment)
                    all_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error extracting features from {audio_path}: {str(e)}")

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")

    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def apply_pca(embeddings, n_components, fit_pca=None):
    """应用PCA降维并返回降维后的特征和PCA模型"""
    if embeddings.size == 0:
        return np.array([]), None

    # 如果提供了预训练的PCA模型，直接使用它进行转换
    if fit_pca is not None:
        return fit_pca.transform(embeddings), fit_pca

    # 否则创建新的PCA模型并拟合数据
    pca = PCA(n_components=n_components, svd_solver='full')
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings, pca


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算两个多元高斯分布之间的Fréchet距离"""
    diff = mu1 - mu2

    # 添加正则化项确保协方差矩阵是正定的
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    # 计算协方差矩阵的几何平均
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # 检查复数部分
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算距离公式
    distance = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    # 确保距离非负
    return max(distance, 0)


def calculate_fad(real_emb, gen_emb):
    """计算FAD分数"""
    if real_emb.size == 0 or gen_emb.size == 0:
        return float('inf')

    mu_real = np.mean(real_emb, axis=0)
    sigma_real = np.cov(real_emb, rowvar=False)

    mu_gen = np.mean(gen_emb, axis=0)
    sigma_gen = np.cov(gen_emb, rowvar=False)

    # 确保协方差矩阵是正定的
    min_eig_real = np.min(np.real(np.linalg.eigvals(sigma_real)))
    min_eig_gen = np.min(np.real(np.linalg.eigvals(sigma_gen)))

    # 如果协方差矩阵不是正定的，添加正则化项
    if min_eig_real < 1e-8:
        print("Warning: Real embeddings covariance matrix not positive definite. Adding regularization.")
        sigma_real += np.eye(sigma_real.shape[0]) * 1e-6

    if min_eig_gen < 1e-8:
        print("Warning: Generated embeddings covariance matrix not positive definite. Adding regularization.")
        sigma_gen += np.eye(sigma_gen.shape[0]) * 1e-6

    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def main():
    # 加载模型
    print("Loading VGGish model for L-AUDIO...")
    vggish_model = load_vggish_model()

    print("Loading PANNs model for L-MUSIC...")
    panns_model = load_panns_model()
    print("PANNs model loaded successfully")

    # 提取L-AUDIO特征
    print("\nExtracting L-AUDIO features for real audio...")
    real_audio_emb = extract_vggish_embeddings(Config.real_audio_dir, vggish_model)
    print(f"Real L-AUDIO embeddings shape: {real_audio_emb.shape}")

    print("Extracting L-AUDIO features for generated audio...")
    gen_audio_emb = extract_vggish_embeddings(Config.gen_audio_dir, vggish_model)
    print(f"Generated L-AUDIO embeddings shape: {gen_audio_emb.shape}")

    # 提取L-MUSIC特征
    print("\nExtracting L-MUSIC features for real audio...")
    real_music_emb = extract_panns_embeddings(Config.real_audio_dir, panns_model)
    print(f"Real L-MUSIC embeddings shape: {real_music_emb.shape}")

    print("Extracting L-MUSIC features for generated audio...")
    gen_music_emb = extract_panns_embeddings(Config.gen_audio_dir, panns_model)
    print(f"Generated L-MUSIC embeddings shape: {gen_music_emb.shape}")

    # 对L-MUSIC特征进行PCA降维（使用真实音频拟合PCA）
    print("\nApplying PCA to L-MUSIC features...")
    real_music_reduced, pca_model = apply_pca(real_music_emb, Config.target_dim)
    gen_music_reduced, _ = apply_pca(gen_music_emb, Config.target_dim, pca_model)

    print(f"Real L-MUSIC after PCA: {real_music_reduced.shape}")
    print(f"Generated L-MUSIC after PCA: {gen_music_reduced.shape}")

    # 计算FAD分数
    print("\nCalculating FAD scores...")

    fad_audio = calculate_fad(real_audio_emb, gen_audio_emb)
    print(f"L-AUDIO FAD Score: {fad_audio:.4f}")

    fad_music = calculate_fad(real_music_reduced, gen_music_reduced)
    print(f"L-MUSIC FAD Score: {fad_music:.4f}")

    # 输出最终结果
    print("\n" + "=" * 50)
    print(f"L-AUDIO FAD Score: {fad_audio:.4f}")
    print(f"L-MUSIC FAD Score: {fad_music:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()