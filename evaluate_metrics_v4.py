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
    f1_threshold = 0.01
    # 增加数值稳定性参数
    frechet_eps = 1e-6  # Frechet距离计算的正则项
    kl_eps = 1e-10  # KL散度计算的epsilon


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

    try:
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
    except Exception as e:
        print(f"Mel计算错误: {str(e)}")
        return np.zeros((128, 10))


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


# 修复1: 增强数值稳定性的Frechet距离计算 - 修复协方差矩阵虚部问题
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=config.frechet_eps):
    """改进版Frechet距离计算，增加数值稳定性"""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    # 确保协方差矩阵是对称的
    sigma1 = (sigma1 + sigma1.T) / 2
    sigma2 = (sigma2 + sigma2.T) / 2

    # 添加正则项确保协方差矩阵正定
    offset = np.eye(sigma1.shape[0]) * eps
    sigma1 += offset
    sigma2 += offset

    diff = mu1 - mu2

    # 使用特征值分解计算平方根，避免复数问题
    def sqrtm_psd(matrix):
        # 对称矩阵特征值分解
        w, v = np.linalg.eigh(matrix)
        # 确保特征值非负
        w = np.maximum(w, 0)
        # 计算矩阵平方根
        return v @ np.diag(np.sqrt(w)) @ v.T

    # 计算几何平均的平方根
    sqrt_sigma1 = sqrtm_psd(sigma1)
    covmean = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    covmean = sqrtm_psd(covmean)

    tr_covmean = np.trace(covmean)

    # 计算距离并确保非负
    distance = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return max(distance, 0)  # 确保非负


# 修复2: 修复KID计算（增加特征归一化）
def calculate_kid(real_activations, gen_activations, max_size=100):
    n_samples = min(len(real_activations), len(gen_activations), max_size)
    if n_samples < 2:
        return 0.0

    real_sub = real_activations[np.random.choice(len(real_activations), n_samples, replace=False)]
    gen_sub = gen_activations[np.random.choice(len(gen_activations), n_samples, replace=False)]

    # 添加特征归一化
    real_sub = (real_sub - real_sub.mean(0)) / (real_sub.std(0) + 1e-10)
    gen_sub = (gen_sub - gen_sub.mean(0)) / (gen_sub.std(0) + 1e-10)

    kernel_real = np.dot(real_sub, real_sub.T)
    kernel_gen = np.dot(gen_sub, gen_sub.T)
    kernel_cross = np.dot(real_sub, gen_sub.T)

    kid = np.mean(kernel_real) + np.mean(kernel_gen) - 2 * np.mean(kernel_cross)
    return max(kid, 0)  # 确保非负


# 修复3: 修复KL散度计算
def calculate_kl_divergence(real_activations, gen_activations):
    # 添加epsilon防止除零错误
    eps = config.kl_eps

    # 确保概率分布有效
    P = real_activations / (real_activations.sum(1, keepdims=True) + eps)
    Q = gen_activations / (gen_activations.sum(1, keepdims=True) + eps)

    # 添加clip防止log(0)
    P = np.clip(P, eps, 1 - eps)
    Q = np.clip(Q, eps, 1 - eps)

    # 按元素计算避免数值溢出
    log_ratio = np.log(P / (Q + eps) + eps)
    kl_values = P * log_ratio

    # 返回平均KL散度，跳过NaN值
    return np.nanmean(kl_values)


# 修复4: 修复ISC计算（避免NaN）
def calculate_isc(features):
    if features.size == 0:
        return 0.0

    features_tensor = torch.tensor(features).float()
    # 增加数值稳定性处理
    features_tensor = features_tensor - features_tensor.max(dim=1, keepdim=True)[0]
    py = F.softmax(features_tensor, dim=1).mean(dim=0)

    scores = []
    for i in range(features_tensor.shape[0]):
        pyx = F.softmax(features_tensor[i], dim=0)
        # 添加epsilon避免log(0)
        kl = F.kl_div((py + 1e-10).log(), pyx + 1e-10, reduction='sum').item()
        scores.append(kl)

    # 处理全零特征的情况
    mean_score = np.mean(scores) if scores else 0.0
    return np.exp(mean_score)


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
    if len(audio) == 0:
        return np.zeros((0, 128))

    try:
        if sr != config.vggish_sample_rate:
            audio = resampy.resample(audio, sr, config.vggish_sample_rate)

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        log_mel = vggish_input.waveform_to_examples(audio, config.vggish_sample_rate)

        if len(log_mel) == 0:
            return np.zeros((0, 128))

        device = next(vggish_model.parameters()).device
        log_mel_tensor = torch.tensor(log_mel).float().to(device)

        if log_mel_tensor.dim() == 3:
            log_mel_tensor = log_mel_tensor.unsqueeze(1)

        features_list = []
        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(log_mel_tensor), batch_size):
                batch = log_mel_tensor[i:i + batch_size].to(device)
                features = vggish_model(batch).cpu()

                # === 修复开始 ===
                # 自适应维度归一化
                if features.dim() == 1:
                    features = F.normalize(features, p=2, dim=0)
                elif features.dim() >= 2:
                    features = F.normalize(features, p=2, dim=1)
                # === 修复结束 ===

                if features.dim() == 1:
                    features = features.unsqueeze(0)
                features_list.append(features)

        if features_list:
            return torch.cat(features_list, dim=0).numpy()
    except Exception as e:
        print(f"VGGish特征提取错误: {str(e)}")

    return np.zeros((0, 128))


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
    mae_scores = []
    f1_scores = []

    for i in tqdm(range(len(real_audios)), desc="音频指标"):
        lsd_scores.append(calculate_lsd(real_audios[i], gen_audios[i]))
        psnr_scores.append(calculate_psnr(real_audios[i], gen_audios[i]))
        ssim_scores.append(calculate_ssim(real_audios[i], gen_audios[i]))
        mae_scores.append(calculate_mae(real_audios[i], gen_audios[i]))
        f1_scores.append(calculate_f1_score(real_audios[i], gen_audios[i]))

    results['LSD'] = np.mean(lsd_scores) if lsd_scores else 0.0
    results['PSNR'] = np.mean(psnr_scores) if psnr_scores else 0.0
    results['SSIM'] = np.mean(ssim_scores) if ssim_scores else 0.0
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

    # 过滤空特征
    real_features_audio = [f for f in real_features_audio if f.size > 0 and f.shape[0] > 0]
    gen_features_audio = [f for f in gen_features_audio if f.size > 0 and f.shape[0] > 0]

    if not real_features_audio or not gen_features_audio:
        print("警告: VGGish特征提取失败，跳过L-AUDIO指标计算")
    else:
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

            # 确保协方差矩阵对称
            sigma_real_audio = (sigma_real_audio + sigma_real_audio.T) / 2
            sigma_gen_audio = (sigma_gen_audio + sigma_gen_audio.T) / 2

            # ========== 移植v2版本的FAD计算 ==========
            # 只计算FAD (L-AUDIO)
            results['FAD']['L-AUDIO'] = calculate_frechet_distance(
                mu_real_audio, sigma_real_audio,
                mu_gen_audio, sigma_gen_audio
            )
            # ========== 移植结束 ==========

            results['ISC']['L-AUDIO'] = calculate_isc(gen_audio_feats)
            results['KID']['L-AUDIO'] = calculate_kid(real_audio_feats, gen_audio_feats)

            # KL计算（添加异常处理）
            try:
                real_sigmoid = torch.sigmoid(torch.tensor(real_audio_feats)).numpy()
                gen_sigmoid = torch.sigmoid(torch.tensor(gen_audio_feats)).numpy()

                # 归一化概率分布
                real_sigmoid = real_sigmoid / (real_sigmoid.sum(1, keepdims=True) + config.kl_eps)
                gen_sigmoid = gen_sigmoid / (gen_sigmoid.sum(1, keepdims=True) + config.kl_eps)

                # 防止NaN值
                real_sigmoid = np.clip(real_sigmoid, config.kl_eps, 1 - config.kl_eps)
                gen_sigmoid = np.clip(gen_sigmoid, config.kl_eps, 1 - config.kl_eps)

                kl_sigmoid = calculate_kl_divergence(real_sigmoid, gen_sigmoid)
            except Exception as e:
                print(f"KL(sigmoid)计算错误: {str(e)}")
                kl_sigmoid = float('nan')

            try:
                real_softmax = F.softmax(torch.tensor(real_audio_feats), dim=1).numpy()
                gen_softmax = F.softmax(torch.tensor(gen_audio_feats), dim=1).numpy()

                # 归一化概率分布
                real_softmax = real_softmax / (real_softmax.sum(1, keepdims=True) + config.kl_eps)
                gen_softmax = gen_softmax / (gen_softmax.sum(1, keepdims=True) + config.kl_eps)

                # 防止NaN值
                real_softmax = np.clip(real_softmax, config.kl_eps, 1 - config.kl_eps)
                gen_softmax = np.clip(gen_softmax, config.kl_eps, 1 - config.kl_eps)

                kl_softmax = calculate_kl_divergence(real_softmax, gen_softmax)
            except Exception as e:
                print(f"KL(softmax)计算错误: {str(e)}")
                kl_softmax = float('nan')

            results['KL']['L-AUDIO'] = {
                'KL(sigmoid)': kl_sigmoid,
                'KL(softmax)': kl_softmax
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
    py = np.mean(gen_all_probs, axis=0) + config.kl_eps
    scores = []
    for i in range(gen_all_probs.shape[0]):
        pyx = gen_all_probs[i] + config.kl_eps
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
        try:
            mu_real_music = np.mean(real_music_feats, axis=0)
            sigma_real_music = np.cov(real_music_feats, rowvar=False)
            mu_gen_music = np.mean(gen_music_feats, axis=0)
            sigma_gen_music = np.cov(gen_music_feats, rowvar=False)

            # 确保协方差矩阵对称
            sigma_real_music = (sigma_real_music + sigma_real_music.T) / 2
            sigma_gen_music = (sigma_gen_music + sigma_gen_music.T) / 2

            # ========== 移植v2版本的FID计算 ==========
            # 只计算FID (L-MUSIC)
            results['FID']['L-MUSIC'] = calculate_frechet_distance(
                mu_real_music, sigma_real_music,
                mu_gen_music, sigma_gen_music
            )
            # ========== 移植结束 ==========

            results['KID']['L-MUSIC'] = calculate_kid(real_music_feats, gen_music_feats)

            # KL计算（添加异常处理）
            try:
                real_sigmoid = torch.sigmoid(torch.tensor(real_music_feats)).numpy()
                gen_sigmoid = torch.sigmoid(torch.tensor(gen_music_feats)).numpy()

                # 归一化概率分布
                real_sigmoid = real_sigmoid / (real_sigmoid.sum(1, keepdims=True) + config.kl_eps)
                gen_sigmoid = gen_sigmoid / (gen_sigmoid.sum(1, keepdims=True) + config.kl_eps)

                # 防止NaN值
                real_sigmoid = np.clip(real_sigmoid, config.kl_eps, 1 - config.kl_eps)
                gen_sigmoid = np.clip(gen_sigmoid, config.kl_eps, 1 - config.kl_eps)

                kl_sigmoid = calculate_kl_divergence(real_sigmoid, gen_sigmoid)
            except Exception as e:
                print(f"KL(sigmoid)计算错误: {str(e)}")
                kl_sigmoid = float('nan')

            try:
                real_softmax = F.softmax(torch.tensor(real_music_feats), dim=1).numpy()
                gen_softmax = F.softmax(torch.tensor(gen_music_feats), dim=1).numpy()

                # 归一化概率分布
                real_softmax = real_softmax / (real_softmax.sum(1, keepdims=True) + config.kl_eps)
                gen_softmax = gen_softmax / (gen_softmax.sum(1, keepdims=True) + config.kl_eps)

                # 防止NaN值
                real_softmax = np.clip(real_softmax, config.kl_eps, 1 - config.kl_eps)
                gen_softmax = np.clip(gen_softmax, config.kl_eps, 1 - config.kl_eps)

                kl_softmax = calculate_kl_divergence(real_softmax, gen_softmax)
            except Exception as e:
                print(f"KL(softmax)计算错误: {str(e)}")
                kl_softmax = float('nan')

            results['KL']['L-MUSIC'] = {
                'KL(sigmoid)': kl_sigmoid,
                'KL(softmax)': kl_softmax
            }
        except Exception as e:
            print(f"计算L-MUSIC的FID时出现错误: {str(e)}")
            results['FID']['L-MUSIC'] = float('nan')

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