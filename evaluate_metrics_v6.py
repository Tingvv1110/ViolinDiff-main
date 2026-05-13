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
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import sys

# 忽略警告信息
warnings.filterwarnings("ignore")

# 配置类，存储各种参数
class Config:
    sample_rate = 22050
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    # 设备选择，如果有可用的GPU则使用CUDA，否则使用CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vggish_sample_rate = 16000
    f1_threshold = 0.01
    frechet_eps = 1e-6
    kl_eps = 1e-10
    vggish_batch_size = 256
    max_files = 1000

# 创建配置对象
config = Config()

# 加载音频文件的函数
def load_audio(file_path, target_sr=None):
    try:
        # 读取音频文件，always_2d=True确保音频数据为二维数组
        audio, sr = sf.read(file_path, always_2d=True)
        # 如果音频是多声道，取均值转换为单声道
        if audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio.flatten()

        # 音频归一化，避免数值过大
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # 如果需要，将音频重采样到目标采样率
        if target_sr and sr != target_sr:
            audio = resampy.resample(audio, sr, target_sr)
        elif not target_sr and sr != config.sample_rate:
            audio = resampy.resample(audio, sr, config.sample_rate)

        return audio.astype(np.float32)
    except Exception as e:
        print(f"加载音频错误: {file_path} - {str(e)}")
        return np.zeros(0, dtype=np.float32)

# 计算音频的Mel频谱图
def compute_mel(audio, sr=config.sample_rate):
    # 如果音频长度过短，返回全零的Mel频谱图
    if len(audio) < config.hop_length:
        return np.zeros((128, 10))

    try:
        # 使用librosa计算Mel频谱图
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=128,
            power=1.0
        )
        # 将功率谱转换为dB尺度
        return librosa.power_to_db(mel, ref=np.max)
    except Exception as e:
        print(f"Mel计算错误: {str(e)}")
        return np.zeros((128, 10))

# 计算音频的短时傅里叶变换（STFT）的幅度谱
def compute_stft(audio):
    # 如果音频长度过短，返回全零的幅度谱
    if len(audio) < config.hop_length:
        return np.zeros((config.n_fft // 2 + 1, 10))

    # 使用librosa计算STFT
    stft = librosa.stft(
        audio,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length
    )
    # 取STFT的幅度
    magnitude = np.abs(stft)
    return magnitude

# 计算平均绝对误差（MAE）
def calculate_mae(audio_real, audio_gen):
    # 取两个音频的最小长度
    min_len = min(len(audio_real), len(audio_gen))
    if min_len == 0:
        return 0.0

    # 截取到最小长度
    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]
    # 使用sklearn计算MAE
    return mean_absolute_error(audio_real, audio_gen)

# 计算F1分数
def calculate_f1_score(audio_real, audio_gen, threshold=config.f1_threshold):
    # 取两个音频的最小长度
    min_len = min(len(audio_real), len(audio_gen))
    if min_len == 0:
        return 0.0

    # 截取到最小长度
    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]

    # 将音频信号二值化（大于阈值为1，小于等于阈值为0）
    bin_real = (np.abs(audio_real) > threshold).astype(int)
    bin_gen = (np.abs(audio_gen) > threshold).astype(int)

    # 使用sklearn计算F1分数
    return f1_score(bin_real, bin_gen)

# 计算对数谱距离（LSD）
def calculate_lsd(audio_real, audio_gen):
    # 取两个音频的最小长度
    min_len = min(len(audio_real), len(audio_gen))
    if min_len == 0:
        return 0.0

    # 截取到最小长度
    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]

    # 计算两个音频的STFT幅度谱
    stft_real = compute_stft(audio_real)
    stft_gen = compute_stft(audio_gen)

    # 计算对数幅度谱
    log_real = np.log10(np.maximum(stft_real ** 2, 1e-10))
    log_gen = np.log10(np.maximum(stft_gen ** 2, 1e-10))

    # 计算每帧的LSD
    lsd_per_frame = np.sqrt(np.mean((log_real - log_gen) ** 2, axis=0))
    # 返回平均LSD
    return np.mean(lsd_per_frame)

# 计算峰值信噪比（PSNR）
def calculate_psnr(audio_real, audio_gen):
    # 取两个音频的最小长度
    min_len = min(len(audio_real), len(audio_gen))
    if min_len == 0:
        return 0.0

    # 截取到最小长度
    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]

    # 取真实音频的最大绝对值
    max_val = np.max(np.abs(audio_real))
    # 计算均方误差（MSE）
    mse = np.mean((audio_real - audio_gen) ** 2)
    if mse == 0:
        return float('inf')
    # 计算PSNR
    return 20 * np.log10(max_val) - 10 * np.log10(mse)

# 计算结构相似性指数（SSIM）
def calculate_ssim(audio_real, audio_gen):
    # 取两个音频的最小长度
    min_len = min(len(audio_real), len(audio_gen))
    if min_len == 0:
        return 0.0

    # 截取到最小长度
    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]

    return ssim(audio_real, audio_gen)

# 计算Frechet距离
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=config.frechet_eps):
    """改进版Frechet距离计算，增加数值稳定性"""
    # 将均值转换为一维数组
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    # 将协方差矩阵转换为二维数组
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    # 确保协方差矩阵是对称的
    sigma1 = (sigma1 + sigma1.T) / 2
    sigma2 = (sigma2 + sigma2.T) / 2

    # 添加正则项确保协方差矩阵正定
    offset = np.eye(sigma1.shape[0]) * eps
    sigma1 += offset
    sigma2 += offset

    # 计算均值差
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

    # 计算协方差矩阵平方根的迹
    tr_covmean = np.trace(covmean)

    # 计算距离并确保非负
    distance = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return max(distance, 0)  # 确保非负

# 计算核 inception 距离（KID）
def calculate_kid(real_activations, gen_activations, max_size=100):
    # 取最小样本数
    n_samples = min(len(real_activations), len(gen_activations), max_size)
    if n_samples < 2:
        return 0.0

    # 随机采样
    real_sub = real_activations[np.random.choice(len(real_activations), n_samples, replace=False)]
    gen_sub = gen_activations[np.random.choice(len(gen_activations), n_samples, replace=False)]

    # 添加特征归一化
    real_sub = (real_sub - real_sub.mean(0)) / (real_sub.std(0) + 1e-10)
    gen_sub = (gen_sub - gen_sub.mean(0)) / (gen_sub.std(0) + 1e-10)

    # 计算核矩阵
    kernel_real = np.dot(real_sub, real_sub.T)
    kernel_gen = np.dot(gen_sub, gen_sub.T)
    kernel_cross = np.dot(real_sub, gen_sub.T)

    # 计算KID
    kid = np.mean(kernel_real) + np.mean(kernel_gen) - 2 * np.mean(kernel_cross)
    return max(kid, 0)  # 确保非负

# 计算KL散度
def calculate_kl_divergence(P, Q):
    """改进的KL散度计算，避免数值不稳定"""
    eps = config.kl_eps

    # 确保概率分布有效
    P = np.clip(P, eps, 1 - eps)
    Q = np.clip(Q, eps, 1 - eps)

    return tf.keras.losses.KLDivergence()(P, Q).numpy()

# 计算Inception分数（ISC）
def calculate_isc(features):
    if features.size == 0:
        return 0.0

    # 将特征转换为PyTorch张量
    features_tensor = torch.tensor(features).float()
    # 增加数值稳定性处理
    features_tensor = features_tensor - features_tensor.max(dim=1, keepdim=True)[0]
    # 计算平均概率分布
    py = F.softmax(features_tensor, dim=1).mean(dim=0)

    scores = []
    for i in range(features_tensor.shape[0]):
        # 计算每个样本的概率分布
        pyx = F.softmax(features_tensor[i], dim=0)
        # 添加epsilon避免log(0)
        kl = F.kl_div((py + 1e-10).log(), pyx + 1e-10, reduction='sum').item()
        scores.append(kl)

    # 处理全零特征的情况
    mean_score = np.mean(scores) if scores else 0.0
    return np.exp(mean_score)

# VGGish模型包装类
class VGGishWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载VGGish模型
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

# 提取VGGish特征
def extract_vggish_features(audio, sr=config.sample_rate):
    if len(audio) == 0:
        return np.zeros((0, 128))

    try:
        # 如果采样率不匹配，进行重采样
        if sr != config.vggish_sample_rate:
            audio = resampy.resample(audio, sr, config.vggish_sample_rate)

        # 音频归一化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # 计算VGGish所需的对数Mel频谱
        log_mel = vggish_input.waveform_to_examples(audio, config.vggish_sample_rate)

        if len(log_mel) == 0:
            return np.zeros((0, 128))

        # 获取模型所在设备
        device = next(vggish_model.parameters()).device
        # 将对数Mel频谱转换为张量并移动到设备上
        log_mel_tensor = torch.tensor(log_mel).float().to(device)

        if log_mel_tensor.dim() == 3:
            log_mel_tensor = log_mel_tensor.unsqueeze(1)

        features_list = []
        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(log_mel_tensor), batch_size):
                # 分批处理
                batch = log_mel_tensor[i:i + batch_size].to(device)
                features = vggish_model(batch).cpu()

                # 自适应维度归一化
                if features.dim() == 1:
                    features = F.normalize(features, p=2, dim=0)
                elif features.dim() >= 2:
                    features = F.normalize(features, p=2, dim=1)

                if features.dim() == 1:
                    features = features.unsqueeze(0)
                features_list.append(features)

        if features_list:
            return torch.cat(features_list, dim=0).numpy()
    except Exception as e:
        print(f"VGGish特征提取错误: {str(e)}")

    return np.zeros((0, 128))

# 评估音频生成质量的主函数
def evaluate_metrics(real_audio_dir, gen_audio_dir):
    # 初始化结果字典
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

    # 获取真实音频文件列表
    real_files = sorted([os.path.join(real_audio_dir, f) for f in os.listdir(real_audio_dir) if f.endswith('.wav')])
    # 获取生成音频文件列表
    gen_files = sorted([os.path.join(gen_audio_dir, f) for f in os.listdir(gen_audio_dir) if f.endswith('.wav')])

    # 取最小文件数
    min_files = min(len(real_files), len(gen_files), config.max_files)
    if min_files == 0:
        print("错误：没有找到音频文件！")
        return results

    # 截取文件列表到最小文件数
    real_files = real_files[:min_files]
    gen_files = gen_files[:min_files]

    print(f"处理 {min_files} 个音频文件对")

    # 音频级指标计算
    print("\n计算音频级指标...")
    lsd_scores, psnr_scores, ssim_scores, mae_scores, f1_scores = [], [], [], [], []

    for i in tqdm(range(min_files), desc="音频指标"):
        # 加载真实音频
        real_audio = load_audio(real_files[i])
        # 加载生成音频
        gen_audio = load_audio(gen_files[i])

        # 取最小长度
        min_len = min(len(real_audio), len(gen_audio))
        if min_len == 0:
            continue

        # 截取到最小长度
        real_audio = real_audio[:min_len]
        gen_audio = gen_audio[:min_len]

        # 计算LSD
        lsd_scores.append(calculate_lsd(real_audio, gen_audio))
        # 计算PSNR
        psnr_scores.append(calculate_psnr(real_audio, gen_audio))
        # 计算SSIM
        ssim_scores.append(calculate_ssim(real_audio, gen_audio))
        # 计算MAE
        mae_scores.append(calculate_mae(real_audio, gen_audio))
        # 计算F1分数
        f1_scores.append(calculate_f1_score(real_audio, gen_audio))

    # 特征提取
    print("\n提取VGGish特征 (L-AUDIO)...")
    real_features_audio, gen_features_audio = [], []

    for i in tqdm(range(min_files), desc="VGGish特征"):
        # 加载真实音频
        real_audio = load_audio(real_files[i])
        # 加载生成音频
        gen_audio = load_audio(gen_files[i])

        # 取最小长度
        min_len = min(len(real_audio), len(gen_audio))
        if min_len == 0:
            continue

        # 截取到最小长度
        real_audio = real_audio[:min_len]
        gen_audio = gen_audio[:min_len]

        # 提取真实音频的VGGish特征
        real_feat = extract_vggish_features(real_audio)
        # 提取生成音频的VGGish特征
        gen_feat = extract_vggish_features(gen_audio)

        if real_feat.size > 0 and gen_feat.size > 0:
            real_features_audio.append(real_feat)
            gen_features_audio.append(gen_feat)

    # 计算L-AUDIO指标
    if real_features_audio and gen_features_audio:
        try:
            # 拼接特征
            real_audio_feats = np.concatenate(real_features_audio)
            gen_audio_feats = np.concatenate(gen_features_audio)

            # 特征维度处理
            if real_audio_feats.ndim == 1:
                real_audio_feats = real_audio_feats.reshape(-1, 1)
            if gen_audio_feats.ndim == 1:
                gen_audio_feats = gen_audio_feats.reshape(-1, 1)

            # 计算真实音频特征的均值
            mu_real_audio = np.mean(real_audio_feats, axis=0)
            # 计算真实音频特征的协方差矩阵
            sigma_real_audio = np.cov(real_audio_feats, rowvar=False)
            # 计算生成音频特征的均值
            mu_gen_audio = np.mean(gen_audio_feats, axis=0)
            # 计算生成音频特征的协方差矩阵
            sigma_gen_audio = np.cov(gen_audio_feats, rowvar=False)

            # 确保协方差矩阵对称
            sigma_real_audio = (sigma_real_audio + sigma_real_audio.T) / 2
            sigma_gen_audio = (sigma_gen_audio + sigma_gen_audio.T) / 2

            # 计算FAD
            results['FAD']['L-AUDIO'] = calculate_frechet_distance(
                mu_real_audio, sigma_real_audio,
                mu_gen_audio, sigma_gen_audio
            )

            # 计算ISC
            results['ISC']['L-AUDIO'] = calculate_isc(gen_audio_feats)
            # 计算KID
            results['KID']['L-AUDIO'] = calculate_kid(real_audio_feats, gen_audio_feats)

            # 计算KL散度（sigmoid）
            real_sigmoid = torch.sigmoid(torch.tensor(real_audio_feats)).numpy()
            gen_sigmoid = torch.sigmoid(torch.tensor(gen_audio_feats)).numpy()
            results['KL']['L-AUDIO']['KL(sigmoid)'] = calculate_kl_divergence(real_sigmoid, gen_sigmoid)

            # 计算KL散度（softmax）
            real_softmax = F.softmax(torch.tensor(real_audio_feats), dim=1).numpy()
            gen_softmax = F.softmax(torch.tensor(gen_audio_feats), dim=1).numpy()
            results['KL']['L-AUDIO']['KL(softmax)'] = calculate_kl_divergence(real_softmax, gen_softmax)
        except Exception as e:
            print(f"L-AUDIO指标计算错误: {str(e)}")

    # 提取Mel特征 (L-MUSIC)
    print("\n提取Mel特征 (L-MUSIC)...")
    real_mels, gen_mels = [], []

    for i in tqdm(range(min_files), desc="Mel特征"):
        # 加载真实音频
        real_audio = load_audio(real_files[i])
        # 加载生成音频
        gen_audio = load_audio(gen_files[i])

        # 取最小长度
        min_len = min(len(real_audio), len(gen_audio))
        if min_len == 0:
            continue

        # 截取到最小长度
        real_audio = real_audio[:min_len]
        gen_audio = gen_audio[:min_len]

        # 计算真实音频的Mel频谱
        real_mel = compute_mel(real_audio)
        # 计算生成音频的Mel频谱
        gen_mel = compute_mel(gen_audio)

        # 取最小帧数
        min_frames = min(real_mel.shape[1], gen_mel.shape[1])
        if min_frames == 0:
            continue

        # 截取到最小帧数
        real_mel = real_mel[:, :min_frames]
        gen_mel = gen_mel[:, :min_frames]

        real_mels.append(real_mel)
        gen_mels.append(gen_mel)

    # 计算L-MUSIC指标
    if real_mels and gen_mels:
        try:
            # 初始化Inception模型
            inception_model = InceptionV3(include_top=False, pooling='avg')

            # 提取特征和概率
            real_features_music, gen_features_music = [], []

            for i in tqdm(range(len(real_mels)), desc="Inception特征"):
                real_mel = real_mels[i]
                gen_mel = gen_mels[i]

                # 创建伪彩色图像
                real_img = np.stack([real_mel] * 3, axis=0)
                gen_img = np.stack([gen_mel] * 3, axis=0)

                # 转换为张量并插值
                real_img = np.transpose(real_img, (1, 2, 0))
                gen_img = np.transpose(gen_img, (1, 2, 0))

                real_img = tf.image.resize(real_img, (299, 299))
                gen_img = tf.image.resize(gen_img, (299, 299))

                real_img = preprocess_input(real_img)
                gen_img = preprocess_input(gen_img)

                real_img = np.expand_dims(real_img, axis=0)
                gen_img = np.expand_dims(gen_img, axis=0)

                # 提取特征
                real_feat = inception_model.predict(real_img)
                gen_feat = inception_model.predict(gen_img)
                real_features_music.append(real_feat)
                gen_features_music.append(gen_feat)

            # 计算ISC
            gen_all_feats = np.concatenate(gen_features_music)
            results['ISC']['L-MUSIC'] = calculate_isc(gen_all_feats)

            # 计算FID和KID
            real_music_feats = np.concatenate(real_features_music)
            gen_music_feats = np.concatenate(gen_features_music)

            mu_real_music = np.mean(real_music_feats, axis=0)
            sigma_real_music = np.cov(real_music_feats, rowvar=False)
            mu_gen_music = np.mean(gen_music_feats, axis=0)
            sigma_gen_music = np.cov(gen_music_feats, rowvar=False)

            # 确保协方差矩阵对称
            sigma_real_music = (sigma_real_music + sigma_real_music.T) / 2
            sigma_gen_music = (sigma_gen_music + sigma_gen_music.T) / 2

            results['FID']['L-MUSIC'] = calculate_frechet_distance(
                mu_real_music, sigma_real_music,
                mu_gen_music, sigma_gen_music
            )
            results['KID']['L-MUSIC'] = calculate_kid(real_music_feats, gen_music_feats)

            # 计算KL散度
            real_sigmoid = torch.sigmoid(torch.tensor(real_music_feats)).numpy()
            gen_sigmoid = torch.sigmoid(torch.tensor(gen_music_feats)).numpy()
            results['KL']['L-MUSIC']['KL(sigmoid)'] = calculate_kl_divergence(real_sigmoid, gen_sigmoid)

            real_softmax = F.softmax(torch.tensor(real_music_feats), dim=1).numpy()
            gen_softmax = F.softmax(torch.tensor(gen_music_feats), dim=1).numpy()
            results['KL']['L-MUSIC']['KL(softmax)'] = calculate_kl_divergence(real_softmax, gen_softmax)
        except Exception as e:
            print(f"L-MUSIC指标计算错误: {str(e)}")

    # 计算平均音频指标
    results['LSD'] = np.mean(lsd_scores) if lsd_scores else 0.0
    results['PSNR'] = np.mean(psnr_scores) if psnr_scores else 0.0
    results['SSIM'] = np.mean(ssim_scores) if ssim_scores else 0.0
    results['MAE'] = np.mean(mae_scores) if mae_scores else 0.0
    results['F1'] = np.mean(f1_scores) if f1_scores else 0.0

    return results

# 主函数
if __name__ == "__main__":
    global vggish_model
    # 加载VGGish模型
    vggish_model = VGGishWrapper().to(config.device)
    vggish_model.eval()
    print("VGGish模型加载成功")

    # parser = argparse.ArgumentParser(description='音频生成质量评估')
    # parser.add_argument('--real_dir', type=str, required=True, help='真实音频目录')
    # parser.add_argument('--gen_dir', type=str, required=True, help='生成音频目录')
    # parser.add_argument('--output', type=str, default='evaluation_results.txt', help='输出结果文件')
    # parser.add_argument('--f1_threshold', type=float, default=0.01, help='F1-score计算使用的阈值')
    # parser.add_argument('--max_files', type=int, default=1000, help='最大处理文件数')
    #
    # # 解析命令行参数
    # args = parser.parse_args()

    # 更新配置
    config.f1_threshold = 0.01
    config.max_files = 1000

    real_dir = r"/root/autodl-tmp/ViolinDiff/true_wav_dir"
    gen_dir = r"/root/autodl-tmp/ViolinDiff/test_wav_dir"
    output = r"evaluation_results.txt"

    print(f"评估设置:")
    print(f"  真实音频目录: {real_dir}")
    print(f"  生成音频目录: {gen_dir}")
    print(f"  输出文件: {output}")

    # 评估指标
    metrics = evaluate_metrics(real_dir, gen_dir)

    # 输出结果（保持原格式）
    print("\n评估结果:")
    print("FAD:", metrics['FAD'])
    print("ISC:", metrics['ISC'])
    print("KID:", metrics['KID'])
    print("KL:", metrics['KL'])
    print(f"LSD: {metrics['LSD']:.4f}")
    print(f"PSNR: {metrics['PSNR']:.4f}")
    print(f"SSIM: {metrics['SSIM']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"F1-score: {metrics['F1']:.4f}")

    # 将结果保存到文件
    with open(output, 'w') as f:
        f.write("Experimental Results\n\n")
        f.write("| Key | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| FAD | {metrics['FAD']} |\n")
        f.write(f"| ISC | {metrics['ISC']} |\n")
        f.write(f"| KID | {metrics['KID']} |\n")
        f.write(f"| KL | {metrics['KL']} |\n")
        f.write(f"| LSD | {metrics['LSD']:.4f} |\n")
        f.write(f"| PSNR | {metrics['PSNR']:.4f} |\n")
        f.write(f"| SSIM | {metrics['SSIM']:.4f} |\n")
        f.write(f"| MAE | {metrics['MAE']:.4f} |\n")
        f.write(f"| F1-score | {metrics['F1']:.4f} |\n")

    print(f"\n结果已保存到 {output}")