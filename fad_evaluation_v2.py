import os
import numpy as np
import torch
import torchaudio
from scipy import linalg
from harritaylor_torchvggish_master.torchvggish import vggish_input
from harritaylor_torchvggish_master.torchvggish.vggish import VGGish


# 配置参数
class Config:
    real_audio_dir = "/root/autodl-tmp/ViolinDiff/true_wav_dir"  # 真实音频目录
    gen_audio_dir = "/root/autodl-tmp/ViolinDiff/test_wav_dir"  # 生成音频目录
    checkpoint_dir = "./checkpoints"  # 模型权重目录
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32  # 处理音频的批大小
    sample_rate = 16000  # VGGish要求的采样率


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


def load_musicnn_model():
    """加载Musicnn模型用于L-MUSIC"""
    # 这里使用预训练的Musicnn模型
    # 实际应用中应使用专门的音乐特征提取模型
    model = torch.hub.load('seungheondoh/conv-tasnet', 'musicnn', sample_rate=Config.sample_rate)
    model.to(Config.device)
    model.eval()
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


def extract_musicnn_embeddings(audio_dir, model):
    """使用Musicnn提取L-MUSIC特征"""
    all_embeddings = []
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    for file in audio_files:
        audio_path = os.path.join(audio_dir, file)

        # 加载并预处理音频
        waveform, sr = torchaudio.load(audio_path)
        if sr != Config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, Config.sample_rate)
            waveform = resampler(waveform)

        # 确保音频长度至少为1秒
        min_length = Config.sample_rate  # 1秒
        if waveform.shape[1] < min_length:
            padding = min_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding), "constant", 0)

        # 分批处理
        for i in range(0, waveform.shape[1], Config.sample_rate * 10):  # 每10秒一批
            end_idx = min(i + Config.sample_rate * 10, waveform.shape[1])
            batch = waveform[:, i:end_idx].to(Config.device)

            # 提取特征
            with torch.no_grad():
                embeddings = model(batch)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算两个多元高斯分布之间的Fréchet距离"""
    diff = mu1 - mu2

    # 计算协方差矩阵的几何平均
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # 检查复数部分
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算距离公式
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


def calculate_fad(real_emb, gen_emb):
    """计算FAD分数"""
    mu_real = np.mean(real_emb, axis=0)
    sigma_real = np.cov(real_emb, rowvar=False)

    mu_gen = np.mean(gen_emb, axis=0)
    sigma_gen = np.cov(gen_emb, rowvar=False)

    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def main():
    # 加载模型
    print("Loading VGGish model for L-AUDIO...")
    vggish_model = load_vggish_model()

    print("Loading Musicnn model for L-MUSIC...")
    musicnn_model = load_musicnn_model()

    # 提取L-AUDIO特征
    print("\nExtracting L-AUDIO features for real audio...")
    real_audio_emb = extract_vggish_embeddings(Config.real_audio_dir, vggish_model)
    print(f"Real L-AUDIO embeddings shape: {real_audio_emb.shape}")

    print("Extracting L-AUDIO features for generated audio...")
    gen_audio_emb = extract_vggish_embeddings(Config.gen_audio_dir, vggish_model)
    print(f"Generated L-AUDIO embeddings shape: {gen_audio_emb.shape}")

    # 提取L-MUSIC特征
    print("\nExtracting L-MUSIC features for real audio...")
    real_music_emb = extract_musicnn_embeddings(Config.real_audio_dir, musicnn_model)
    print(f"Real L-MUSIC embeddings shape: {real_music_emb.shape}")

    print("Extracting L-MUSIC features for generated audio...")
    gen_music_emb = extract_musicnn_embeddings(Config.gen_audio_dir, musicnn_model)
    print(f"Generated L-MUSIC embeddings shape: {gen_music_emb.shape}")

    # 计算FAD分数
    print("\nCalculating FAD scores...")

    if real_audio_emb.size > 0 and gen_audio_emb.size > 0:
        fad_audio = calculate_fad(real_audio_emb, gen_audio_emb)
        print(f"L-AUDIO FAD Score: {fad_audio:.4f}")
    else:
        print("Error: No L-AUDIO embeddings found.")
        fad_audio = None

    if real_music_emb.size > 0 and gen_music_emb.size > 0:
        fad_music = calculate_fad(real_music_emb, gen_music_emb)
        print(f"L-MUSIC FAD Score: {fad_music:.4f}")
    else:
        print("Error: No L-MUSIC embeddings found.")
        fad_music = None

    # 输出最终结果
    print("\n" + "=" * 50)
    if fad_audio is not None:
        print(f"L-AUDIO FAD Score: {fad_audio:.4f}")
    if fad_music is not None:
        print(f"L-MUSIC FAD Score: {fad_music:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()