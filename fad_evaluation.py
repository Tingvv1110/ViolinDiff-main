import os
import numpy as np
import torch
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


def load_model():
    """加载VGGish模型"""
    urls = {
        "vggish": f"file://{os.path.abspath(os.path.join(Config.checkpoint_dir, 'vggish-10086976.pth'))}",
        "pca": f"file://{os.path.abspath(os.path.join(Config.checkpoint_dir, 'vggish_pca_params-970ea276.pth'))}"
    }
    model = VGGish(
        urls=urls,
        pretrained=True,
        preprocess=False,  # 关键修复：禁用输入预处理
        postprocess=False,  # FAD使用原始嵌入，不需要后处理
        device=Config.device
    )
    model.eval()
    return model


def extract_embeddings(audio_dir, model):
    """从目录中的所有音频文件提取嵌入特征"""
    all_embeddings = []
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    for file in audio_files:
        # 加载音频并转换为VGGish输入格式
        audio_path = os.path.join(audio_dir, file)
        examples = vggish_input.wavfile_to_examples(audio_path)

        # 分批处理音频片段
        for i in range(0, len(examples), Config.batch_size):
            batch = examples[i:i + Config.batch_size].to(Config.device)
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
    print("Loading VGGish model...")
    model = load_model()

    # 提取特征
    print("Extracting real audio embeddings...")
    real_emb = extract_embeddings(Config.real_audio_dir, model)
    print(f"Real embeddings shape: {real_emb.shape}")

    print("Extracting generated audio embeddings...")
    gen_emb = extract_embeddings(Config.gen_audio_dir, model)
    print(f"Generated embeddings shape: {gen_emb.shape}")

    # 计算FAD
    if real_emb.size == 0 or gen_emb.size == 0:
        print("Error: No embeddings found. Check audio files and directories.")
        return

    fad = calculate_fad(real_emb, gen_emb)
    print("\n" + "=" * 50)
    print(f"FAD Score: {fad:.4f}")
    print(f"L-AUDIO: {fad:.4f}")  # L-AUDIO使用标准FAD
    print(f"L-MUSIC: {fad:.4f}")  # L-MUSIC使用相同FAD
    print("=" * 50)


if __name__ == "__main__":
    main()