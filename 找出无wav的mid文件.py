import os

mid_dir = r"/root/autodl-tmp/ViolinDiff/test_mid_dir"
wav_dir = r"/root/autodl-tmp/ViolinDiff/true_wav_dir"

mid_names = os.listdir(mid_dir)
wav_names = os.listdir(wav_dir)

for mid_name in mid_names:
    name = mid_name.split(".mid")[0]
    wav_name = name+".wav"
    if wav_name not in wav_names:
        print(mid_name)