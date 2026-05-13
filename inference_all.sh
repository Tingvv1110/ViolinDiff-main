#!/bin/bash

# 要处理的MIDI文件所在目录 - 请替换为实际目录
MIDI_DIR="/root/autodl-tmp/ViolinDiff/test_mid_dir"

# 基础输出目录
BASE_OUTPUT_DIR="/root/autodl-tmp/ViolinDiff/inference_wav"

# 模型目录
SYNTH_DIR="/root/autodl-tmp/ViolinDiff/weights/synth"
BEND_DIR="/root/autodl-tmp/ViolinDiff/weights/bend"

# 模型文件名列表
MODEL_FILES=(
  "1000_3000.pt" "2000_6000.pt" "3000_9000.pt" "4000_12000.pt" "5000_15000.pt" "6000_18000.pt" "7000_21000.pt" "8000_24000.pt" "9000_27000.pt" "10000_30000.pt"
  "11000_33000.pt" "12000_36000.pt" "13000_39000.pt" "14000_42000.pt" "15000_45000.pt" "16000_48000.pt" "17000_51000.pt" "18000_54000.pt" "19000_57000.pt" "20000_60000.pt"
  "21000_63000.pt" "22000_66000.pt" "23000_69000.pt" "24000_72000.pt" "25000_75000.pt" "26000_78000.pt" "27000_81000.pt" "28000_84000.pt" "29000_87000.pt" "30000_90000.pt"
  "31000_93000.pt" "32000_96000.pt" "33000_99000.pt" "34000_102000.pt" "35000_105000.pt" "36000_108000.pt" "37000_111000.pt" "38000_114000.pt" "39000_117000.pt" "40000_120000.pt"
)

# 遍历所有模型文件
for model_file in "${MODEL_FILES[@]}"; do
    # 提取模型文件名中下划线前的数字作为文件夹名称前缀
    folder_prefix="${model_file%_*}"
    # 构建当前模型对应的输出目录
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${folder_prefix}_epoch"
    # 确保输出目录存在
    mkdir -p "$OUTPUT_DIR"

    # 遍历MIDI目录下的所有mid文件
    for midi_file in "$MIDI_DIR"/*.mid; do
        # 提取MIDI文件名（不含扩展名）
        filename=$(basename -- "$midi_file")
        filename_no_ext="${filename%.*}"

        # 为每个文件使用当前模型执行推理命令
        python3 inference.py \
        --synth_pth "${SYNTH_DIR}/${model_file}" \
        --bend_pth "${BEND_DIR}/${model_file}" \
        --midi_pth "$midi_file" \
        --save_pth "${OUTPUT_DIR}/${filename_no_ext}.wav" \
        --performer 13 \
        --device cuda
    done
done