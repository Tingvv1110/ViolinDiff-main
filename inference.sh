#!/bin/bash

# 要处理的MIDI文件所在目录
MIDI_DIR="./test_mid_dir"

# 基础输出目录（用于动态生成子目录）
BASE_OUTPUT_DIR="./inference_wav"

# 模型文件基础路径
BASE_SYNTH_DIR="./weights/synth"
BASE_BEND_DIR="./weights/bend"

# 外层循环：从1000递增到40000，步长1000
for n in {1000..10000..1000}; do
    # 计算第二个数值（n的3倍，对应xxx_3xxx.pt的格式）
    m=$((n * 3))

    # 动态生成当前轮次的输出目录
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${n}_epoch"

    # 动态生成模型文件路径
    synth_pth="${BASE_SYNTH_DIR}/${n}_${m}.pt"
    bend_pth="${BASE_BEND_DIR}/${n}_${m}.pt"

    # 确保当前输出目录存在
    mkdir -p "$OUTPUT_DIR"

    # 遍历MIDI目录下的所有mid文件
    for midi_file in "$MIDI_DIR"/*.mid; do
        # 提取文件名（不含扩展名）
        filename=$(basename -- "$midi_file")
        filename_no_ext="${filename%.*}"

        # 目标wav文件路径
        target_wav="$OUTPUT_DIR/${filename_no_ext}.wav"

        # 检查wav文件是否已存在，不存在才执行推理
        if [ ! -f "$target_wav" ]; then
            python3 inference.py \
            --synth_pth "$synth_pth" \
            --bend_pth "$bend_pth" \
            --midi_pth "$midi_file" \
            --save_pth "$target_wav" \
            --performer 13 \
            --device cuda:0
        else
            echo "文件 $target_wav 已存在，跳过生成"
        fi
    done

    echo "完成 $n 轮次处理"
done

echo "所有轮次处理完毕"