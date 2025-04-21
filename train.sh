#!/bin/bash

# 设置训练的环境变量
export YOLO_DATA="data.yaml"  # 数据配置文件路径
export YOLO_EPOCHS=600        # 训练轮数
export YOLO_BATCH=2          # 批次大小
export YOLO_IMGSZ=640        # 输入图像尺寸
export YOLO_SCALE=0.5        # 图像缩放比例
export YOLO_MOSAIC=1.0       # Mosaic数据增强概率
export YOLO_MIXUP=0.0        # Mixup数据增强概率
export YOLO_COPY_PASTE=0.1   # Copy-Paste数据增强概率
export YOLO_DEVICE="0"       # 使用第一块GPU

# 使用nohup在后台运行训练,并将输出追加到train_log.txt
nohup python train.py >> train_log.txt 2>&1 &

# 打印进程ID
echo "训练已启动,进程ID: $!"
echo "日志正在写入train_log.txt"