import os
import torch
from ultralytics import YOLO


print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

model = YOLO("yolov12n.yaml")

results = model.train(
  # 训练数据集的配置文件路径，从环境变量YOLO_DATA中获取
  data=os.getenv("YOLO_DATA"),
  # 训练轮数，默认600轮，从环境变量YOLO_EPOCHS中获取
  epochs=int(os.getenv("YOLO_EPOCHS", 600)),
  # 批次大小，默认2，从环境变量YOLO_BATCH中获取
  batch=int(os.getenv("YOLO_BATCH", 2)), 
  # 输入图像尺寸，默认640x640，从环境变量YOLO_IMGSZ中获取
  imgsz=int(os.getenv("YOLO_IMGSZ", 640)),
  # 图像缩放比例，默认0.5，不同模型推荐值：S/M/L/X均为0.9
  scale=float(os.getenv("YOLO_SCALE", 0.5)),  # S:0.9; M:0.9; L:0.9; X:0.9
  # Mosaic数据增强的概率，默认1.0，从环境变量YOLO_MOSAIC中获取
  mosaic=float(os.getenv("YOLO_MOSAIC", 1.0)),
  # Mixup数据增强的概率，默认0.0，不同模型推荐值：S:0.05; M/L:0.15; X:0.2
  mixup=float(os.getenv("YOLO_MIXUP", 0.0)),  # S:0.05; M:0.15; L:0.15; X:0.2
  # Copy-Paste数据增强的概率，默认0.1，不同模型推荐值：S:0.15; M:0.4; L:0.5; X:0.6
  copy_paste=float(os.getenv("YOLO_COPY_PASTE", 0.1)),  # S:0.15; M:0.4; L:0.5; X:0.6
  # 训练设备，默认使用第一块GPU(0)，从环境变量YOLO_DEVICE中获取
  device=os.getenv("YOLO_DEVICE", "0"),
)


# Evaluate model performance on the validation set
metrics = model.val()


# Save the model
model.save("yolov12n.pt")

