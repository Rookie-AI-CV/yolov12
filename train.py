# 导入必要的库
import os
import torch
from ultralytics import YOLO

# 打印CUDA设备信息
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")  # 检查是否可用CUDA
print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")  # 获取GPU设备名称

# 初始化YOLO模型
model = YOLO(os.getenv("YOLO_MODEL_YAML", "yolov12n.yaml"))  # 从环境变量获取模型配置文件,默认使用yolov12n.yaml

# 开始训练模型
results = model.train(
    # 数据加载相关参数
    workers=int(os.getenv("YOLO_WORKERS", 8)),  # 数据加载的工作进程数
    batch=int(os.getenv("YOLO_BATCH", 16)),  # 训练的批次大小
    imgsz=int(os.getenv("YOLO_IMGSZ", 640)),  # 输入图像尺寸
    cache=os.getenv("YOLO_CACHE", "False").lower() == "true",  # 是否缓存图像到RAM
    rect=os.getenv("YOLO_RECT", "False").lower() == "true",  # 是否使用矩形训练
    
    # 优化器相关参数
    optimizer=os.getenv("YOLO_OPTIMIZER", "auto"),  # 优化器类型
    weight_decay=float(os.getenv("YOLO_WEIGHT_DECAY", 0.0005)),  # 权重衰减
    momentum=float(os.getenv("YOLO_MOMENTUM", 0.937)),  # 动量
    lr0=float(os.getenv("YOLO_LR0", 0.01)),  # 初始学习率
    lrf=float(os.getenv("YOLO_LRF", 0.01)),  # 最终学习率
    
    # 训练策略参数
    epochs=int(os.getenv("YOLO_EPOCHS", 100)),  # 训练轮数
    cos_lr=os.getenv("YOLO_COS_LR", "False").lower() == "true",  # 是否使用余弦学习率
    warmup_epochs=float(os.getenv("YOLO_WARMUP_EPOCHS", 3.0)),  # 预热轮数
    warmup_momentum=float(os.getenv("YOLO_WARMUP_MOMENTUM", 0.8)),  # 预热动量
    warmup_bias_lr=float(os.getenv("YOLO_WARMUP_BIAS_LR", 0.1)),  # 预热偏置学习率
    close_mosaic=int(os.getenv("YOLO_CLOSE_MOSAIC", 10)),  # 关闭马赛克增强的轮数
    
    # 损失函数权重
    box=float(os.getenv("YOLO_BOX", 7.5)),  # 边界框损失权重
    cls=float(os.getenv("YOLO_CLS", 0.5)),  # 分类损失权重
    dfl=float(os.getenv("YOLO_DFL", 1.5)),  # DFL损失权重
    kobj=float(os.getenv("YOLO_KOBJ", 2.0)),  # 关键点目标损失权重
    pose=float(os.getenv("YOLO_POSE", 12.0)),  # 姿态损失权重
    
    # 数据集相关参数
    data=os.getenv("YOLO_DATA"),  # 数据集配置文件路径
    single_cls=os.getenv("YOLO_SINGLE_CLS", "False").lower() == "true",  # 是否为单类别检测
    classes=[int(x) for x in os.getenv("YOLO_CLASSES").split(",")] if os.getenv("YOLO_CLASSES") else None,  # 指定训练类别
    fraction=float(os.getenv("YOLO_FRACTION", 1.0)),  # 训练数据集比例
    
    # 数据增强参数
    hsv_h=float(os.getenv("YOLO_HSV_H", 0.015)),  # HSV色调增强
    hsv_s=float(os.getenv("YOLO_HSV_S", 0.7)),  # HSV饱和度增强
    hsv_v=float(os.getenv("YOLO_HSV_V", 0.4)),  # HSV亮度增强
    degrees=float(os.getenv("YOLO_DEGREES", 0.0)),  # 旋转角度范围
    translate=float(os.getenv("YOLO_TRANSLATE", 0.1)),  # 平移范围
    scale=float(os.getenv("YOLO_SCALE", 0.5)),  # 缩放范围
    shear=float(os.getenv("YOLO_SHEAR", 0.0)),  # 剪切角度
    perspective=float(os.getenv("YOLO_PERSPECTIVE", 0.0)),  # 透视变换
    flipud=float(os.getenv("YOLO_FLIPUD", 0.0)),  # 上下翻转概率
    fliplr=float(os.getenv("YOLO_FLIPLR", 0.5)),  # 左右翻转概率
    bgr=float(os.getenv("YOLO_BGR", 0.0)),  # BGR通道翻转概率
    mosaic=float(os.getenv("YOLO_MOSAIC", 1.0)),  # 马赛克增强概率
    mixup=float(os.getenv("YOLO_MIXUP", 0.0)),  # mixup增强概率
    copy_paste=float(os.getenv("YOLO_COPY_PASTE", 0.0)),  # 复制粘贴概率
    copy_paste_mode=os.getenv("YOLO_COPY_PASTE_MODE", "flip"),  # 复制粘贴模式
    auto_augment=os.getenv("YOLO_AUTO_AUGMENT", "randaugment"),  # 自动增强策略
    erasing=float(os.getenv("YOLO_ERASING", 0.4)),  # 随机擦除概率
    
    # 模型相关参数
    model=os.getenv("YOLO_MODEL"),  # 预训练模型路径
    pretrained=os.getenv("YOLO_PRETRAINED", "True"),  # 是否使用预训练权重
    freeze=int(os.getenv("YOLO_FREEZE")) if os.getenv("YOLO_FREEZE") else None,  # 冻结层数
    dropout=float(os.getenv("YOLO_DROPOUT", 0.0)),  # dropout比例
    
    # 训练过程参数
    device=os.getenv("YOLO_DEVICE", None),  # 训练设备
    amp=os.getenv("YOLO_AMP", "True").lower() == "true",  # 是否使用混合精度训练
    deterministic=os.getenv("YOLO_DETERMINISTIC", "True").lower() == "true",  # 是否使用确定性训练
    seed=int(os.getenv("YOLO_SEED", 0)),  # 随机种子
    val=os.getenv("YOLO_VAL", "True").lower() == "true",  # 是否进行验证
    
    # 保存相关参数
    project=os.getenv("YOLO_PROJECT"),  # 项目名称
    name=os.getenv("YOLO_NAME"),  # 实验名称
    exist_ok=os.getenv("YOLO_EXIST_OK", "False").lower() == "true",  # 是否允许覆盖已存在的实验目录
    save=os.getenv("YOLO_SAVE", "True").lower() == "true",  # 是否保存模型
    save_period=int(os.getenv("YOLO_SAVE_PERIOD", -1)),  # 保存周期
    
    # 其他参数
    patience=int(os.getenv("YOLO_PATIENCE", 100)),  # 早停耐心值
    plots=os.getenv("YOLO_PLOTS", "False").lower() == "true",  # 是否绘制训练图表
    profile=os.getenv("YOLO_PROFILE", "False").lower() == "true",  # 是否进行性能分析
    resume=os.getenv("YOLO_RESUME", "False").lower() == "true",  # 是否恢复中断的训练
    time=float(os.getenv("YOLO_TIME")) if os.getenv("YOLO_TIME") else None,  # 训练时间限制
)

# 在验证集上评估模型性能
metrics = model.val()

# 保存训练好的模型
model.save(os.getenv("YOLO_SAVE_PATH", "yolov12n.pt"))  # 保存路径默认为yolov12n.pt
