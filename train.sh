#!/bin/bash

# 数据加载相关参数
export YOLO_DATA="data.yaml"                  # * 数据集配置文件路径,包含训练和验证数据、类名和类数
export YOLO_WORKERS=8                         # 数据加载工作线程数,影响数据预处理和输入模型的速度
export YOLO_BATCH=2                           # * 批量大小,可设为整数、自动模式(-1)或指定GPU利用率(0-1)
export YOLO_IMGSZ=640                         # * 训练的目标图像尺寸,影响模型精度和计算复杂度
export YOLO_CACHE="false"                     # 是否缓存数据集图像(RAM/disk/false),可提高训练速度但增加内存使用
export YOLO_RECT="false"                      # 是否使用矩形训练,可优化批次减少填充,提高效率但可能影响精度

# 优化器相关参数
export YOLO_OPTIMIZER="auto"                 # * 优化器类型(SGD/Adam/AdamW等),影响收敛速度和稳定性
export YOLO_WEIGHT_DECAY=0.0005              # L2正则化系数,对大权重进行惩罚防止过拟合
export YOLO_MOMENTUM=0.937                   # SGD动量因子或Adam的beta1,用于将历史梯度纳入当前更新
export YOLO_LR0=0.01                         # * 初始学习率,对优化过程至关重要
export YOLO_LRF=0.01                         # 最终学习率占初始学习率的比例,用于学习率调度

# 训练策略参数
export YOLO_EPOCHS=600                       # * 训练总轮数,每轮对整个数据集训练一次
export YOLO_COS_LR="false"                   # 是否使用余弦学习率调度,有助于更好的收敛
export YOLO_WARMUP_EPOCHS=3.0                # 学习率预热轮数,从低值逐渐增加到初始学习率
export YOLO_WARMUP_MOMENTUM=0.8              # 预热阶段的初始动量,逐渐调整到设定动量
export YOLO_WARMUP_BIAS_LR=0.1               # 预热阶段的偏置参数学习率,帮助稳定初始训练
export YOLO_CLOSE_MOSAIC=10                  # 训练结束前禁用马赛克增强的轮数,0表示禁用

# 损失函数权重
export YOLO_BOX=7.5                          # 边界框损失权重,影响边框坐标预测的重视程度
export YOLO_CLS=0.5                          # 分类损失权重,影响类别预测的重要性
export YOLO_DFL=1.5                          # 分布焦点损失权重,用于精细分类
export YOLO_KOBJ=2.0                         # 关键点目标性损失权重,平衡检测可信度与姿态精度
export YOLO_POSE=12.0                        # 姿态损失权重,影响姿态关键点预测的重视程度

# 数据集相关参数
export YOLO_SINGLE_CLS="false"               # 是否将所有类别视为单一类别,适用于二元分类
export YOLO_FRACTION=1.0                     # 使用的数据集比例,允许在数据子集上训练

# 模型相关参数
export YOLO_MODEL_YAML="yolov12n.yaml"       # * 模型配置文件,定义模型结构
export YOLO_MODEL=""                         # * 预训练模型路径(.pt文件)
export YOLO_PRETRAINED="true"                # 是否使用预训练权重,可提高训练效率
export YOLO_DROPOUT=0.0                      # 分类任务中的dropout率,通过随机丢弃防止过拟合

# 训练过程参数
export YOLO_DEVICE="0"                       # * 训练设备(GPU编号/cpu/mps)
export YOLO_AMP="true"                       # 是否启用自动混合精度训练,可减少内存使用并加快训练
export YOLO_DETERMINISTIC="true"             # 是否使用确定性算法确保可重复性
export YOLO_SEED=0                           # 随机种子,确保相同配置下结果可重复
export YOLO_VAL="true"                       # 是否在训练过程中进行验证评估

# 保存相关参数
export YOLO_PROJECT=""                       # 项目目录名称,用于组织存储实验结果
export YOLO_NAME=""                          # 训练运行的名称,用于在项目目录下创建子目录
export YOLO_EXIST_OK="false"                 # 是否允许覆盖已存在的实验目录
export YOLO_SAVE="true"                      # 是否保存训练检查点和最终模型
export YOLO_SAVE_PERIOD=-1                   # 保存模型检查点的频率(轮数),-1表示禁用
export YOLO_SAVE_PATH="yolov12n.pt"          # * 模型保存路径

# 其他参数
export YOLO_PATIENCE=100                     # 早停耐心值,验证指标未改善时的最大等待轮数
export YOLO_PLOTS="false"                    # 是否生成训练验证指标图和预测示例图
export YOLO_PROFILE="false"                  # 是否对ONNX和TensorRT速度进行分析
export YOLO_RESUME="false"                   # * 是否从上次检查点恢复训练

train_log="train_log.txt"

# 记录脚本开始执行的时间和配置
echo "==================== 训练开始 ====================" >> $train_log
echo "执行时间: $(date)" >> $train_log
echo "" >> $train_log
echo "当前训练配置:" >> $train_log
env | grep "YOLO_" >> $train_log
echo "" >> $train_log
echo "==================== 训练日志 ====================" >> $train_log

# 使用nohup在后台运行训练,并将输出追加到train_log.txt
nohup python train.py >> $train_log 2>&1 &

# 打印进程ID并记录到日志
pid=$!
echo "训练已启动,进程ID: $pid"
echo "日志正在写入$train_log"
echo "进程ID: $pid" >> $train_log
echo "" >> $train_log