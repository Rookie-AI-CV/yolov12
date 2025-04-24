import json
import os
import logging
from tqdm import tqdm
import cv2
import numpy as np
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_dataset_paths(root_dir):
    """
    获取数据集的训练和测试数据路径
    
    Args:
        root_dir: 数据集根目录路径
        
    Returns:
        train_dir: 训练数据目录路径
        train_anno: 训练数据标注文件路径 
        valid_dir: 验证数据目录路径
        valid_anno: 验证数据标注文件路径
    """
    # 训练数据路径
    train_dir = os.path.join(root_dir, 'train')
    train_anno = os.path.join(train_dir, '_annotations.coco.json')
    
    # 验证数据路径
    valid_dir = os.path.join(root_dir, 'valid') 
    valid_anno = os.path.join(valid_dir, '_annotations.coco.json')
    
    # 验证路径是否存在
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(train_anno):
        raise FileNotFoundError(f"Training annotation file not found: {train_anno}")
    if not os.path.exists(valid_dir):
        raise FileNotFoundError(f"Validation directory not found: {valid_dir}")
    if not os.path.exists(valid_anno):
        raise FileNotFoundError(f"Validation annotation file not found: {valid_anno}")
        
    return train_dir, train_anno, valid_dir, valid_anno


def process_annotations(annotation_file):
    """
    处理COCO格式标注文件,获取归一化的边界框和类别信息
    
    Args:
        annotation_file: COCO格式标注JSON文件路径
    
    Returns:
        images_info: 包含处理后图像信息的字典,带有归一化边界框
    """
    # 加载标注文件
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    logger.info("Processing annotation file...")
    
    # 创建图像信息字典
    images_info = {v['id']: v for v in annotations['images']}
    
    # 处理标注信息
    for annotation in annotations['annotations']:
        width = images_info[annotation['image_id']]['width']
        height = images_info[annotation['image_id']]['height']
        bbox = annotation['bbox']
        # 转换为YOLO格式: 中心点x,中心点y,宽度,高度
        x, y, w, h = bbox
        bbox[0] = (x + w/2) / width  # 中心点x
        bbox[1] = (y + h/2) / height # 中心点y
        bbox[2] = w / width          # 宽度
        bbox[3] = h / height         # 高度
        if 'bbox' not in images_info[annotation['image_id']]:
            images_info[annotation['image_id']]['bbox'] = [bbox]
        else:
            images_info[annotation['image_id']]['bbox'].append(bbox)
            
        category_id = annotation['category_id']
        if 'category_id' not in images_info[annotation['image_id']]:
            images_info[annotation['image_id']]['category_id'] = [category_id]
        else:
            images_info[annotation['image_id']]['category_id'].append(category_id)
    
    logger.info("Annotation processing completed")
    return images_info


def generate_yaml_config(json_file, dataset_root, save_path):
    """
    生成YOLO数据集的YAML配置文件
    
    Args:
        json_file: COCO格式标注JSON文件路径
        dataset_root: 相对于工作目录的数据集根目录
        save_path: 生成的YAML文件保存路径
    """
    # 从JSON文件读取类别信息
    with open(json_file, 'r') as f:
        annotation = json.load(f)
    logger.info("Generating YAML configuration file...")
    dataset_root = os.path.abspath(dataset_root)
    # 创建YAML内容
    yaml_content = f"""
path: {dataset_root}  # dataset root dir
train:
  - {os.path.join(dataset_root, 'images', 'train')}   
val:
  - {os.path.join(dataset_root, 'images', 'valid')}  

# Classes
names:
"""
    for anno in annotation['categories']:
        yaml_content += f"  {anno['id']}: {anno['name']}\n"

    # 写入YAML文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    logger.info(f"YAML configuration saved to: {save_path}")


def cocodateset_to_yolostddataset(coco_images_dir, annotation_file, save_dir, dataset_type='train', convert_gray=False):
    """
    将COCO格式数据集转换为YOLO标准数据集格式
    
    Args:
        coco_images_dir: COCO数据集图片目录
        annotation_file: COCO格式标注文件路径 
        save_dir: 保存YOLO格式数据集的根目录
        dataset_type: 数据集类型，默认为'train', 可选值为'train'、'valid'、'test'
        convert_gray: 是否转换为灰度图，默认为False
        
    数据集转换过程:
    1. 创建YOLO格式目录结构: 
       save_dir/
         ├── images/
         │   └── train/
         └── labels/
             └── train/
    2. 复制图片并重命名为6位数字序号
    3. 生成对应的标签文件,每行格式为: <category_id> <x_center> <y_center> <width> <height>
    """
    import json
    import shutil

    # 创建保存目录
    save_images_dir = os.path.join(save_dir, "images", dataset_type)
    save_labels_dir = os.path.join(save_dir, "labels", dataset_type)
    os.makedirs(save_images_dir, exist_ok=True)
    os.makedirs(save_labels_dir, exist_ok=True)
    
    logger.info(f"Converting {dataset_type} dataset...")
    
    # 处理标注信息
    images_info = process_annotations(annotation_file)
    logger.info(f"Processing {len(images_info)} images...")
    
    for k, image in tqdm(images_info.items(), desc="Converting images", unit="img"):
        logger.debug(f"Processing image {k+1}/{len(images_info)}: {image['file_name']}")
        
        file_name = image['file_name']
        old_path = os.path.join(coco_images_dir, file_name)
        new_name = f"{k:06d}.jpg"
        new_path = os.path.join(save_images_dir, new_name)
        
        if convert_gray:
            # 先复制原图
            shutil.copy(old_path, new_path)
            # 再转换为灰度图
            img = cv2.imread(new_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 将单通道灰度图复制为3通道
            gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(new_path, gray_3channel)
        else:
            # 直接复制原图
            shutil.copy(old_path, new_path)
        
        # 更新文件名
        images_info[k]['file_name'] = new_name
        
        # 生成YOLO格式标签文件
        label_path = os.path.join(save_labels_dir, f"{k:06d}.txt")
        with open(label_path, 'w') as f:
            if 'category_id' in image.keys():
                for category_id, bbox in zip(image['category_id'], image['bbox']):
                    # YOLO格式: <类别id> <中心点x> <中心点y> <宽度> <高度>
                    f.write(f"{category_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
    
    logger.info(f"Dataset conversion completed. Processed {len(images_info)} images")


def main(input_dir, output_dir, yaml_path, convert_gray=False):
    """
    主函数
    
    Args:
        input_dir: 输入数据集目录
        output_dir: 输出数据集目录
        yaml_path: YAML配置文件保存路径
        convert_gray: 是否转换为灰度图
    """
    logger.info("Starting dataset conversion process...")
    train_dir, train_anno, valid_dir, valid_anno = get_dataset_paths(input_dir)
    
    # 训练集转换
    cocodateset_to_yolostddataset(train_dir, train_anno, output_dir, "train", convert_gray)
    # 验证集转换
    cocodateset_to_yolostddataset(valid_dir, valid_anno, output_dir, "valid", convert_gray)
    # 生成yaml配置文件
    generate_yaml_config(train_anno, output_dir, yaml_path)
    logger.info("Dataset conversion completed successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将COCO数据集转换为YOLO格式')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='包含COCO数据集的输入目录')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='YOLO格式数据集的输出目录')
    parser.add_argument('-y', '--yaml_path', type=str, help='YAML配置文件保存路径')
    parser.add_argument('--gray', action='store_true', help='是否转换为灰度图')
    
    args = parser.parse_args()
    
    if args.yaml_path is None:
        args.yaml_path = args.output_dir + "/chengdu.yaml"
        
    main(args.input_dir, args.output_dir, args.yaml_path, args.gray)
