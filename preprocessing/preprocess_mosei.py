import os
import numpy as np
from mmsdk import mmdatasdk
from tqdm import tqdm
import platform
import psutil

# 检查是否在 Apple Silicon 上运行
IS_APPLE_SILICON = platform.processor() == 'arm' and platform.system() == 'Darwin'

# 配置 NumPy 以使用多线程
if IS_APPLE_SILICON:
    # 设置线程数为物理核心数
    os.environ['OMP_NUM_THREADS'] = str(psutil.cpu_count(logical=False))
    os.environ['MKL_NUM_THREADS'] = str(psutil.cpu_count(logical=False))
    # 启用 MKL 优化
    np.show_config()

def safe_mean(x):
    """Safely compute the mean of an array, handling various edge cases.
    
    Args:
        x: Input array or sequence
        
    Returns:
        Mean value as float32, or None if mean cannot be computed
    """
    try:
        # Convert input to numpy array if it isn't already
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        # Handle empty arrays
        if x.size == 0:
            return None
            
        # Handle non-numeric data
        if not np.issubdtype(x.dtype, np.number):
            return None
            
        # Handle arrays with NaN/Inf values
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            # Remove NaN and Inf values before computing mean
            x = x[~np.isnan(x) & ~np.isinf(x)]
            if x.size == 0:
                return None
                
        # Compute mean and convert to float32
        mean_val = np.mean(x).astype(np.float32)
        
        # Check if mean is finite
        if not np.isfinite(mean_val):
            return None
            
        return mean_val
        
    except Exception as e:
        print(f"Error in safe_mean: {str(e)}")
        return None

def process_features(dataset, vids, feature_name):
    """Process features for a list of video IDs using vectorized operations.
    
    Args:
        dataset: The MMSDK dataset
        vids: List of video IDs
        feature_name: Name of the feature to process
        
    Returns:
        List of processed features
    """
    features = []
    for vid in tqdm(vids, desc=f"Processing {feature_name}"):
        try:
            data = dataset.computational_sequences[feature_name].data[vid]['features']
            if feature_name == 'COVAREP':
                # For audio features, compute mean
                mean_val = safe_mean(data)
                if mean_val is not None:
                    features.append(mean_val)
            else:
                # For text features, keep original
                if data is not None and data.size > 0:
                    features.append(data)
        except Exception as e:
            print(f"Error processing {feature_name} for video {vid}: {str(e)}")
            continue
    return features

def download_and_align():
    # 设置数据目录
    DATA_DIR = "data/CMU_MOSEI"
    ALIGNED_DIR = os.path.join(DATA_DIR, "aligned")
    os.makedirs(ALIGNED_DIR, exist_ok=True)
    
    # 下载文本、音频特征和标签
    csd_files = {
        'glove_vectors': 'CMU_MOSEI_TimestampedWordVectors.csd',
        'COVAREP': 'CMU_MOSEI_COVAREP.csd',
        'labels': 'CMU_MOSEI_Labels.csd'
    }
    
    # 检查文件是否已存在
    missing_files = []
    for key, filename in csd_files.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            missing_files.append((key, filename))
    
    if missing_files:
        print("开始下载缺失的文件...")
        for key, filename in missing_files:
            print(f"下载 {filename}...")
            url = f"http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/{filename}" if key == 'glove_vectors' else \
                  f"http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/acoustic/{filename}" if key == 'COVAREP' else \
                  f"http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/{filename}"
            os.system(f"wget {url} -P {DATA_DIR}")
    
    # 加载数据集
    print("加载数据集...")
    recipe = {
        'glove_vectors': os.path.join(DATA_DIR, csd_files['glove_vectors']),
        'COVAREP': os.path.join(DATA_DIR, csd_files['COVAREP']),
        'labels': os.path.join(DATA_DIR, csd_files['labels'])
    }
    
    try:
        dataset = mmdatasdk.mmdataset(recipe)
        
        # 打印数据集结构
        print("\n数据集结构:")
        for key in dataset.computational_sequences:
            print(f"\n{key} 序列:")
            data = dataset.computational_sequences[key].data
            print(f"样本数量: {len(data)}")
            if data:
                sample = next(iter(data.values()))
                print(f"特征形状: {sample['features'].shape}")
                print(f"特征类型: {sample['features'].dtype}")
        
        # 对齐特征
        print("\n对齐特征...")
        dataset.align('glove_vectors')
        
        # 获取标准数据集分割
        print("\n获取数据集分割...")
        try:
            splits = mmdatasdk.standard_datasets.CMU_MOSEI.standard_folds
            print("成功加载标准分割")
            for split in splits:
                print(f"{split} 集样本数: {len(splits[split])}")
        except:
            print("无法加载标准分割，使用默认分割...")
            all_vids = list(dataset.computational_sequences['glove_vectors'].data.keys())
            total = len(all_vids)
            splits = {
                'train': all_vids[:int(0.8*total)],
                'valid': all_vids[int(0.8*total):int(0.9*total)],
                'test': all_vids[int(0.9*total):]
            }
        
        # 保存对齐后的特征
        print("\n保存对齐后的特征...")
        for split_name, split_vids in splits.items():
            print(f"\n处理 {split_name} 集...")
            try:
                # 获取该分割中的所有视频ID
                valid_vids = []
                for vid in split_vids:
                    if vid in dataset.computational_sequences['glove_vectors'].data and \
                       vid in dataset.computational_sequences['COVAREP'].data and \
                       vid in dataset.computational_sequences['labels'].data:
                        valid_vids.append(vid)
                
                print(f"有效样本数: {len(valid_vids)}")
                
                # 使用向量化操作处理特征
                text_features = process_features(dataset, valid_vids, 'glove_vectors')
                audio_features = process_features(dataset, valid_vids, 'COVAREP')
                
                # 处理标签
                labels = []
                raw_sentiments = []  # 添加原始情感值列表
                for vid in tqdm(valid_vids, desc="处理标签"):
                    try:
                        # 获取情感标签
                        label = dataset.computational_sequences['labels'].data[vid]['features']
                        mean_sentiment = label.mean()
                        raw_sentiments.append(mean_sentiment)  # 保存原始情感值
                        
                        # 将连续值转换为离散类别
                        # < 0.1 -> 0 (消极)
                        # [0.1, 0.3] -> 1 (中性)
                        # > 0.3 -> 2 (积极)
                        if mean_sentiment < 0.1:
                            labels.append(0)  # 消极
                        elif mean_sentiment > 0.3:
                            labels.append(2)  # 积极
                        else:
                            labels.append(1)  # 中性
                    except Exception as e:
                        print(f"处理样本 {vid} 的标签时出错: {str(e)}")
                        continue
                
                # 打印原始情感值的统计信息
                raw_sentiments = np.array(raw_sentiments)
                print(f"\n原始情感值统计 ({split_name}):")
                print(f"最小值: {raw_sentiments.min():.3f}")
                print(f"最大值: {raw_sentiments.max():.3f}")
                print(f"平均值: {raw_sentiments.mean():.3f}")
                print(f"中位数: {np.median(raw_sentiments):.3f}")
                print(f"标准差: {raw_sentiments.std():.3f}")
                print(f"情感值分布:")
                print(f"< 0.1: {np.sum(raw_sentiments < 0.1)} 样本 (消极)")
                print(f"[0.1, 0.3]: {np.sum((raw_sentiments >= 0.1) & (raw_sentiments <= 0.3))} 样本 (中性)")
                print(f"> 0.3: {np.sum(raw_sentiments > 0.3)} 样本 (积极)")
                
                # 检查是否有有效数据
                if not text_features or not audio_features or not labels:
                    print(f"警告: {split_name} 集没有有效数据")
                    continue
                
                # 转换为numpy数组
                text_features = np.array(text_features, dtype=np.float32)
                audio_features = np.array(audio_features, dtype=np.float32)
                labels = np.array(labels, dtype=np.int64)
                
                # 保存为numpy文件
                np.save(os.path.join(ALIGNED_DIR, f'{split_name}_text.npy'), text_features)
                np.save(os.path.join(ALIGNED_DIR, f'{split_name}_audio.npy'), audio_features)
                np.save(os.path.join(ALIGNED_DIR, f'{split_name}_labels.npy'), labels)
                print(f"成功保存 {split_name} 集数据")
                print(f"文本特征形状: {text_features.shape}")
                print(f"音频特征形状: {audio_features.shape}")
                print(f"标签形状: {labels.shape}")
                # 打印标签分布
                unique, counts = np.unique(labels, return_counts=True)
                print("标签分布:")
                for label, count in zip(unique, counts):
                    print(f"类别 {label}: {count} 样本")
            except Exception as e:
                print(f"处理 {split_name} 集时出错: {str(e)}")
                continue
                
        print("\n预处理完成！")
    except Exception as e:
        print(f"处理数据集时出错: {str(e)}")

if __name__ == '__main__':
    download_and_align() 