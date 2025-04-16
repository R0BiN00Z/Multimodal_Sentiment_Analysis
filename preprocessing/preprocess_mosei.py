import os
import numpy as np
from mmsdk import mmdatasdk
from tqdm import tqdm

def download_and_align():
    # 设置数据目录
    DATA_DIR = "data/CMU_MOSEI"
    ALIGNED_DIR = os.path.join(DATA_DIR, "aligned")
    os.makedirs(ALIGNED_DIR, exist_ok=True)
    
    # 只下载文本和音频特征
    csd_files = {
        'glove_vectors': 'CMU_MOSEI_TimestampedWordVectors.csd',
        'COVAREP': 'CMU_MOSEI_COVAREP.csd',
        'All Labels': 'CMU_MOSEI_Labels.csd'
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
        'All Labels': os.path.join(DATA_DIR, csd_files['All Labels'])
    }
    
    dataset = mmdatasdk.mmdataset(recipe)
    
    # 对齐特征
    print("对齐特征...")
    dataset.align('All Labels', collapse_functions=[np.mean])
    
    # 保存对齐后的特征
    print("保存对齐后的特征...")
    splits = ['train', 'valid', 'test']
    for split in splits:
        print(f"处理 {split} 集...")
        split_data = dataset.computational_sequences['All Labels'].data[split]
        
        text_features = []
        audio_features = []
        labels = []
        
        for vid, data in tqdm(split_data.items()):
            # 获取文本特征
            text = dataset.computational_sequences['glove_vectors'].data[split][vid]['features']
            text_features.append(text)
            
            # 获取音频特征
            audio = dataset.computational_sequences['COVAREP'].data[split][vid]['features']
            audio_features.append(audio)
            
            # 获取标签
            label = data['features']
            labels.append(label)
        
        # 保存为numpy文件
        np.save(os.path.join(ALIGNED_DIR, f'{split}_text.npy'), np.array(text_features))
        np.save(os.path.join(ALIGNED_DIR, f'{split}_audio.npy'), np.array(audio_features))
        np.save(os.path.join(ALIGNED_DIR, f'{split}_labels.npy'), np.array(labels))
    
    print("预处理完成！")

if __name__ == '__main__':
    download_and_align() 