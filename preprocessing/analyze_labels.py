import os
import numpy as np
import matplotlib.pyplot as plt
from mmsdk import mmdatasdk

def analyze_distribution(sentiments, thresholds, name):
    """分析在给定阈值下的类别分布"""
    neg_thresh, pos_thresh = thresholds
    negative = np.sum(sentiments < neg_thresh)
    neutral = np.sum((sentiments >= neg_thresh) & (sentiments <= pos_thresh))
    positive = np.sum(sentiments > pos_thresh)
    total = len(sentiments)
    
    print(f"\n{name} (阈值: [{neg_thresh}, {pos_thresh}]):")
    print(f"负面情感 (< {neg_thresh}): {negative} 样本 ({negative/total*100:.2f}%)")
    print(f"中性情感 [{neg_thresh}, {pos_thresh}]: {neutral} 样本 ({neutral/total*100:.2f}%)")
    print(f"正面情感 (> {pos_thresh}): {positive} 样本 ({positive/total*100:.2f}%)")
    return negative, neutral, positive

def analyze_labels():
    # 设置数据目录
    DATA_DIR = "data/CMU_MOSEI"
    
    # 加载标签文件
    print("加载标签文件...")
    recipe = {
        'labels': os.path.join(DATA_DIR, 'CMU_MOSEI_Labels.csd')
    }
    
    try:
        dataset = mmdatasdk.mmdataset(recipe)
        
        # 收集所有情感值
        sentiments = []
        for vid in dataset.computational_sequences['labels'].data:
            try:
                # 获取标签数据，它是一个包含情感值的数组
                label_data = dataset.computational_sequences['labels'].data[vid]['features']
                # 计算每个样本的平均情感值
                mean_sentiment = np.mean(label_data)
                sentiments.append(mean_sentiment)
            except Exception as e:
                print(f"处理样本 {vid} 时出错: {str(e)}")
                continue
        
        sentiments = np.array(sentiments)
        
        # 打印基本统计信息
        print("\n情感值统计信息:")
        print(f"样本总数: {len(sentiments)}")
        print(f"最小值: {np.min(sentiments):.3f}")
        print(f"最大值: {np.max(sentiments):.3f}")
        print(f"平均值: {np.mean(sentiments):.3f}")
        print(f"中位数: {np.median(sentiments):.3f}")
        print(f"标准差: {np.std(sentiments):.3f}")
        
        # 尝试不同的阈值设置
        thresholds_to_try = [
            ((-0.5, 0.5), "原始阈值"),
            ((-0.2, 0.2), "较小阈值"),
            ((-0.15, 0.15), "更小阈值"),
            ((0.1, 0.3), "偏正阈值"),  # 考虑到数据整体偏正
        ]
        
        # 分析每种阈值设置下的分布
        for thresholds, name in thresholds_to_try:
            analyze_distribution(sentiments, thresholds, name)
        
        # 生成直方图
        plt.figure(figsize=(12, 6))
        plt.hist(sentiments, bins=50, edgecolor='black')
        plt.title('CMU-MOSEI 数据集情感值分布')
        plt.xlabel('情感值')
        plt.ylabel('样本数量')
        plt.grid(True, alpha=0.3)
        
        # 添加所有阈值线
        colors = ['r', 'g', 'b', 'purple']
        for (neg_thresh, pos_thresh), name in thresholds_to_try:
            color = colors.pop(0)
            plt.axvline(x=neg_thresh, color=color, linestyle='--', 
                       label=f'{name} 负阈值 ({neg_thresh})')
            plt.axvline(x=pos_thresh, color=color, linestyle='--', 
                       label=f'{name} 正阈值 ({pos_thresh})')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig('sentiment_distribution_thresholds.png', bbox_inches='tight')
        print("\n直方图已保存为 'sentiment_distribution_thresholds.png'")
        
        # 计算分位数
        percentiles = [10, 25, 50, 75, 90]
        print("\n分位数分析:")
        for p in percentiles:
            value = np.percentile(sentiments, p)
            print(f"{p}分位数: {value:.3f}")
        
    except Exception as e:
        print(f"分析标签时出错: {str(e)}")

if __name__ == "__main__":
    analyze_labels() 