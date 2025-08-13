#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic Engagement Analysis Script
分析不同主题在社交媒体指标上的表现差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, kruskal, f_oneway, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TopicEngagementAnalyzer:
    def __init__(self):
        self.df = None
        self.results = {}
        
    def load_data(self, filename):
        """加载LDA结果数据"""
        try:
            self.df = pd.read_excel(filename, sheet_name='Document Topic Assignment')
            print(f"✅ 成功加载数据: {len(self.df)} 条记录")
            
            # 数据清理：处理中文数字格式
            engagement_cols = ['liked_count', 'comment_count', 'share_count', 'collected_count']
            
            for col in engagement_cols:
                if col in self.df.columns:
                    # 清理数据：将中文数字转换为数字
                    self.df[col] = self.df[col].astype(str).apply(self._convert_chinese_number)
                    # 转换为数值类型，无法转换的设为NaN
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    print(f"📊 {col}: 清理后有效数据 {self.df[col].notna().sum()}/{len(self.df)} 条")
            
            # 修正topic标号：将0-4映射到1-5
            self.df['dominant_topic'] = self.df['dominant_topic'] + 1
            
            print(f"📊 主题分布 (修正后标号):")
            print(self.df['dominant_topic'].value_counts().sort_index())
            return True
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def _convert_chinese_number(self, value):
        """转换中文数字格式为数字"""
        if pd.isna(value) or value == '':
            return np.nan
        
        value = str(value).strip()
        
        # 处理"万"单位
        if '万' in value:
            # 提取数字部分
            import re
            numbers = re.findall(r'[\d.]+', value)
            if numbers:
                try:
                    num = float(numbers[0])
                    return num * 10000  # 万转换为具体数字
                except:
                    return np.nan
        
        # 处理"千"单位
        elif '千' in value:
            import re
            numbers = re.findall(r'[\d.]+', value)
            if numbers:
                try:
                    num = float(numbers[0])
                    return num * 1000  # 千转换为具体数字
                except:
                    return np.nan
        
        # 处理纯数字
        elif value.replace('.', '').replace('-', '').isdigit():
            try:
                return float(value)
            except:
                return np.nan
        
        # 处理包含"+"的情况
        elif '+' in value:
            # 提取第一个数字
            import re
            numbers = re.findall(r'[\d.]+', value)
            if numbers:
                try:
                    return float(numbers[0])
                except:
                    return np.nan
        
        # 其他情况返回NaN
        return np.nan
    
    def descriptive_statistics(self):
        """描述性统计分析"""
        print("\n📊 === 描述性统计分析 ===")
        
        # 按主题分组的描述性统计
        engagement_cols = ['liked_count', 'comment_count', 'share_count', 'collected_count']
        
        for col in engagement_cols:
            print(f"\n🔍 {col} 统计:")
            grouped = self.df.groupby('dominant_topic')[col]
            stats = grouped.agg(['count', 'mean', 'std', 'median', 'min', 'max'])
            print(stats.round(2))
            
            # 保存到结果字典
            self.results[f'{col}_descriptive'] = stats
    
    def normality_test(self):
        """正态性检验"""
        print("\n📈 === 正态性检验 (Shapiro-Wilk) ===")
        
        engagement_cols = ['liked_count', 'comment_count', 'share_count', 'collected_count']
        normality_results = {}
        
        for col in engagement_cols:
            print(f"\n🔍 {col}:")
            col_results = {}
            
            for topic in sorted(self.df['dominant_topic'].unique()):
                data = self.df[self.df['dominant_topic'] == topic][col].dropna()
                if len(data) > 3:  # 至少需要3个数据点
                    stat, p_value = shapiro(data)
                    print(f"  Topic {topic}: p={p_value:.4f} {'✅' if p_value > 0.05 else '❌'}")
                    col_results[topic] = {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}
                else:
                    print(f"  Topic {topic}: 数据不足")
                    col_results[topic] = {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}
            
            normality_results[col] = col_results
        
        self.results['normality_test'] = normality_results
        return normality_results
    
    def variance_homogeneity_test(self):
        """方差齐性检验"""
        print("\n📊 === 方差齐性检验 (Levene) ===")
        
        engagement_cols = ['liked_count', 'comment_count', 'share_count', 'collected_count']
        variance_results = {}
        
        for col in engagement_cols:
            print(f"\n🔍 {col}:")
            
            # 准备各组数据
            groups = []
            group_labels = []
            
            for topic in sorted(self.df['dominant_topic'].unique()):
                data = self.df[self.df['dominant_topic'] == topic][col].dropna()
                if len(data) > 1:
                    groups.append(data)
                    group_labels.append(f'Topic_{topic}')
            
            if len(groups) > 1:
                stat, p_value = levene(*groups)
                print(f"  Levene检验: p={p_value:.4f} {'✅' if p_value > 0.05 else '❌'}")
                variance_results[col] = {'statistic': stat, 'p_value': p_value, 'is_homogeneous': p_value > 0.05}
            else:
                print("  数据不足，无法进行方差齐性检验")
                variance_results[col] = {'statistic': np.nan, 'p_value': np.nan, 'is_homogeneous': False}
        
        self.results['variance_test'] = variance_results
        return variance_results
    
    def mean_difference_test(self):
        """均值差异检验"""
        print("\n🎯 === 均值差异检验 ===")
        
        engagement_cols = ['liked_count', 'comment_count', 'share_count', 'collected_count']
        difference_results = {}
        
        for col in engagement_cols:
            print(f"\n🔍 {col}:")
            
            # 准备各组数据
            groups = []
            group_labels = []
            
            for topic in sorted(self.df['dominant_topic'].unique()):
                data = self.df[self.df['dominant_topic'] == topic][col].dropna()
                if len(data) > 1:
                    groups.append(data)
                    group_labels.append(f'Topic_{topic}')
            
            if len(groups) > 1:
                # 检查是否所有组都正态分布
                normality_results = self.results.get('normality_test', {}).get(col, {})
                all_normal = all(result.get('is_normal', False) for result in normality_results.values())
                
                # 检查方差齐性
                variance_results = self.results.get('variance_test', {}).get(col, {})
                is_homogeneous = variance_results.get('is_homogeneous', False)
                
                if all_normal and is_homogeneous:
                    # 使用ANOVA
                    stat, p_value = f_oneway(*groups)
                    test_type = 'ANOVA'
                    print(f"  ANOVA: p={p_value:.4f} {'✅' if p_value < 0.05 else '❌'}")
                else:
                    # 使用Kruskal-Wallis
                    stat, p_value = kruskal(*groups)
                    test_type = 'Kruskal-Wallis'
                    print(f"  Kruskal-Wallis: p={p_value:.4f} {'✅' if p_value < 0.05 else '❌'}")
                
                # 事后多重比较
                post_hoc_results = None
                if p_value < 0.05:
                    print("  📋 事后多重比较:")
                    if test_type == 'ANOVA':
                        # Tukey HSD
                        tukey = pairwise_tukeyhsd(self.df[col], self.df['dominant_topic'])
                        print(tukey)
                        post_hoc_results = tukey
                    else:
                        # 对于非参数检验，进行两两比较
                        print("  两两比较 (Mann-Whitney U):")
                        for i in range(len(groups)):
                            for j in range(i+1, len(groups)):
                                stat, p_val = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                                print(f"    {group_labels[i]} vs {group_labels[j]}: p={p_val:.4f}")
                
                difference_results[col] = {
                    'test_type': test_type,
                    'statistic': stat,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05,
                    'post_hoc': post_hoc_results
                }
            else:
                print("  数据不足，无法进行差异检验")
                difference_results[col] = {
                    'test_type': 'N/A',
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'is_significant': False,
                    'post_hoc': None
                }
        
        self.results['difference_test'] = difference_results
        return difference_results
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("\n📊 === 创建可视化图表 ===")
        
        engagement_cols = ['liked_count', 'comment_count', 'share_count', 'collected_count']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(engagement_cols):
            ax = axes[i]
            
            # 箱线图
            sns.boxplot(x='dominant_topic', y=col, data=self.df, ax=ax)
            ax.set_title(f'{col} by Topic', fontsize=14, fontweight='bold')
            ax.set_xlabel('Topic', fontsize=12)
            ax.set_ylabel(col, fontsize=12)
            
            # 添加均值点
            means = self.df.groupby('dominant_topic')[col].mean()
            for j, topic in enumerate(sorted(self.df['dominant_topic'].unique())):
                if topic in means.index:
                    ax.scatter(j, means[topic], color='red', s=100, zorder=5, label='Mean' if j == 0 else "")
        
        plt.tight_layout()
        plt.savefig('topic_engagement_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ 可视化图表已保存: topic_engagement_analysis.png")
        plt.close()
        
        # 创建小提琴图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(engagement_cols):
            ax = axes[i]
            
            # 小提琴图
            sns.violinplot(x='dominant_topic', y=col, data=self.df, ax=ax)
            ax.set_title(f'{col} Distribution by Topic', fontsize=14, fontweight='bold')
            ax.set_xlabel('Topic', fontsize=12)
            ax.set_ylabel(col, fontsize=12)
        
        plt.tight_layout()
        plt.savefig('topic_engagement_distribution.png', dpi=300, bbox_inches='tight')
        print("✅ 分布图已保存: topic_engagement_distribution.png")
        plt.close()
    
    def save_results(self, filename='topic_engagement_analysis_results.xlsx'):
        """保存分析结果到Excel文件"""
        print(f"\n💾 === 保存分析结果 ===")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 1. 描述性统计
            for col in ['liked_count', 'comment_count', 'share_count', 'collected_count']:
                if f'{col}_descriptive' in self.results:
                    self.results[f'{col}_descriptive'].to_excel(writer, sheet_name=f'{col}_descriptive')
            
            # 2. 正态性检验结果
            normality_df = pd.DataFrame()
            for col in ['liked_count', 'comment_count', 'share_count', 'collected_count']:
                if 'normality_test' in self.results and col in self.results['normality_test']:
                    col_results = self.results['normality_test'][col]
                    for topic, result in col_results.items():
                        normality_df.loc[f'{col}_Topic_{topic}', 'statistic'] = result['statistic']
                        normality_df.loc[f'{col}_Topic_{topic}', 'p_value'] = result['p_value']
                        normality_df.loc[f'{col}_Topic_{topic}', 'is_normal'] = result['is_normal']
            
            if not normality_df.empty:
                normality_df.to_excel(writer, sheet_name='normality_test')
            
            # 3. 方差齐性检验结果
            variance_df = pd.DataFrame()
            for col in ['liked_count', 'comment_count', 'share_count', 'collected_count']:
                if 'variance_test' in self.results and col in self.results['variance_test']:
                    result = self.results['variance_test'][col]
                    variance_df.loc[col, 'statistic'] = result['statistic']
                    variance_df.loc[col, 'p_value'] = result['p_value']
                    variance_df.loc[col, 'is_homogeneous'] = result['is_homogeneous']
            
            if not variance_df.empty:
                variance_df.to_excel(writer, sheet_name='variance_test')
            
            # 4. 均值差异检验结果
            difference_df = pd.DataFrame()
            for col in ['liked_count', 'comment_count', 'share_count', 'collected_count']:
                if 'difference_test' in self.results and col in self.results['difference_test']:
                    result = self.results['difference_test'][col]
                    difference_df.loc[col, 'test_type'] = result['test_type']
                    difference_df.loc[col, 'statistic'] = result['statistic']
                    difference_df.loc[col, 'p_value'] = result['p_value']
                    difference_df.loc[col, 'is_significant'] = result['is_significant']
            
            if not difference_df.empty:
                difference_df.to_excel(writer, sheet_name='difference_test')
            
            # 5. 原始数据
            self.df.to_excel(writer, sheet_name='raw_data', index=False)
        
        print(f"✅ 分析结果已保存: {filename}")
    
    def run_analysis(self, input_filename):
        """运行完整分析流程"""
        print("🔍 === 主题参与度分析 ===")
        print("=" * 50)
        
        # 1. 加载数据
        if not self.load_data(input_filename):
            return False
        
        # 2. 描述性统计
        self.descriptive_statistics()
        
        # 3. 正态性检验
        self.normality_test()
        
        # 4. 方差齐性检验
        self.variance_homogeneity_test()
        
        # 5. 均值差异检验
        self.mean_difference_test()
        
        # 6. 创建可视化
        self.create_visualizations()
        
        # 7. 保存结果
        self.save_results()
        
        print("\n🎉 === 分析完成 ===")
        return True

def main():
    """主函数"""
    analyzer = TopicEngagementAnalyzer()
    
    # 使用指定的LDA结果文件
    target_file = 'optimized_lda_results_a0.1_b0.5_t5.xlsx'
    
    if os.path.exists(target_file):
        print(f"📁 使用指定的LDA结果文件: {target_file}")
        analyzer.run_analysis(target_file)
    else:
        print(f"❌ 指定的文件不存在: {target_file}")
        print("💡 请确保文件存在于当前目录中")
        
        # 如果指定文件不存在，显示可用的文件
        import glob
        lda_files = glob.glob('optimized_lda_results_*.xlsx')
        if lda_files:
            print(f"📋 当前目录可用的LDA结果文件:")
            for file in lda_files:
                print(f"  - {file}")
        else:
            print("❌ 当前目录没有找到任何LDA结果文件")

if __name__ == "__main__":
    import os
    main() 