#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Publication-Quality Plots for LDA Analysis
创建可发表标准的LDA分析图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings("ignore")

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 设置图表尺寸和DPI
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class PublicationPlotCreator:
    def __init__(self):
        self.word_freq_data = None
        self.engagement_data = None
        
    def load_word_frequency_data(self):
        """加载词频数据"""
        # 基于更新的英文词频数据，确保没有重复键
        word_freq_dict = {
            'Feminism': 1409, 'Gender': 1107, 'Male': 953, 'Life': 646, 'Woman': 600,
            'World': 586, 'Power': 548, 'Equality': 508, 'Choice': 499, 'Like': 440,
            'Freedom': 397, 'Girl': 395, 'Man': 387, 'Independence': 375,
            'Definition': 372, 'Family': 372, 'Hope': 342, 'Misogyny': 319, 'Awaken': 316,
            'Body': 306, 'Movie': 296, 'Understanding': 282, 'Expression': 276,
            'Friend': 272, 'Rights': 269, 'Feelings': 268, 'Possessions': 259, 'Consciousness': 256,
            'Mother': 254, 'Growth': 251, 'Thinking': 245, 'Want': 245,
            'Dilemma': 239, 'Discussion': 237, 'Teacher': 235, 'Pursuit': 234,
            'Story': 231, 'Prejudice': 226, 'Experience': 222, 'Thought': 218, 'Movement': 214,
            'Sexism': 214, 'Author': 213, 'Identity': 211, 'Change': 211, 'Emotion': 204,
            'Face with': 202, 'Focus': 202, 'Stuff': 201, 'Work piece': 196,
            'Empowerment': 196, 'Mom': 195, 'Female power': 191, 'Inner': 189,
            'Marriage': 189, 'Courage': 184, 'Children': 183, 'Respect': 179,
            'Support': 178, 'Pain': 177, 'Group': 176, 'Telling': 171,
            'Discrimination': 170, 'Feeling': 164, 'Voice': 163, 'Gentleness': 163, 'Tags': 162,
            'Rejection': 161, 'Powerful': 161, 'Media': 160, 'Spirit': 159, 'Opportunity': 156,
            'Daughter': 156, 'Workplace': 156, 'Social': 151, 'Oppression': 149,
            'Men and women': 145, 'Creativity': 144, 'Politics': 143, 'Discipline': 142,
            'Impression': 141, 'Worth': 140, 'Chizuko Ueno': 138, 'Break': 138, 'Topic': 138,
            'Anxiety': 138, 'Found': 135, 'Reading': 131
        }
        
        self.word_freq_data = pd.DataFrame(list(word_freq_dict.items()), 
                                         columns=['Word', 'Frequency'])
        self.word_freq_data = self.word_freq_data.sort_values('Frequency', ascending=False)
        return self.word_freq_data
    
    def load_engagement_data(self):
        """加载engagement指标数据"""
        # 基于新图片中的正确数据，按Topic 1-5顺序排列
        engagement_dict = {
            'Topic': [1, 2, 3, 4, 5],
            'Count': [1194, 394, 344, 43, 31],
            'Likes': [4521.64, 1273.12, 2437.09, 7044.79, 2897.74],
            'Comments': [191.99, 93.71, 102.79, 278.67, 90.23],
            'Shares': [298.38, 103.77, 135.73, 366.79, 126.68],
            'Collections': [1175.02, 388.18, 543.13, 971.35, 339.58]
        }
        
        self.engagement_data = pd.DataFrame(engagement_dict)
        return self.engagement_data
    
    def create_wordcloud(self, save_path='wordcloud.png'):
        """创建词云图"""
        if self.word_freq_data is None:
            self.load_word_frequency_data()
        
        # 创建词云
        wordcloud = WordCloud(
            width=1200, 
            height=800,
            background_color='white',
            max_words=100,
            colormap='viridis',
            font_path='/System/Library/Fonts/Arial Unicode.ttf',  # 尝试使用系统字体
            relative_scaling=0.5,
            random_state=42
        )
        
        # 生成词云
        word_freq_dict = dict(zip(self.word_freq_data['Word'], self.word_freq_data['Frequency']))
        wordcloud.generate_from_frequencies(word_freq_dict)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Frequency Cloud - Feminism Discourse', 
                    fontsize=20, fontweight='bold', pad=20)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ 词云图已保存: {save_path}")
        plt.show()
        
    def create_word_frequency_plot(self, save_path='word_frequency_plot.png'):
        """创建词频纵向排序图"""
        if self.word_freq_data is None:
            self.load_word_frequency_data()
        
        # 选择前30个高频词
        top_words = self.word_freq_data.head(30)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # 创建水平条形图
        bars = ax.barh(range(len(top_words)), top_words['Frequency'], 
                      color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
        
        # 设置y轴标签
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words['Word'], fontsize=12)
        
        # 设置x轴标签
        ax.set_xlabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title('Top 30 Word Frequencies in Feminism Discourse', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 在条形上添加数值标签
        for i, (bar, freq) in enumerate(zip(bars, top_words['Frequency'])):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                   f'{freq}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # 设置网格
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        # 反转y轴，使最高频词在顶部
        ax.invert_yaxis()
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ 词频图已保存: {save_path}")
        plt.show()
        
    def create_engagement_bar_chart(self, save_path='engagement_bar_chart.png'):
        """创建engagement指标柱状图"""
        if self.engagement_data is None:
            self.load_engagement_data()
        
        # 准备数据
        topics = self.engagement_data['Topic'].tolist()
        metrics = ['Likes', 'Comments', 'Shares', 'Collections']
        
        # 设置颜色
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 设置柱状图的位置
        x = np.arange(len(topics))
        width = 0.2
        
        # 绘制每个指标的柱状图
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            values = self.engagement_data[metric].tolist()
            bars = ax.bar(x + i*width, values, width, label=metric, 
                         color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # 在柱子上添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{value:.1f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
        
        # 设置图表属性
        ax.set_xlabel('Topic Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Engagement Count', fontsize=14, fontweight='bold')
        ax.set_title('Engagement Metrics Comparison Across Topics', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f'Topic {t}' for t in topics], fontsize=12)
        ax.legend(title='Engagement Metrics', title_fontsize=12, fontsize=11)
        
        # 设置网格
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Engagement柱状图已保存: {save_path}")
        plt.show()
        
    def create_all_plots(self):
        """创建所有图表"""
        print("🎨 开始创建可发表标准的图表...")
        
        # 1. 词云图
        print("\n1️⃣ 创建词云图...")
        self.create_wordcloud()
        
        # 2. 词频图
        print("\n2️⃣ 创建词频图...")
        self.create_word_frequency_plot()
        
        # 3. Engagement柱状图
        print("\n3️⃣ 创建Engagement柱状图...")
        self.create_engagement_bar_chart()
        
        print("\n🎉 所有图表创建完成！")

def main():
    """主函数"""
    print("🎨 === 创建可发表标准的LDA分析图表 ===")
    print("=" * 50)
    
    # 创建图表创建器
    plot_creator = PublicationPlotCreator()
    
    # 创建所有图表
    plot_creator.create_all_plots()

if __name__ == "__main__":
    main() 