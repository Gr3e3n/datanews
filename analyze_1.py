# -*- coding: utf-8 -*-
"""
AGILE指数分析：治理与发展匹配度诊断
完整分析脚本 v1.0

功能概述：
1. 加载并检查AGILE指数数据
2. 计算治理与发展匹配度指标
3. 划分国家类型（四象限分析）
4. 可视化分析（诊断矩阵、分布图、雷达图）
5. 关联性统计分析
6. 生成分析报告摘要

使用说明：
1. 确保已安装所需库：pandas, numpy, matplotlib, seaborn, scipy
2. 将AGILE数据CSV文件命名为 'agile_index_2025_detailed.csv' 并放在同一目录
3. 直接运行本脚本
"""

# ==================== 第一部分：环境准备与库导入 ====================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
import matplotlib

# 简单有效的字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

import matplotlib
import matplotlib.font_manager as fm


def check_and_set_font():
    """检查并设置中文字体"""
    # 查看所有可用字体
    fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = [f for f in fonts if any(c in f.lower() for c in ['yahei', 'simhei', 'microsoft', 'kai', 'song'])]

    print("可用的中文字体：")
    for font in sorted(set(chinese_fonts)):
        print(f"  - {font}")

    # 按优先级尝试不同字体
    font_candidates = [
        'Microsoft YaHei',
        'SimHei',  # 黑体
        'SimSun',  # 宋体
        'KaiTi',  # 楷体
        'FangSong',  # 仿宋
        'DejaVu Sans'  # 最后回退到这个
    ]

    for font in font_candidates:
        if font in fonts:
            plt.rcParams['font.sans-serif'] = [font]
            print(f"✓ 使用字体: {font}")
            return True

    print("⚠ 警告：未找到常用中文字体，使用默认字体")
    return False


# 在导入后立即调用
print("=" * 60)
print("AGILE指数分析：治理与发展匹配度诊断")
print("=" * 60)
print("字体检查...")
check_and_set_font()
print("=" * 60)
print("AGILE指数分析：治理与发展匹配度诊断")
print("=" * 60)
print("=" * 60)
print("AGILE指数分析：治理与发展匹配度诊断")
print("=" * 60)

def load_and_check_data():
    """加载数据并检查完整性"""
    data_file = 'data.xlsx'  # 修改为正确的文件名

    if not os.path.exists(data_file):
        print(f"错误：找不到数据文件 '{data_file}'")
        print("请确保Excel文件与脚本在同一目录")
        return None

    try:
        df = pd.read_excel(data_file)  # 使用 pd.read_excel 读取 Excel 文件
        print(f"✓ 成功加载数据：{df.shape[0]} 行，{df.shape[1]} 列")

        # 检查必要列
        required_cols = ['国家', '总分', 'P1', 'P3', 'P2', 'P4', 'D4', 'D14']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"警告：缺少以下必要列：{missing_cols}")
            return None

        print("✓ 数据列检查通过")
        print("\n数据预览：")
        print(df[['国家', '总分', '排名', 'P1', 'P3']].head())

        return df
    except Exception as e:
        print(f"加载数据时出错：{e}")
        return None

# 加载数据
df = load_and_check_data()
if df is None:
    exit()
# ==================== 第三部分：核心计算与类型划分 ====================
print("\n" + "=" * 60)
print("核心计算：治理与发展匹配度分析")
print("=" * 60)

# 1. 计算匹配度指标
df['匹配度_比值'] = df['P3'] / df['P1']
df['发展_治理_差值'] = df['P1'] - df['P3']
df['匹配度_状态'] = np.where(df['匹配度_比值'] > 1, '治理超前', '治理滞后')

# 2. 计算均值和阈值
p1_mean = df['P1'].mean()
p3_mean = df['P3'].mean()
print(f"P1 (发展水平) 平均值: {p1_mean:.2f}")
print(f"P3 (治理工具) 平均值: {p3_mean:.2f}")


# 3. 划分四象限国家类型
def classify_country(p1, p3):
    """根据P1和P3得分划分国家类型"""
    if p1 >= p1_mean and p3 >= p3_mean:
        return '全面领先型'
    elif p1 < p1_mean and p3 >= p3_mean:
        return '治理超前型'
    elif p1 >= p1_mean and p3 < p3_mean:
        return '治理滞后型'
    else:
        return '基础建设型'


df['国家类型'] = df.apply(lambda row: classify_country(row['P1'], row['P3']), axis=1)

# 4. 显示分类结果
print("\n国家类型分布：")
type_counts = df['国家类型'].value_counts()
for type_name, count in type_counts.items():
    print(f"  {type_name}: {count}个国家 ({count / len(df) * 100:.1f}%)")

# 5. 显示极端案例
print("\n极端匹配度案例：")
most_over = df.loc[df['匹配度_比值'].idxmax()]
most_under = df.loc[df['匹配度_比值'].idxmin()]
print(f"  治理最超前: {most_over['国家']} (比值={most_over['匹配度_比值']:.2f})")
print(f"  治理最滞后: {most_under['国家']} (比值={most_under['匹配度_比值']:.2f})")

# ==================== 第四部分：可视化分析 ====================
print("\n" + "=" * 60)
print("可视化分析生成中...")
print("=" * 60)

# 创建保存图片的目录
output_dir = 'analysis_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 颜色映射
type_colors = {
    '全面领先型': '#2E86AB',  # 蓝色
    '治理超前型': '#A23B72',  # 紫色
    '治理滞后型': '#F18F01',  # 橙色
    '基础建设型': '#C73E1D'  # 红色
}

# 图1：治理-发展诊断矩阵（四象限图）
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左图：四象限散点图
for ctype, color in type_colors.items():
    subset = df[df['国家类型'] == ctype]
    ax1.scatter(subset['P1'], subset['P3'],
                alpha=0.7, label=ctype, color=color, s=80, edgecolors='white', linewidth=0.5)

    # 为每个类型标注1个代表性国家
    if not subset.empty:
        rep_country = subset.iloc[0]
        ax1.annotate(rep_country['国家'],
                     (rep_country['P1'], rep_country['P3']),
                     fontsize=9, alpha=0.9, xytext=(5, 5), textcoords='offset points')

# 绘制均值线
ax1.axvline(x=p1_mean, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax1.axhline(y=p3_mean, color='gray', linestyle='--', alpha=0.7, linewidth=1)

# 添加象限标签
ax1.text(p1_mean * 1.05, p3_mean * 1.05, '全面领先型', fontsize=10, alpha=0.7)
ax1.text(p1_mean * 0.4, p3_mean * 1.05, '治理超前型', fontsize=10, alpha=0.7)
ax1.text(p1_mean * 1.05, p3_mean * 0.4, '治理滞后型', fontsize=10, alpha=0.7)
ax1.text(p1_mean * 0.4, p3_mean * 0.4, '基础建设型', fontsize=10, alpha=0.7)

ax1.set_xlabel('AI发展水平 (P1)', fontsize=12, fontweight='bold')
ax1.set_ylabel('AI治理工具 (P3)', fontsize=12, fontweight='bold')
ax1.set_title('AI治理与发展匹配度诊断矩阵', fontsize=14, fontweight='bold', pad=15)
ax1.legend(title='国家类型', loc='upper left', bbox_to_anchor=(1.02, 1))
ax1.grid(True, alpha=0.3)

# 右图：匹配度比值分布
for ctype, color in type_colors.items():
    subset = df[df['国家类型'] == ctype]
    ax2.hist(subset['匹配度_比值'], alpha=0.6, label=ctype,
             color=color, bins=12, edgecolor='white')

ax2.axvline(x=1.0, color='black', linestyle='-', linewidth=2, label='均衡线 (P3/P1=1)')
ax2.axvline(x=df['匹配度_比值'].mean(), color='darkblue', linestyle=':',
            linewidth=2, label=f'平均值 ({df["匹配度_比值"].mean():.2f})')

ax2.set_xlabel('治理-发展匹配度比值 (P3/P1)', fontsize=12, fontweight='bold')
ax2.set_ylabel('国家数量', fontsize=12, fontweight='bold')
ax2.set_title('治理投入相对超前/滞后分布', fontsize=14, fontweight='bold', pad=15)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_治理发展匹配度诊断.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 01_治理发展匹配度诊断.png")

# 图2：不同类型国家关键指标对比
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# 对比的指标
comparison_metrics = [
    ('D4', '风险暴露度 (越低风险越高)', 'rainbow_r'),
    ('D14', '社会接受度', 'viridis'),
    ('P4', '治理成效', 'plasma'),
    ('总分', 'AGILE总分', 'cool')
]

for idx, (metric, title, cmap) in enumerate(comparison_metrics):
    if metric in df.columns:
        # 按国家类型分组计算统计量
        type_stats = []
        for ctype in df['国家类型'].unique():
            values = df[df['国家类型'] == ctype][metric].dropna()
            if len(values) > 0:
                type_stats.append({
                    '类型': ctype,
                    '平均值': values.mean(),
                    '中位数': values.median(),
                    '标准差': values.std(),
                    '样本数': len(values)
                })

        stats_df = pd.DataFrame(type_stats)

        # 绘制条形图
        bars = axes[idx].bar(stats_df['类型'], stats_df['平均值'],
                             color=[type_colors[t] for t in stats_df['类型']],
                             alpha=0.8, edgecolor='black')

        # 添加误差线
        axes[idx].errorbar(stats_df['类型'], stats_df['平均值'],
                           yerr=stats_df['标准差'], fmt='none',
                           color='black', capsize=5)

        # 在柱子上标注数值
        for bar, avg in zip(bars, stats_df['平均值']):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                           f'{avg:.1f}', ha='center', va='bottom', fontsize=9)

        axes[idx].set_title(title, fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('得分', fontsize=11)
        axes[idx].tick_params(axis='x', rotation=15)
        axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/02_不同类型国家关键指标对比.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 02_不同类型国家关键指标对比.png")

# 图3：典型案例雷达图对比
fig3 = plt.figure(figsize=(10, 8))

# 选择典型案例国家
case_studies = {
    '全面领先型': '中国',
    '治理超前型': '韩国',
    '治理滞后型': '爱尔兰',
    '基础建设型': '印度'
}

# 选择对比维度
radar_categories = ['P1', 'P3', 'D4', 'D14', 'P2', 'P4']
radar_labels = ['发展水平', '治理工具', '风险暴露', '社会接受', '治理环境', '治理成效']

# 准备雷达图数据
angles = np.linspace(0, 2 * np.pi, len(radar_categories), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

ax3 = fig3.add_subplot(111, projection='polar')

# 绘制每个案例国家
for idx, (ctype, country) in enumerate(case_studies.items()):
    if country in df['国家'].values:
        country_data = df.loc[df['国家'] == country, radar_categories].values.flatten().tolist()
        country_data += country_data[:1]  # 闭合图形

        ax3.plot(angles, country_data, 'o-', linewidth=2,
                 label=f"{country} ({ctype})", color=list(type_colors.values())[idx])
        ax3.fill(angles, country_data, alpha=0.1, color=list(type_colors.values())[idx])

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(radar_labels, fontsize=11)
ax3.set_ylim(0, 100)
ax3.set_title('典型案例国家AI治理能力多维对比', fontsize=14, fontweight='bold', pad=20)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))
ax3.grid(True)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_典型案例雷达图对比.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 03_典型案例雷达图对比.png")

# 显示所有图表
plt.show()

# ==================== 第五部分：关联性统计分析 ====================
print("\n" + "=" * 60)
print("关联性统计分析")
print("=" * 60)

# 分析匹配度与关键结果指标的相关性
match_metrics = ['匹配度_比值', '发展_治理_差值']
outcome_metrics = ['D4', 'D14', 'P4', '总分']

correlation_results = []

print("匹配度指标与结果指标的相关性分析：")
print("-" * 50)

for match_metric in match_metrics:
    for outcome in outcome_metrics:
        if match_metric in df.columns and outcome in df.columns:
            # 清理数据
            valid_data = df[[match_metric, outcome]].dropna()

            if len(valid_data) >= 5:  # 确保有足够样本
                # 计算Pearson相关系数
                pearson_corr, pearson_p = stats.pearsonr(valid_data[match_metric], valid_data[outcome])

                # 计算Spearman相关系数（对异常值更稳健）
                spearman_corr, spearman_p = stats.spearmanr(valid_data[match_metric], valid_data[outcome])

                # 保存结果
                correlation_results.append({
                    '匹配度指标': match_metric,
                    '结果指标': outcome,
                    'Pearson相关系数': pearson_corr,
                    'Pearson_p值': pearson_p,
                    'Spearman相关系数': spearman_corr,
                    'Spearman_p值': spearman_p,
                    '样本量': len(valid_data)
                })

                # 打印显著结果
                is_significant = pearson_p < 0.05 or spearman_p < 0.05
                significance_star = " ***" if is_significant else ""

                outcome_name = {
                    'D4': '风险暴露度',
                    'D14': '社会接受度',
                    'P4': '治理成效',
                    '总分': '总分'
                }.get(outcome, outcome)

                print(f"{match_metric} vs {outcome_name}:")
                print(f"  Pearson: r={pearson_corr:.3f}, p={pearson_p:.3f}{significance_star}")
                print(f"  Spearman: ρ={spearman_corr:.3f}, p={spearman_p:.3f}{significance_star}")

print("\n" + "=" * 60)
print("统计检验：不同类型国家间差异")
print("=" * 60)

# 对不同国家类型的关键指标进行ANOVA检验（若样本量足够）
for metric in ['D4', 'D14', 'P4']:
    if metric in df.columns:
        groups = [df[df['国家类型'] == ctype][metric].dropna().values
                  for ctype in df['国家类型'].unique()
                  if len(df[df['国家类型'] == ctype][metric].dropna()) >= 3]

        if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
            try:
                # 方差齐性检验
                _, levene_p = stats.levene(*groups)

                # ANOVA检验
                _, anova_p = stats.f_oneway(*groups)

                metric_name = {
                    'D4': '风险暴露度',
                    'D14': '社会接受度',
                    'P4': '治理成效'
                }.get(metric, metric)

                print(f"\n{metric_name}在不同国家类型间的差异：")
                print(f"  方差齐性检验p值: {levene_p:.3f}")
                print(f"  ANOVA检验p值: {anova_p:.3f}")

                if anova_p < 0.05:
                    print(f"  → 不同国家类型在{metric_name}上存在显著差异")

                    # 如果ANOVA显著，进行事后检验（Tukey HSD）
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd

                    # 准备数据
                    data_values = []
                    data_groups = []

                    for i, ctype in enumerate(df['国家类型'].unique()):
                        values = df[df['国家类型'] == ctype][metric].dropna()
                        if len(values) >= 3:
                            data_values.extend(values)
                            data_groups.extend([ctype] * len(values))

                    if len(data_values) > 0:
                        tukey = pairwise_tukeyhsd(data_values, data_groups, alpha=0.05)
                        print("  事后检验(Tukey HSD)结果：")
                        print(tukey)
                else:
                    print(f"  → 不同国家类型在{metric_name}上无显著差异")

            except Exception as e:
                print(f"  对{metric}进行ANOVA检验时出错: {e}")

# ==================== 第六部分：生成分析报告摘要 ====================
print("\n" + "=" * 60)
print("分析报告核心发现摘要")
print("=" * 60)

# 1. 基础统计
print("\n1. 基础统计信息：")
print(f"   分析国家总数: {len(df)}")
print(f"   平均发展水平(P1): {df['P1'].mean():.2f}")
print(f"   平均治理工具(P3): {df['P3'].mean():.2f}")
print(f"   平均匹配度比值: {df['匹配度_比值'].mean():.2f}")

# 2. 国家类型特征
print("\n2. 国家类型特征分析：")

type_summary = []
for ctype in df['国家类型'].unique():
    subset = df[df['国家类型'] == ctype]

    summary = {
        '类型': ctype,
        '数量': len(subset),
        '占比': f"{len(subset) / len(df) * 100:.1f}%",
        '平均P1': subset['P1'].mean(),
        '平均P3': subset['P3'].mean(),
        '平均匹配度': subset['匹配度_比值'].mean(),
        '平均风险(D4)': subset['D4'].mean(),
        '平均社会接受(D14)': subset['D14'].mean(),
        '代表国家': subset.iloc[0]['国家']
    }
    type_summary.append(summary)

summary_df = pd.DataFrame(type_summary)
print(summary_df.to_string(index=False))

# 3. 关键发现
print("\n3. 关键发现与政策启示：")

# 找出高风险低治理的国家
high_risk_low_gov = df[(df['D4'] < df['D4'].quantile(0.25)) &  # 高风险（D4低）
                       (df['匹配度_比值'] < 0.8)]  # 治理严重滞后

if not high_risk_low_gov.empty:
    print("   * 高风险-低治理警示：")
    for _, row in high_risk_low_gov.head(3).iterrows():
        print(f"     {row['国家']}: 风险暴露度={row['D4']:.1f}, 匹配度={row['匹配度_比值']:.2f}")

# 找出高接受度高治理的国家
high_acc_high_gov = df[(df['D14'] > df['D14'].quantile(0.75)) &  # 高社会接受度
                       (df['匹配度_比值'] > 1.2)]  # 治理显著超前

if not high_acc_high_gov.empty:
    print("\n   * 高接受度-高治理典范：")
    for _, row in high_acc_high_gov.head(3).iterrows():
        print(f"     {row['国家']}: 社会接受度={row['D14']:.1f}, 匹配度={row['匹配度_比值']:.2f}")

# 4. 政策建议
print("\n4. 初步政策建议：")
print("   * 对'治理滞后型'国家：优先建立敏捷监管框架，加强AI风险监测能力")
print("   * 对'治理超前型'国家：评估监管成本效益，确保不抑制创新活力")
print("   * 对'基础建设型'国家：技术与治理能力同步建设，借鉴国际最佳实践")
print("   * 对'全面领先型'国家：推动治理标准国际化，参与全球AI治理规则制定")

# 5. 保存详细数据
print("\n" + "=" * 60)
print("数据输出")
print("=" * 60)

# 保存分析后的数据
df_analysis = df[['国家', '国家类型', '匹配度_比值', '发展_治理_差值', '匹配度_状态',
                  'P1', 'P3', 'D4', 'D14', 'P4', '总分', '排名']].copy()

# 添加排名信息
df_analysis['匹配度_排名'] = df_analysis['匹配度_比值'].rank(ascending=False).astype(int)
df_analysis['发展水平_排名'] = df_analysis['P1'].rank(ascending=False).astype(int)
df_analysis['治理工具_排名'] = df_analysis['P3'].rank(ascending=False).astype(int)

# 保存到CSV
output_csv = f'{output_dir}/analysis_results.csv'
df_analysis.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"✓ 已保存详细分析结果: {output_csv}")

# 保存相关性结果
if correlation_results:
    corr_df = pd.DataFrame(correlation_results)
    corr_csv = f'{output_dir}/correlation_analysis.csv'
    corr_df.to_csv(corr_csv, index=False, encoding='utf-8-sig')
    print(f"✓ 已保存相关性分析结果: {corr_csv}")

# 生成国家类型详细报告
type_report = []
for ctype in df['国家类型'].unique():
    subset = df[df['国家类型'] == ctype]
    top_countries = subset.nlargest(3, '总分')[['国家', '总分', '匹配度_比值', 'D4', 'D14']]

    type_report.append({
        '国家类型': ctype,
        '国家数量': len(subset),
        '发展水平范围': f"{subset['P1'].min():.1f}-{subset['P1'].max():.1f}",
        '治理工具范围': f"{subset['P3'].min():.1f}-{subset['P3'].max():.1f}",
        '代表国家': ', '.join(top_countries['国家'].tolist()),
        '主要特征': f"发展{subset['P1'].mean():.1f}/治理{subset['P3'].mean():.1f}/匹配度{subset['匹配度_比值'].mean():.2f}"
    })

report_df = pd.DataFrame(type_report)
report_csv = f'{output_dir}/country_type_summary.csv'
report_df.to_csv(report_csv, index=False, encoding='utf-8-sig')
print(f"✓ 已保存国家类型摘要: {report_csv}")

print("\n" + "=" * 60)
print("分析完成！")
print(f"所有结果已保存到 '{output_dir}' 目录")
print("=" * 60)