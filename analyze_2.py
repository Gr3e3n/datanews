# -*- coding: utf-8 -*-
"""
政策工具组合效应分析（角度二）
识别全球AI治理模式并评估其有效性
重构版：图片保存至文件夹，不显示
"""

# ==================== 第一部分：环境准备 ====================

# ==================== 第一部分：环境准备 ====================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import os
import sys

# 简单的字体设置（与 analyze_1.py 保持一致）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")
warnings.filterwarnings('ignore')


# 检查并设置字体的函数（从 analyze_1.py 复制并简化）
def check_and_set_font():
    """检查并设置中文字体"""
    import matplotlib.font_manager as fm

    # 获取所有可用字体
    fonts = [f.name for f in fm.fontManager.ttflist]

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


# 调用字体设置函数
print("=" * 60)
print("政策工具组合效应分析（角度二）")
print("=" * 60)
print("字体检查...")
check_and_set_font()
print("=" * 60)

# ==================== 第二部分：创建输出目录 ====================
output_dir = 'policy_analysis_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"✓ 输出目录: {output_dir}")

# ==================== 第三步：数据加载与准备 ====================
print("=" * 60)
print("政策工具组合效应分析")
print("=" * 60)


def load_data():
    try:
        # 尝试读取Excel文件
        df = pd.read_excel('data.xlsx', engine='openpyxl')
        print(f"✓ 成功加载数据: {df.shape[0]}个国家, {df.shape[1]}个指标")
        return df
    except FileNotFoundError:
        print("错误: 找不到data.xlsx文件")
        return None
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None


df = load_data()
if df is None:
    exit()

# 检查必要的列
policy_dimensions = ['D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12']  # P3的7个维度
required_cols = policy_dimensions + ['国家', 'P1', 'D3', 'D4', 'D14', 'P4']

# 检查总分列是否存在
if '总分' not in df.columns and '排名' not in df.columns:
    # 尝试计算总分
    print("注意: 数据中无'总分'列，将使用P1作为替代指标")
    if 'P1' in df.columns:
        df['总分'] = df['P1']  # 使用P1作为总分替代
    else:
        print("错误: 缺少总分和P1列")
        exit()

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"警告: 缺少部分列: {missing_cols}")
    # 尝试继续，但警告用户
    for col in missing_cols:
        if col not in ['D3', 'D4', 'D14', 'P4']:
            print(f"错误: 必须列{col}缺失")
            exit()

print(f"✓ 数据列检查通过")
print(f"政策工具维度: {policy_dimensions}")

# ==================== 第四步：构建特征向量 ====================
print("\n" + "=" * 60)
print("步骤1: 构建政策工具特征向量")
print("=" * 60)

# 提取政策工具特征
policy_features = df[policy_dimensions].copy()
country_names = df['国家'].copy()

print("政策工具特征描述统计:")
print(policy_features.describe().round(2))

# ==================== 第五步：数据标准化 ====================
print("\n" + "=" * 60)
print("步骤2: 数据标准化")
print("=" * 60)

scaler = StandardScaler()
policy_scaled = scaler.fit_transform(policy_features)
policy_scaled_df = pd.DataFrame(policy_scaled, columns=policy_dimensions)
policy_scaled_df['国家'] = country_names.values

print("标准化后数据预览:")
print(policy_scaled_df.head())

# ==================== 第六步：确定最佳聚类数量 ====================
print("\n" + "=" * 60)
print("步骤3: 确定最佳聚类数量")
print("=" * 60)

# 使用肘部法则和轮廓系数确定最佳K值
inertia = []
silhouette_scores = []
k_range = range(2, 7)  # 尝试2-6个聚类

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(policy_scaled)
    inertia.append(kmeans.inertia_)

    if len(set(labels)) > 1:  # 轮廓系数需要至少2个聚类
        silhouette_scores.append(silhouette_score(policy_scaled, labels))
    else:
        silhouette_scores.append(0)

# 绘制肘部法则图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('聚类数量 (K)', fontsize=12)
axes[0].set_ylabel('惯性 (Inertia)', fontsize=12)
axes[0].set_title('肘部法则: 惯性 vs 聚类数量', fontsize=14)
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('聚类数量 (K)', fontsize=12)
axes[1].set_ylabel('轮廓系数', fontsize=12)
axes[1].set_title('轮廓系数 vs 聚类数量', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_聚类数量选择.png'), dpi=300, bbox_inches='tight')
plt.close(fig)  # 关闭图形，不在屏幕上显示
print("✓ 已保存: 01_聚类数量选择.png")

# 自动选择最佳K值（以轮廓系数为主）
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"✓ 最佳聚类数量: K = {optimal_k} (轮廓系数: {max(silhouette_scores):.3f})")

# ==================== 第七步：执行K-means聚类 ====================
print("\n" + "=" * 60)
print("步骤4: 执行K-means聚类分析")
print("=" * 60)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(policy_scaled)

# 将聚类结果添加到数据框
df['治理模式'] = cluster_labels
policy_scaled_df['治理模式'] = cluster_labels

# 统计每个聚类的国家数量
cluster_counts = df['治理模式'].value_counts().sort_index()
print("\n各治理模式包含国家数量:")
for cluster_id, count in cluster_counts.items():
    countries = df[df['治理模式'] == cluster_id]['国家'].tolist()
    print(f"  模式{cluster_id}: {count}个国家")
    print(f"    代表国家: {', '.join(countries[:3])}" + ("..." if len(countries) > 3 else ""))

# ==================== 第八步：解读治理模式 ====================
print("\n" + "=" * 60)
print("步骤5: 解读治理模式特征")
print("=" * 60)

# 维度标签（中文）
dim_labels = {
    'D6': '战略规划', 'D7': '治理机构', 'D8': '伦理原则',
    'D9': '影响评估', 'D10': '标准认证', 'D11': '立法现状',
    'D12': '国际参与'
}

# 计算每个聚类在7个维度上的平均得分
cluster_profiles = []
for cluster_id in range(optimal_k):
    cluster_data = df[df['治理模式'] == cluster_id]

    profile = {'治理模式': f'模式{cluster_id}'}
    for dim in policy_dimensions:
        profile[dim] = cluster_data[dim].mean()

    # 计算其他关键指标
    profile['国家数量'] = len(cluster_data)
    profile['代表国家'] = cluster_data.iloc[0]['国家'] if len(cluster_data) > 0 else 'N/A'
    profile['平均发展水平'] = cluster_data['P1'].mean() if 'P1' in cluster_data.columns else 0

    cluster_profiles.append(profile)

cluster_df = pd.DataFrame(cluster_profiles)
print("\n各治理模式的政策工具特征（原始得分均值）:")
print(cluster_df.round(2))


# 为每种模式命名（基于特征）
def name_cluster_mode(row):
    """根据特征为聚类命名"""
    features = {dim: row[dim] for dim in policy_dimensions}

    # 判断特征
    high_legislation = features['D11'] > 70 if 'D11' in features else False
    high_ethics = features['D8'] > 70 if 'D8' in features else False
    high_participation = features['D12'] > 70 if 'D12' in features else False
    high_standards = features['D10'] > 70 if 'D10' in features else False

    if high_legislation and high_standards:
        return "硬性监管型"
    elif high_ethics and high_participation and not high_legislation:
        return "软性引导型"
    elif high_legislation and high_ethics:
        return "平衡治理型"
    elif all(features.get(dim, 0) < 50 for dim in policy_dimensions):
        return "基础起步型"
    else:
        return "混合发展型"


# 添加模式名称
cluster_df['模式名称'] = cluster_df.apply(name_cluster_mode, axis=1)

# 将模式名称映射回原始数据
mode_name_mapping = {f'模式{i}': name for i, name in enumerate(cluster_df['模式名称'])}
df['模式名称'] = df['治理模式'].apply(lambda x: mode_name_mapping.get(f'模式{x}', f'模式{x}'))

print("\n各治理模式命名与特征:")
for _, row in cluster_df.iterrows():
    print(f"\n{row['模式名称']} ({row['治理模式']}):")
    print(f"  包含{row['国家数量']}个国家，代表: {row['代表国家']}")
    print(f"  特征: ", end="")
    high_features = []
    for dim in policy_dimensions:
        if dim in row and row[dim] > 70:
            high_features.append(f"{dim_labels.get(dim, dim)}({row[dim]:.0f})")
    if high_features:
        print(", ".join(high_features[:3]))
    else:
        print("无明显高特征")

# ==================== 第九步：可视化 - 雷达图 ====================
print("\n" + "=" * 60)
print("步骤6: 可视化分析 - 政策工具雷达图")
print("=" * 60)

# 准备雷达图数据
angles = np.linspace(0, 2 * np.pi, len(policy_dimensions), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 维度标签（中文）
dim_labels_chinese = [dim_labels.get(dim, dim) for dim in policy_dimensions]

# 颜色设置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

# 绘制每个模式的雷达图
for idx, (_, row) in enumerate(cluster_df.iterrows()):
    # 获取该模式在7个维度上的平均得分
    values = [row.get(dim, 0) for dim in policy_dimensions]
    values += values[:1]  # 闭合图形

    # 绘制线条
    ax.plot(angles, values, 'o-', linewidth=2,
            label=f"{row['模式名称']} ({row['国家数量']}国)",
            color=colors[idx % len(colors)])
    # 填充颜色
    ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

# 设置雷达图参数
ax.set_xticks(angles[:-1])
ax.set_xticklabels(dim_labels_chinese, fontsize=11)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
ax.set_title('全球AI治理模式：政策工具配置对比', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_治理模式雷达图.png'), dpi=300, bbox_inches='tight')
plt.close(fig)  # 关闭图形，不在屏幕上显示
print("✓ 已保存: 02_治理模式雷达图.png")

# ==================== 第十步：评估不同模式的有效性 ====================
print("\n" + "=" * 60)
print("步骤7: 评估不同治理模式的有效性")
print("=" * 60)

# 定义评估指标
effectiveness_metrics = {
    '风险控制': 'D4' if 'D4' in df.columns else None,
    '社会接受度': 'D14' if 'D14' in df.columns else None,
    '创新活力': 'D3' if 'D3' in df.columns else None,
    '治理成效': 'P4' if 'P4' in df.columns else None,
    '发展水平': 'P1' if 'P1' in df.columns else None
}

# 过滤掉不存在的指标
effectiveness_metrics = {k: v for k, v in effectiveness_metrics.items() if v is not None}

# 计算每个模式在各指标上的平均表现
effectiveness_results = []

for mode_name in df['模式名称'].unique():
    mode_data = df[df['模式名称'] == mode_name]

    result = {'治理模式': mode_name, '国家数量': len(mode_data)}

    for metric_name, metric_col in effectiveness_metrics.items():
        if metric_col in mode_data.columns:
            result[metric_name] = mode_data[metric_col].mean()

    effectiveness_results.append(result)

effectiveness_df = pd.DataFrame(effectiveness_results)

print("\n不同治理模式的有效性表现（平均分）:")
print(effectiveness_df.round(2))

# ==================== 第十一步：可视化 - 有效性对比条形图 ====================
print("\n" + "=" * 60)
print("步骤8: 可视化 - 有效性对比条形图")
print("=" * 60)

# 准备数据用于条形图
metrics_to_plot = [m for m in ['风险控制', '社会接受度', '创新活力', '治理成效']
                   if m in effectiveness_df.columns]

if metrics_to_plot:
    n_metrics = len(metrics_to_plot)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot[:4]):  # 最多显示4个指标
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break

        # 获取数据
        mode_names = effectiveness_df['治理模式']
        if metric == '风险控制':
            # 对于风险控制，我们可能想显示"风险水平"（100 - D4得分）
            values = 100 - effectiveness_df[metric]
            metric_display = '风险水平'  # 高分表示高风险
        else:
            values = effectiveness_df[metric]
            metric_display = metric

        # 创建条形图
        bars = ax.bar(mode_names, values,
                      color=colors[:len(mode_names)],
                      alpha=0.8, edgecolor='black')

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)

        ax.set_title(f'{metric_display}对比', fontsize=14, fontweight='bold')
        ax.set_ylabel('得分' if metric != '风险控制' else '风险水平', fontsize=12)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')

        # 特别标注最优表现
        if metric == '风险控制':
            best_idx = values.argmin()  # 风险最低
        else:
            best_idx = values.argmax()  # 得分最高

        if best_idx < len(bars):
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(2)

    # 隐藏多余的子图
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('不同AI治理模式的有效性对比', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_治理模式有效性对比.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图形，不在屏幕上显示
    print("✓ 已保存: 03_治理模式有效性对比.png")
else:
    print("警告: 无有效指标可用于绘制条形图")

# ==================== 第十二步：统计检验 ====================
print("\n" + "=" * 60)
print("步骤9: 统计显著性检验")
print("=" * 60)

if effectiveness_metrics:
    print("ANOVA方差分析结果:")
    print("-" * 50)

    for metric_name, metric_col in effectiveness_metrics.items():
        if metric_col not in df.columns:
            continue

        # 准备各组数据
        groups = []
        group_labels = []

        for mode_name in df['模式名称'].unique():
            group_data = df[df['模式名称'] == mode_name][metric_col].dropna()
            if len(group_data) >= 3:  # 每组至少3个样本
                groups.append(group_data.values)
                group_labels.append(mode_name)

        if len(groups) >= 2:
            # 执行ANOVA检验
            f_stat, p_value = stats.f_oneway(*groups)

            print(f"\n{metric_name} ({metric_col}):")
            print(f"  F统计量 = {f_stat:.3f}, p值 = {p_value:.3f}")

            if p_value < 0.05:
                print(f"  → 不同治理模式在{metric_name}上存在显著差异 (p < 0.05)")

                # 如果显著，进行事后检验（Tukey HSD）
                try:
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd

                    # 准备数据
                    data_values = []
                    data_groups = []

                    for mode_name, group_values in zip(group_labels, groups):
                        data_values.extend(group_values)
                        data_groups.extend([mode_name] * len(group_values))

                    tukey = pairwise_tukeyhsd(data_values, data_groups, alpha=0.05)
                    print(f"  事后检验(Tukey HSD)显著差异对:")
                    # 简化输出，只显示显著的比较
                    for i in range(len(tukey.groupsunique)):
                        for j in range(i + 1, len(tukey.groupsunique)):
                            idx = i * len(tukey.groupsunique) + j - (i + 1) * (i + 2) // 2
                            if idx < len(tukey.reject) and tukey.reject[idx]:
                                print(f"    {tukey.groupsunique[i]} vs {tukey.groupsunique[j]}: "
                                      f"均值差={tukey.meandiffs[idx]:.2f}, p={tukey.pvalues[idx]:.3f}")
                except ImportError:
                    print("  (需要statsmodels库进行事后检验)")
                except Exception as e:
                    print(f"  事后检验出错: {e}")
            else:
                print(f"  → 不同治理模式在{metric_name}上无显著差异")
        else:
            print(f"\n{metric_name}: 样本量不足进行ANOVA检验")
else:
    print("警告: 无有效指标可用于统计检验")

# ==================== 第十三步：典型案例分析 ====================
print("\n" + "=" * 60)
print("步骤10: 典型案例分析")
print("=" * 60)

# 从每个模式中选择一个代表性国家
representative_countries = {}

for mode_name in df['模式名称'].unique():
    mode_data = df[df['模式名称'] == mode_name]

    if len(mode_data) > 0:
        # 选择该模式下总分最高的国家作为代表
        if '总分' in mode_data.columns:
            representative_idx = mode_data['总分'].idxmax()
        else:
            representative_idx = mode_data.index[0]

        representative = mode_data.loc[representative_idx]
        representative_countries[mode_name] = representative['国家']

        print(f"\n{mode_name} 代表国家: {representative['国家']}")
        if '总分' in representative:
            print(f"  总分: {representative['总分']:.1f}")
        if '排名' in representative:
            print(f"  排名: {int(representative['排名'])}")
        if 'P1' in representative:
            print(f"  发展水平(P1): {representative['P1']:.1f}")
        print(f"  政策工具特征:")
        for dim in policy_dimensions:
            if dim in representative:
                print(f"    {dim_labels.get(dim, dim)}: {representative[dim]:.1f}")

# 可视化典型案例的政策工具配置
if representative_countries:
    n_countries = len(representative_countries)
    if n_countries > 0:
        fig, axes = plt.subplots(1, n_countries,
                                 figsize=(4 * n_countries, 5),
                                 subplot_kw=dict(polar=True))

        if n_countries == 1:
            axes = [axes]  # 确保axes是可迭代的

        for idx, (mode_name, country) in enumerate(representative_countries.items()):
            ax = axes[idx] if idx < len(axes) else None
            if ax is None:
                break

            # 获取该国数据
            country_data = df[df['国家'] == country]
            if not country_data.empty:
                country_data = country_data.iloc[0]

                # 政策工具得分
                values = [country_data.get(dim, 0) for dim in policy_dimensions]
                values += values[:1]  # 闭合

                # 绘制雷达图
                ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx % len(colors)])
                ax.fill(angles, values, alpha=0.2, color=colors[idx % len(colors)])

                # 设置图形参数
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(dim_labels_chinese, fontsize=9)
                ax.set_ylim(0, 100)
                ax.set_title(f'{country}\n({mode_name})', fontsize=12, fontweight='bold', pad=15)

        plt.suptitle('典型案例国家：政策工具配置对比', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '04_典型案例政策工具配置.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)  # 关闭图形，不在屏幕上显示
        print("✓ 已保存: 04_典型案例政策工具配置.png")

# ==================== 第十四步：路径依赖分析 ====================
print("\n" + "=" * 60)
print("步骤11: 路径依赖分析")
print("=" * 60)

print("治理模式与经济发展水平的关系:")
print("-" * 50)

# 计算每个模式的平均发展水平
for mode_name in df['模式名称'].unique():
    mode_data = df[df['模式名称'] == mode_name]

    if 'P1' in mode_data.columns:
        avg_p1 = mode_data['P1'].mean()
        p1_min = mode_data['P1'].min()
        p1_max = mode_data['P1'].max()
    else:
        avg_p1 = p1_min = p1_max = 0

    if '总分' in mode_data.columns:
        avg_total = mode_data['总分'].mean()
    else:
        avg_total = 0

    print(f"\n{mode_name}:")
    print(f"  平均发展水平(P1): {avg_p1:.1f}")
    print(f"  平均总分: {avg_total:.1f}")

    if 'P1' in mode_data.columns:
        print(f"  发展水平范围: {p1_min:.1f} - {p1_max:.1f}")

        # 分析发展水平与模式选择的关系
        if avg_p1 > 60:
            print(f"  → 主要为高发展水平国家")
        elif avg_p1 > 40:
            print(f"  → 主要为中等发展水平国家")
        else:
            print(f"  → 主要为低发展水平国家")

# ==================== 第十五步：生成分析报告 ====================
print("\n" + "=" * 60)
print("分析报告核心发现")
print("=" * 60)

print("\n1. 全球AI治理模式识别:")
for _, row in cluster_df.iterrows():
    print(f"   • {row['模式名称']}: {row['国家数量']}个国家")
    high_dims = [dim_labels.get(dim, dim) for dim in policy_dimensions
                 if dim in row and row[dim] > 70]
    if high_dims:
        print(f"     特征: {', '.join(high_dims[:3])}")
    else:
        print(f"     特征: 无明显高特征维度")

print("\n2. 各治理模式有效性对比:")
if not effectiveness_df.empty:
    best_modes = {}
    for metric in ['社会接受度', '创新活力', '治理成效']:
        if metric in effectiveness_df.columns:
            best_idx = effectiveness_df[metric].idxmax()
            if not pd.isna(best_idx):
                best_mode = effectiveness_df.loc[best_idx, '治理模式']
                best_value = effectiveness_df[metric].max()
                best_modes[metric] = (best_mode, best_value)

    for metric, (mode, value) in best_modes.items():
        print(f"   • {metric}最佳: {mode}模式 ({value:.1f}分)")

print("\n3. 关键政策启示:")
print("   • '硬性监管型'模式: 在风险控制方面可能更有效，但需评估对创新的影响")
print("   • '软性引导型'模式: 可能更有利于建立社会信任和国际合作")
print("   • 模式选择应与国家发展阶段相匹配")
print("   • 不存在'放之四海而皆准'的最优模式，需结合国情定制")

print("\n4. 对中国政策的启示:")
china_data = df[df['国家'] == '中国']
if not china_data.empty:
    china_mode = china_data.iloc[0]['模式名称'] if '模式名称' in china_data.columns else '未知'
    print(f"   • 中国属于'{china_mode}'治理模式")

    if len(policy_dimensions) > 0:
        china_high_dims = []
        china_low_dims = []

        for dim in policy_dimensions:
            if dim in china_data.columns:
                value = china_data[dim].values[0]
                if value > 80:
                    china_high_dims.append(dim_labels.get(dim, dim))
                elif value < 50:
                    china_low_dims.append(dim_labels.get(dim, dim))

        if china_high_dims:
            print(f"   • 优势维度: {', '.join(china_high_dims)}")

        if china_low_dims:
            print(f"   • 可加强维度: {', '.join(china_low_dims)}")
        else:
            print("   • 各维度表现均衡")

# ==================== 第十六步：保存结果 ====================
print("\n" + "=" * 60)
print("数据输出")
print("=" * 60)

# 保存聚类结果
clustering_cols = ['国家', '治理模式', '模式名称'] + policy_dimensions
for col in ['总分', 'P1', 'D3', 'D4', 'D14', 'P4']:
    if col in df.columns:
        clustering_cols.append(col)

clustering_results = df[clustering_cols]
clustering_results.to_csv(os.path.join(output_dir, '聚类分析结果.csv'),
                          index=False, encoding='utf-8-sig')
print(f"✓ 已保存: {output_dir}/聚类分析结果.csv")

# 保存模式特征
cluster_df.to_csv(os.path.join(output_dir, '治理模式特征.csv'),
                  index=False, encoding='utf-8-sig')
print(f"✓ 已保存: {output_dir}/治理模式特征.csv")

# 保存有效性分析结果
if not effectiveness_df.empty:
    effectiveness_df.to_csv(os.path.join(output_dir, '治理模式有效性.csv'),
                            index=False, encoding='utf-8-sig')
    print(f"✓ 已保存: {output_dir}/治理模式有效性.csv")

# 保存统计检验结果（简化版）
stats_results = []
for metric_name, metric_col in effectiveness_metrics.items():
    if metric_col in df.columns:
        stats_results.append({
            '指标': metric_name,
            '列名': metric_col,
            '备注': 'ANOVA结果见控制台输出'
        })

if stats_results:
    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(os.path.join(output_dir, '统计检验摘要.csv'),
                    index=False, encoding='utf-8-sig')
    print(f"✓ 已保存: {output_dir}/统计检验摘要.csv")

print(f"\n分析完成! 所有结果已保存到 '{output_dir}' 目录")
print("生成的图片:")
for file in os.listdir(output_dir):
    if file.endswith('.png'):
        print(f"  • {file}")
print("=" * 60)