# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from matplotlib import font_manager
import os
import numpy as np

# 设置中文字体
# 尝试查找系统中的中文字体文件（如 SimHei），如果找到则加载，否则回退到备用字体列表。
# 这是为了解决 Matplotlib 在默认情况下无法显示中文字符的问题。
try:
    font_path = 'C:/Windows/Fonts/simhei.ttf' 
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
    else:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'sans-serif']
except:
    pass
    
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 统一的绘图样式设置函数
# ============================================================================

def setup_plot_style(ax, highlight_method="BiT-HyRL"):
    """
    统一的绘图样式设置函数，方便调整所有图像的样式。
    
    Args:
        ax: matplotlib axes 对象
        highlight_method: 需要突出的方法名称，默认 "BiT-HyRL"
    """
    # 1. 网格设置：完全取消网格
    ax.grid(False)
    
    # 2. 坐标轴标签加粗，字号加大2号
    ax.xaxis.label.set_fontsize(14)  # 原来12，现在14
    ax.yaxis.label.set_fontsize(14)  # 原来12，现在14
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    
    # 3. 刻度标签加粗，字号加大2号
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)  # 原来10，现在12
    ax.tick_params(axis='both', which='major', labelcolor='black')
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 4. 坐标轴线加粗
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    # 5. 图例设置：加大字号
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(11)  # 原来8，现在11
            text.set_fontweight('bold')

def get_method_colors_and_styles(highlight_method="BiT-HyRL"):
    """
    获取方法颜色和样式配置，突出指定方法。
    这是全局统一的颜色配置，所有图表都使用这套颜色。
    
    Args:
        highlight_method: 需要突出的方法名称，默认 "BiT-HyRL"
    
    Returns:
        tuple: (colors_dict, linestyles_dict, markers_dict)
    """
    # 统一的颜色配置（所有图表共用）- 使用高对比度颜色
    base_colors = {
        "Baseline": '#1f77b4',      # 深蓝色
        "RCP": '#2ca02c',           # 森林绿
        "GA+RL": '#9467bd',         # 紫色
        "Onion+RL": '#d62728',      # 深红色
        "ROMEN+RL": '#ff7f0e',      # 亮橙色
        "BimodalRL": '#17becf',     # 青色/蓝绿色
        "UNITY+RL": '#7f7f7f',      # 中灰色
        "BiT-HyRL": '#e377c2',      # 粉红色（突出显示，与其他颜色对比明显）
        "FRED-ABL": '#bcbd22',      # 黄绿色
        "TEAM": '#8c564b',          # 棕色
        "QDLM": '#aec7e8',          # 浅蓝色
    }
    
    # 基础线型配置
    base_linestyles = {
        "Baseline": '-',
        "RCP": '--',
        "GA+RL": '-.',
        "Onion+RL": ':',
        "ROMEN+RL": '-',
        "BimodalRL": '--',
        "UNITY+RL": '-.',
        "BiT-HyRL": '-',  # 实线，更突出
        "FRED-ABL": '--',
        "TEAM": '-.',
        "QDLM": ':',
    }
    
    # 基础标记配置
    base_markers = {
        "Baseline": 'o',
        "RCP": 's',
        "GA+RL": '^',
        "Onion+RL": 'D',
        "ROMEN+RL": 'v',
        "BimodalRL": 'p',  # 改为pentagon避免unfilled marker警告
        "UNITY+RL": '*',
        "BiT-HyRL": 'o',  # 圆形标记
        "FRED-ABL": 'h',  # 六边形
        "TEAM": 'X',      # X标记（大写，filled）
        "QDLM": 'P',      # 加号（大写，filled）
    }
    
    return base_colors, base_linestyles, base_markers


def get_method_colors_list():
    """
    获取方法颜色列表（按标准顺序）。
    用于bar图和box图的颜色配置。
    
    Returns:
        tuple: (labels_list, colors_list, markers_list)
    """
    colors_dict, _, markers_dict = get_method_colors_and_styles()
    labels = ["Baseline", "RCP", "GA+RL", "Onion+RL", "ROMEN+RL", "BimodalRL", "UNITY+RL", "BiT-HyRL", "FRED-ABL", "TEAM", "QDLM"]
    colors = [colors_dict.get(label, '#888888') for label in labels]
    markers = [markers_dict.get(label, 'o') for label in labels]
    return labels, colors, markers

def add_critical_line(ax, y_data_list=None, threshold_ratio=0.2, color='gray', linestyle='--', linewidth=1.5, alpha=0.5):
    """
    添加临界值横线（最大指标的threshold_ratio倍处）。
    
    Args:
        ax: matplotlib axes 对象
        y_data_list: Y轴数据列表（用于计算最大值），如果为None则使用固定值0.2
        threshold_ratio: 阈值比例，默认 0.2（20%）
        color: 线条颜色（默认灰色，避免红色元素太多）
        linestyle: 线型
        linewidth: 线宽
        alpha: 透明度
    """
    if y_data_list is not None:
        # 计算所有Y数据的最大值
        max_y = 0
        for y_data in y_data_list:
            if y_data:
                max_y = max(max_y, max(y_data))
        y_value = max_y * threshold_ratio
    else:
        # 如果没有提供数据，使用固定值0.2（向后兼容）
        y_value = 0.2
    
    ax.axhline(y=y_value, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=0)

def get_metric_abbreviation(metric_type):
    """
    获取指标的缩写形式。
    
    Args:
        metric_type: 指标类型 (gcc, csa, cce, wcp)
    
    Returns:
        str: 指标缩写
    """
    abbreviations = {
        'gcc': 'GCC',
        'csa': 'CSA',
        'cce': 'CCE',
        'wcp': 'WCP'
    }
    return abbreviations.get(metric_type.lower(), metric_type.upper())

def plot_line(x_ran, y_ran, x_tar, y_tar):
    """
    绘制两种攻击模式（随机攻击 vs 定向攻击）下的对比折线图。
    主要用于简单的二元对比展示。

    Args:
        x_ran (list): 随机攻击下的 X 轴数据（移除比例）。
        y_ran (list): 随机攻击下的 Y 轴数据（GCC 大小）。
        x_tar (list): 定向攻击下的 X 轴数据。
        y_tar (list): 定向攻击下的 Y 轴数据。
    """
    plt.figure()
    plt.plot(x_ran, y_ran, marker='.', color='#1f77b4', linestyle='-', label='Random')
    plt.plot(x_tar, y_tar, marker='<', color='#ff7f0e', linestyle='--', label='Target')
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['axes.unicode_minus'] = False
    plt.xlabel("Proportion of Removed Nodes : f")
    plt.ylabel("Proportion of the Giant Connected Component")
    plt.legend()
    plt.show()

def plot_different_topo(nums, X, Y, colors, linestyles, markers, labels, number, name, save_dir, y_label="Resilience"):
    """
    绘制不同拓扑或部署策略下的鲁棒性对比折线图。
    这是生成 "攻击0.png" 等核心结果图的函数。

    Args:
        nums (int): 需要绘制的曲线数量（对应不同的策略）。
        X (list of lists): 每个策略对应的 X 轴数据列表集合。
        Y (list of lists): 每个策略对应的 Y 轴数据列表集合。
        colors (list): 颜色列表，用于区分不同策略。
        linestyles (list): 线型列表。
        markers (list): 数据点标记列表。
        labels (list): 图例标签列表。
        number (int): 当前迭代编号，用于文件名后缀。
        name (str): 图表标题/文件名前缀（如 "定向攻击"）。
        save_dir (str): 保存目录。
        y_label (str): Y轴标签，默认为"Resilience"。
    """
    # 临时设置默认字体以避免在设置特定中文字体前的警告
    rcParams['font.family'] = 'sans-serif'
    
    # 使用 seaborn 设置绘图风格，使其更加美观
    sns.set_style("white", {
        'axes.facecolor': '#ffffff'
    })
    sns.set_context("notebook", font_scale=1.2)
    
    # 调整图像比例为1:1
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 获取方法颜色和样式配置（用于识别 BiT-HyRL）
    colors_dict, linestyles_dict, markers_dict = get_method_colors_and_styles()
    
    # 循环绘制每一条曲线
    for i in range(nums):
        # 检查是否是 BiT-HyRL（通过标签判断）
        is_bit_hyrl = "BiT-HyRL" in labels[i] if i < len(labels) else False
        
        # BiT-HyRL 使用更粗的线宽和更大的标记
        linewidth = 3.0 if is_bit_hyrl else 2.0
        markersize = 6 if is_bit_hyrl else 4
        
        ax.plot(X[i], Y[i],
                color=colors[i] if i < len(colors) else '#888888',
                linestyle=linestyles[i] if i < len(linestyles) else '-',
                marker=markers[i] if i < len(markers) else 'o',
                label=labels[i] if i < len(labels) else f"Method {i}",
                markersize=markersize,
                linewidth=linewidth)

    # 设置 X 轴主刻度间隔为 0.2
    x_major_locator = MultipleLocator(0.2)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_xlim(0, 1)

    # 设置坐标轴标签（加粗，字号加大）
    ax.set_xlabel("Proportion of Removed Nodes : f", fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    
    # 创建图例（必须在setup_plot_style之前）
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # 应用统一的样式设置（会更新图例样式）
    setup_plot_style(ax)
    
    # 添加临界值横线（最大指标的20%处）
    add_critical_line(ax, y_data_list=Y, threshold_ratio=0.2)

    # 构建保存路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, name + str(number) + '.png')
    
    # 保存图片，dpi=300 保证清晰度，bbox_inches='tight' 裁剪多余空白
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar_chart(data_dict, y_label, save_dir, filename):
    """
    绘制柱状图（带误差棒和散点）。
    使用统一的配色方案，1:1画布比例，不旋转x标签。
    """
    from matplotlib.font_manager import FontProperties
    try:
        font = FontProperties(family='SimHei')
    except:
        font = None
        
    sns.set_style("white", {
        'axes.facecolor': '#ffffff'
    })
    
    # 使用统一的颜色配置
    labels_order, colors_order, markers_order = get_method_colors_list()
    
    # 提取数据
    xvalues_list = []
    valid_labels = []
    valid_colors = []
    valid_markers = []
    
    for i, label in enumerate(labels_order):
        if label in data_dict:
            xvalues_list.append(data_dict[label])
            valid_labels.append(label)
            valid_colors.append(colors_order[i])
            valid_markers.append(markers_order[i])
    
    if not valid_labels:
        return

    # 调整图片比例为1:1
    fig, ax = plt.subplots(figsize=(12, 8))
    x_pos = np.arange(len(valid_labels))
    means = [np.mean(values) for values in xvalues_list]
    stds = [np.std(values) for values in xvalues_list]
    
    # 柱状图
    bars = ax.bar(x_pos, means, yerr=stds, 
                  capsize=8,  # 增大误差棒帽子
                  alpha=0.85, 
                  color=valid_colors,
                  ecolor='#2c3e50',  # 深灰色误差棒
                  width=0.65,
                  edgecolor='white',
                  linewidth=1.5)
    
    # 数值标签（增大字号）
    for i, bar in enumerate(bars):
        height = means[i]
        ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 散点图 (Jitter) - 使用对应颜色
    # 注意：unfilled markers ('x', '+') 不支持 edgecolors
    unfilled_markers = ['x', '+', '1', '2', '3', '4', '|', '_']
    for i, values in enumerate(xvalues_list):
        x_scatter = np.random.normal(x_pos[i], 0.04, size=len(values))
        marker = valid_markers[i]
        if marker in unfilled_markers:
            ax.scatter(x_scatter, values, 
                       alpha=0.6,
                       color=valid_colors[i],
                       marker=marker,
                       s=50, zorder=10)
        else:
            ax.scatter(x_scatter, values, 
                       alpha=0.5,
                       color=valid_colors[i],
                       marker=marker,
                       s=40, zorder=10, edgecolors='white', linewidths=0.5)

    # 增大所有字体
    ax.set_xlabel("Deployment Strategy", fontsize=20, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=20, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_labels, rotation=0, ha='center', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16, width=2, length=6)
    ax.tick_params(axis='x', labelsize=14, width=2, length=6)
    
    # 完全取消网格
    ax.grid(False)
    
    # 加粗坐标轴线
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=valid_colors[i], edgecolor='white', label=valid_labels[i]) 
                      for i in range(len(valid_labels))]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_box_chart(data_dict, y_label, save_dir, filename):
    """
    绘制箱线图。
    使用统一的配色方案，1:1画布比例，不旋转x标签。
    """
    sns.set_style("white", {
        'axes.facecolor': '#ffffff'
    })
    
    # 使用统一的颜色配置
    labels_order, colors_order, _ = get_method_colors_list()
    
    plot_data = []
    plot_labels = []
    palette = []
    
    for i, label in enumerate(labels_order):
        if label in data_dict:
            plot_data.append(data_dict[label])
            plot_labels.append(label)
            palette.append(colors_order[i])

    if not plot_labels:
        return

    # 调整图片比例为1:1
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 使用 boxplot
    # 注意：boxplot 的 input 如果是 list of lists，可以直接画
    bplot = ax.boxplot(plot_data, 
                        patch_artist=True,  # fill with color
                        labels=plot_labels,
                        medianprops=dict(color="#2c3e50", linewidth=2.5),
                        boxprops=dict(linewidth=2),
                        whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='#e74c3c', 
                                       markeredgecolor='white', markersize=6, 
                                       markeredgewidth=0.5, alpha=0.6))
    
    # 填充颜色
    for patch, color in zip(bplot['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('white')
        patch.set_linewidth(1.5)

    # 增大所有字体
    ax.set_xlabel("Deployment Strategy", fontsize=20, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=20, fontweight='bold')
    ax.set_xticklabels(plot_labels, rotation=0, ha='center', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16, width=2, length=6)
    ax.tick_params(axis='x', labelsize=14, width=2, length=6)
    
    # 完全取消网格
    ax.grid(False)
    
    # 加粗坐标轴线
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=palette[i], edgecolor='white', label=plot_labels[i]) 
                      for i in range(len(plot_labels))]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_curve(x_values, dataset_name, attack_mode, experiment_id, save_dir=None):
    """
    兼容旧代码的接口，实际调用新函数。
    绘制不同策略下网络完全崩溃点（移除节点比例）的柱状图。
    """
    if save_dir is None:
        save_dir = os.path.join('results', 'pictures', dataset_name, 'control', attack_mode, str(experiment_id))
        
    # 绘制崩溃点 (X values)
    plot_bar_chart(x_values, "Proportion of Removed Nodes at Collapse (f)", save_dir, "collapse_point_bar.png")
    plot_box_chart(x_values, "Proportion of Removed Nodes at Collapse (f)", save_dir, "collapse_point_box.png")

def plot_curves_for_batches(curves_data, save_dir, metric_type, attack_mode, batch_size, dataset_name):
    """
    绘制前5个batch的折线图（如果batch_size<5就全画）。
    
    Args:
        curves_data (dict): {method: [(x_list, y_list, r_value), ...]} 每个方法对应多个batch的曲线数据
        save_dir (str): 保存目录
        metric_type (str): 指标类型
        attack_mode (str): 攻击模式
        batch_size (int): batch总数
        dataset_name (str): 数据集名称
    """
    # 确定要绘制的batch数量
    num_batches_to_plot = min(5, batch_size)
    
    if num_batches_to_plot == 0:
        return
    
    # 固定顺序，BiT-HyRL 放到最后以便在图例中显示在最下面
    fixed_order = ["Baseline", "RCP", "GA+RL", "Onion+RL", "ROMEN+RL", "BimodalRL", "UNITY+RL", "FRED-ABL", "TEAM", "QDLM", "BiT-HyRL"]
    
    # 获取方法颜色和样式配置
    colors_dict, linestyles_dict, markers_dict = get_method_colors_and_styles()
    
    # 获取指标缩写
    metric_abbr = get_metric_abbreviation(metric_type)
    
    # 为每个batch绘制折线图
    for batch_idx in range(num_batches_to_plot):
        X, Y, labels = [], [], []
        colors_list = []
        linestyles_list = []
        markers_list = []
        
        for method in fixed_order:
            if method not in curves_data or len(curves_data[method]) <= batch_idx:
                continue
            
            x_list, y_list, r_value = curves_data[method][batch_idx]
            X.append(x_list)
            Y.append(y_list)
            labels.append(f"{method} (R={r_value:.4f})")
            
            # 使用配置的颜色和样式
            colors_list.append(colors_dict.get(method, '#888888'))
            linestyles_list.append(linestyles_dict.get(method, '-'))
            markers_list.append(markers_dict.get(method, 'o'))
        
        if not X:
            continue
        
        # 使用plot_different_topo绘制（图像名为metric-removed_nodes_ratio）
        plot_title = f"{metric_abbr}-removed_nodes_ratio - Batch"
        try:
            # 使用plot_different_topo绘制，文件名会自动添加batch_idx（number参数）
            plot_different_topo(len(X), X, Y, colors_list, linestyles_list, markers_list, labels, batch_idx, plot_title, save_dir, y_label=metric_abbr)
        except Exception as e:
            from src.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Plotting failed for batch {batch_idx}: {e}")

def plot_collapse_point_charts(collapse_points, save_dir, metric_type, attack_mode):
    """
    绘制崩溃点的bar图和箱线图。
    
    Args:
        collapse_points (dict): {method: [collapse_point_values]} 每个方法对应多个batch的崩溃点
        save_dir (str): 保存目录
        metric_type (str): 指标类型
        attack_mode (str): 攻击模式
    """
    # 过滤掉没有数据的方法
    filtered_data = {k: v for k, v in collapse_points.items() if v}
    
    if not filtered_data:
        return
    
    # 绘制bar图和箱线图
    plot_bar_chart(filtered_data, f"Collapse Point ({metric_type.upper()})", save_dir, f"collapse_point_bar_{attack_mode}_{metric_type}.png")
    plot_box_chart(filtered_data, f"Collapse Point ({metric_type.upper()})", save_dir, f"collapse_point_box_{attack_mode}_{metric_type}.png")

def plot_attack_steps_for_batches(attack_steps_data, save_dir, metric_type, attack_mode, batch_size, dataset_name):
    """
    绘制前5个batch的攻击步数对比折线图（如果batch_size<5就全画）。
    
    Args:
        attack_steps_data (dict): {method: [(x_list, y_list, steps_list, r_value), ...]} 每个方法对应多个batch的攻击步数数据
        save_dir (str): 保存目录
        metric_type (str): 指标类型
        attack_mode (str): 攻击模式
        batch_size (int): batch总数
        dataset_name (str): 数据集名称
    """
    # 确定要绘制的batch数量
    num_batches_to_plot = min(5, batch_size)
    
    if num_batches_to_plot == 0:
        return
    
    # 固定顺序，BiT-HyRL 放到最后以便在图例中显示在最下面
    # 支持 BiT-HyRL-{base_type} 格式的方法名称
    base_fixed_order = ["Baseline", "RCP", "GA+RL", "Onion+RL", "ROMEN+RL", "BimodalRL", "UNITY+RL", "FRED-ABL", "TEAM", "QDLM", "BiT-HyRL"]
    
    # 动态构建 fixed_order：包含所有实际存在的方法，保持基础顺序
    fixed_order = []
    # 先添加基础顺序中存在的方法
    for method in base_fixed_order:
        if method in attack_steps_data:
            fixed_order.append(method)
    
    # 添加 BiT-HyRL-* 格式的方法（按字母顺序排序）
    bit_hyrl_methods = sorted([m for m in attack_steps_data.keys() if m.startswith('BiT-HyRL-')])
    fixed_order.extend(bit_hyrl_methods)
    
    # 添加其他未在基础顺序中的方法
    for method in sorted(attack_steps_data.keys()):
        if method not in fixed_order:
            fixed_order.append(method)
    
    # 为每个batch绘制折线图
    for batch_idx in range(num_batches_to_plot):
        X, Steps, labels = [], [], []
        
        for method in fixed_order:
            if method not in attack_steps_data or len(attack_steps_data[method]) <= batch_idx:
                continue
            
            # 支持新格式 (x_list, y_list, steps_list, r_value) 和旧格式 (x_list, y_list, steps_list)
            batch_data = attack_steps_data[method][batch_idx]
            if len(batch_data) == 4:
                x_list, y_list, steps_list, r_value = batch_data
            else:
                x_list, y_list, steps_list = batch_data
                r_value = None
            
            if not steps_list or len(steps_list) == 0:
                continue
            
            # 确保 x_list 和 steps_list 长度匹配
            if len(x_list) != len(steps_list):
                # 取较短的长度，确保数据一致性
                min_len = min(len(x_list), len(steps_list))
                x_list = x_list[:min_len]
                steps_list = steps_list[:min_len]
            
            if len(x_list) == 0:
                continue
            
            X.append(x_list)
            Steps.append(steps_list)
            # 在图例中显示 R 值
            if r_value is not None:
                labels.append(f"{method} (R={r_value:.4f})")
            else:
                labels.append(f"{method}")
        
        if not X:
            continue
        
        # 定义指标名称映射
        metric_labels = {
            'gcc': 'GCC (Giant Connected Component)',
            'csa': 'CSA (Control Supply Availability)',
            'cce': 'CCE (Control Entropy)',
            'wcp': 'WCP (Weighted Control Potential)'
        }
        metric_label = metric_labels.get(metric_type, metric_type.upper())
        
        # 使用seaborn设置绘图风格（取消网格）
        sns.set_style("white", {
            'axes.facecolor': '#ffffff'
        })
        sns.set_context("notebook", font_scale=1.2)
        
        # 调整图像比例为1:1
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 获取方法颜色和样式配置
        colors_dict, linestyles_dict, markers_dict = get_method_colors_and_styles()
        
        # 获取指标缩写
        metric_abbr = get_metric_abbreviation(metric_type)
        
        # 循环绘制每一条曲线
        # 构建有效方法列表（按 fixed_order 顺序）
        valid_methods = [m for m in fixed_order if m in attack_steps_data and len(attack_steps_data[m]) > batch_idx]
        
        for i in range(len(X)):
            if i >= len(valid_methods):
                continue
            method = valid_methods[i]
            
            # 获取颜色和样式（如果方法不在配置中，尝试匹配 BiT-HyRL-* 格式）
            base_method = method
            if method.startswith('BiT-HyRL-'):
                base_method = 'BiT-HyRL'  # 使用 BiT-HyRL 的颜色和样式
            
            color = colors_dict.get(base_method, colors_dict.get(method, '#888888'))
            linestyle = linestyles_dict.get(base_method, linestyles_dict.get(method, '-'))
            marker = markers_dict.get(base_method, markers_dict.get(method, 'o'))
            
            # BiT-HyRL 及其变体使用更粗的线宽
            linewidth = 3.0 if (method == "BiT-HyRL" or method.startswith('BiT-HyRL-')) else 2.0
            markersize = 6 if (method == "BiT-HyRL" or method.startswith('BiT-HyRL-')) else 4
            
            ax.plot(X[i], Steps[i],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    label=labels[i],
                    markersize=markersize,
                    linewidth=linewidth)
        
        # 设置 X 轴主刻度间隔为 0.2
        x_major_locator = MultipleLocator(0.2)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.set_xlim(0, 1)
        
        # 设置坐标轴标签（加粗，字号加大）
        ax.set_xlabel("Proportion of Removed Nodes : f", fontsize=14, fontweight='bold')
        ax.set_ylabel(f"Attack Steps ({metric_abbr})", fontsize=14, fontweight='bold')
        # 删除标题
        
        # 创建图例（必须在setup_plot_style之前）
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        
        # 应用统一的样式设置（会更新图例样式）
        setup_plot_style(ax)
        
        # 注意：这里y轴是攻击步数，所以不需要添加y=0.2的线
        
        # 构建保存路径
        filename = f"{attack_mode} - Attack Steps ({metric_type.upper()}) - Batch{batch_idx:02d}.png"
        filepath = os.path.join(save_dir, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info(f"Attack steps plot saved: {filepath}")

def plot_metric_vs_attack_steps(attack_steps_data, save_dir, metric_type, attack_mode, batch_size, dataset_name, initial_nodes):
    """
    绘制指标值 vs 移除节点比例的折线图。
    横坐标：removed_nodes_ratio = attack_step/initial_nodes
    纵坐标：指标值（CCE/CSA/GCC/WCP）
    
    注意：此函数会基于removed_nodes_ratio和指标值重新计算 R 值。
    
    Args:
        attack_steps_data (dict): {method: [(x_list, y_list, steps_list, r_value), ...]} 每个方法对应多个batch的攻击步数数据
        save_dir (str): 保存目录
        metric_type (str): 指标类型 (gcc, csa, cce, wcp)
        attack_mode (str): 攻击模式
        batch_size (int): batch总数
        dataset_name (str): 数据集名称
        initial_nodes (int): 初始节点数
    """
    from network_metrics import compute as _nm_compute
    
    # 确定要绘制的batch数量
    num_batches_to_plot = min(5, batch_size)
    
    if num_batches_to_plot == 0:
        return
    
    # 固定顺序，BiT-HyRL 放到最后以便在图例中显示在最下面
    # 支持 BiT-HyRL-{base_type} 格式的方法名称
    base_fixed_order = ["Baseline", "RCP", "GA+RL", "Onion+RL", "ROMEN+RL", "BimodalRL", "UNITY+RL", "FRED-ABL", "TEAM", "QDLM", "BiT-HyRL"]
    
    # 动态构建 fixed_order：包含所有实际存在的方法，保持基础顺序
    fixed_order = []
    # 先添加基础顺序中存在的方法
    for method in base_fixed_order:
        if method in attack_steps_data:
            fixed_order.append(method)
    
    # 添加 BiT-HyRL-* 格式的方法（按字母顺序排序）
    bit_hyrl_methods = sorted([m for m in attack_steps_data.keys() if m.startswith('BiT-HyRL-')])
    fixed_order.extend(bit_hyrl_methods)
    
    # 添加其他未在基础顺序中的方法
    for method in sorted(attack_steps_data.keys()):
        if method not in fixed_order:
            fixed_order.append(method)
    
    # 获取方法颜色和样式配置
    colors_dict, linestyles_dict, markers_dict = get_method_colors_and_styles()
    
    # 获取指标缩写
    metric_abbr = get_metric_abbreviation(metric_type)
    
    # 为每个batch绘制折线图
    for batch_idx in range(num_batches_to_plot):
        RemovedRatios, Y, labels = [], [], []
        max_removed_ratio = 0  # 记录所有方法的最大removed_ratio
        
        # 第一遍：收集数据并找到最大攻击步数，同时重新计算 R 值
        temp_data = []
        for method in fixed_order:
            if method not in attack_steps_data or len(attack_steps_data[method]) <= batch_idx:
                continue
            
            # 支持新格式 (x_list, y_list, steps_list, r_value) 和旧格式 (x_list, y_list, steps_list)
            batch_data = attack_steps_data[method][batch_idx]
            if len(batch_data) == 4:
                x_list, y_list, steps_list, old_r_value = batch_data
            else:
                x_list, y_list, steps_list = batch_data
                old_r_value = None
            
            if not steps_list or len(steps_list) == 0:
                continue
            
            # 确保 y_list 和 steps_list 长度匹配
            if len(y_list) != len(steps_list):
                min_len = min(len(y_list), len(steps_list))
                y_list = list(y_list[:min_len])
                steps_list = list(steps_list[:min_len])
            
            if len(y_list) == 0:
                continue
            
            # 计算removed_nodes_ratio = attack_step/initial_nodes
            removed_ratio_list = [step / initial_nodes for step in steps_list]
            
            # 重新计算 R 值：基于removed_nodes_ratio和指标值
            if len(removed_ratio_list) > 1:
                r_value = _nm_compute('r_value_interpolated', removed_ratio_list, y_list, num_points=101)
            else:
                r_value = 0.0
            
            temp_data.append((method, removed_ratio_list, list(y_list), r_value))
        
        # 找到最大removed_ratio
        max_removed_ratio = 0
        for _, removed_ratio_list, _, _ in temp_data:
            if removed_ratio_list:
                max_removed_ratio = max(max_removed_ratio, max(removed_ratio_list))
        
        # 第二遍：扩展所有方法的数据到统一的removed_ratio范围
        RemovedRatios = []
        for method, removed_ratio_list, y_list, r_value in temp_data:
            # 如果当前方法的removed_ratio少于最大值，扩展数据
            if removed_ratio_list and max(removed_ratio_list) < max_removed_ratio:
                last_ratio = max(removed_ratio_list)
                last_y = y_list[-1]  # 最后一个 y 值（通常是 0 或接近 0）
                
                # 扩展到最大removed_ratio（使用线性插值）
                num_extra_points = int((max_removed_ratio - last_ratio) * initial_nodes)
                for i in range(1, num_extra_points + 1):
                    new_ratio = last_ratio + (max_removed_ratio - last_ratio) * i / num_extra_points
                    removed_ratio_list.append(new_ratio)
                    y_list.append(last_y)  # 保持最后的 y 值不变
            
            RemovedRatios.append(removed_ratio_list)
            Y.append(y_list)
            # 在图例中显示重新计算的 R 值
            labels.append(f"{method} (R={r_value:.4f})")
        
        if not RemovedRatios:
            continue
        
        # 使用seaborn设置绘图风格（取消网格）
        sns.set_style("white", {
            'axes.facecolor': '#ffffff'
        })
        sns.set_context("notebook", font_scale=1.2)
        
        # 调整图像比例为1:1
        fig, ax = plt.subplots(figsize=(12, 8))
        # 循环绘制每一条曲线，使用配置的颜色和样式
        for i, method in enumerate(fixed_order):
            if method not in [m for m, _, _, _ in temp_data]:
                continue
            
            # 找到对应的数据索引
            method_idx = None
            for idx, (m, _, _, _) in enumerate(temp_data):
                if m == method:
                    method_idx = idx
                    break
            
            if method_idx is None:
                continue
            
            # 获取颜色和样式（如果方法不在配置中，尝试匹配 BiT-HyRL-* 格式）
            base_method = method
            if method.startswith('BiT-HyRL-'):
                base_method = 'BiT-HyRL'  # 使用 BiT-HyRL 的颜色和样式
            
            color = colors_dict.get(base_method, colors_dict.get(method, '#888888'))
            linestyle = linestyles_dict.get(base_method, linestyles_dict.get(method, '-'))
            marker = markers_dict.get(base_method, markers_dict.get(method, 'o'))
            
            # BiT-HyRL 及其变体使用更粗的线宽
            linewidth = 3.0 if (method == "BiT-HyRL" or method.startswith('BiT-HyRL-')) else 2.0
            markersize = 6 if (method == "BiT-HyRL" or method.startswith('BiT-HyRL-')) else 4
            
            ax.plot(RemovedRatios[method_idx], Y[method_idx],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    label=labels[method_idx],
                    markersize=markersize,
                    linewidth=linewidth)
        
        # 设置坐标轴标签（使用缩写）
        ax.set_xlabel("Removed Nodes Ratio", fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_abbr, fontsize=14, fontweight='bold')
        # 删除标题
        ax.set_xlim(0, 1)
        
        # 创建图例（必须在setup_plot_style之前）
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        
        # 应用统一的样式设置（会更新图例样式）
        setup_plot_style(ax)
        
        # 添加临界值横线（最大指标的20%处）
        add_critical_line(ax, y_data_list=Y, threshold_ratio=0.2)
        
        # 构建保存路径
        filename = f"{attack_mode} - {metric_type.upper()} vs Removed Nodes Ratio - Batch{batch_idx:02d}.png"
        filepath = os.path.join(save_dir, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info(f"Metric vs removed nodes ratio plot saved: {filepath}")


def plot_metric_vs_attack_steps_raw(attack_steps_data, save_dir, metric_type, attack_mode, batch_size, dataset_name, initial_nodes):
    """
    绘制指标值 vs 攻击步数的折线图（原始攻击步数）。
    横坐标：attack_step（攻击步数）
    纵坐标：指标值（CCE/CSA/GCC/WCP）
    图像名：metric-attack_step
    
    注意：此函数会基于removed_nodes_ratio (attack_step/initial_nodes) 和指标值重新计算 R 值。
    
    Args:
        attack_steps_data (dict): {method: [(x_list, y_list, steps_list, r_value), ...]} 每个方法对应多个batch的攻击步数数据
        save_dir (str): 保存目录
        metric_type (str): 指标类型 (gcc, csa, cce, wcp)
        attack_mode (str): 攻击模式
        batch_size (int): batch总数
        dataset_name (str): 数据集名称
        initial_nodes (int): 初始节点数
    """
    from network_metrics import compute as _nm_compute
    
    # 确定要绘制的batch数量
    num_batches_to_plot = min(5, batch_size)
    
    if num_batches_to_plot == 0:
        return
    
    # 固定顺序，BiT-HyRL 放到最后以便在图例中显示在最下面
    # 支持 BiT-HyRL-{base_type} 格式的方法名称
    base_fixed_order = ["Baseline", "RCP", "GA+RL", "Onion+RL", "ROMEN+RL", "BimodalRL", "UNITY+RL", "FRED-ABL", "TEAM", "QDLM", "BiT-HyRL"]
    
    # 动态构建 fixed_order：包含所有实际存在的方法，保持基础顺序
    fixed_order = []
    # 先添加基础顺序中存在的方法
    for method in base_fixed_order:
        if method in attack_steps_data:
            fixed_order.append(method)
    
    # 添加 BiT-HyRL-* 格式的方法（按字母顺序排序）
    bit_hyrl_methods = sorted([m for m in attack_steps_data.keys() if m.startswith('BiT-HyRL-')])
    fixed_order.extend(bit_hyrl_methods)
    
    # 添加其他未在基础顺序中的方法
    for method in sorted(attack_steps_data.keys()):
        if method not in fixed_order:
            fixed_order.append(method)
    
    # 获取方法颜色和样式配置
    colors_dict, linestyles_dict, markers_dict = get_method_colors_and_styles()
    
    # 获取指标缩写
    metric_abbr = get_metric_abbreviation(metric_type)
    
    # 为每个batch绘制折线图
    for batch_idx in range(num_batches_to_plot):
        Steps, Y, labels = [], [], []
        max_steps = 0  # 记录所有方法的最大攻击步数
        
        # 第一遍：收集数据并找到最大攻击步数，同时重新计算 R 值
        temp_data = []
        for method in fixed_order:
            if method not in attack_steps_data or len(attack_steps_data[method]) <= batch_idx:
                continue
            
            # 支持新格式 (x_list, y_list, steps_list, r_value) 和旧格式 (x_list, y_list, steps_list)
            batch_data = attack_steps_data[method][batch_idx]
            if len(batch_data) == 4:
                x_list, y_list, steps_list, old_r_value = batch_data
            else:
                x_list, y_list, steps_list = batch_data
                old_r_value = None
            
            if not steps_list or len(steps_list) == 0:
                continue
            
            # 确保 y_list 和 steps_list 长度匹配
            if len(y_list) != len(steps_list):
                min_len = min(len(y_list), len(steps_list))
                y_list = list(y_list[:min_len])
                steps_list = list(steps_list[:min_len])
            
            if len(y_list) == 0:
                continue
            
            # 计算removed_nodes_ratio = attack_step/initial_nodes
            removed_ratio_list = [step / initial_nodes for step in steps_list]
            
            # 重新计算 R 值：基于removed_nodes_ratio和指标值
            if len(removed_ratio_list) > 1:
                r_value = _nm_compute('r_value_interpolated', removed_ratio_list, y_list, num_points=101)
            else:
                r_value = 0.0
            
            # 更新最大攻击步数
            if steps_list:
                max_steps = max(max_steps, max(steps_list))
            
            temp_data.append((method, list(steps_list), list(y_list), r_value))
        
        # 第二遍：扩展所有方法的数据到统一的攻击步数范围
        for method, steps_list, y_list, r_value in temp_data:
            # 如果当前方法的攻击步数少于最大步数，扩展数据
            if steps_list and max(steps_list) < max_steps:
                last_step = max(steps_list)
                last_y = y_list[-1]  # 最后一个 y 值（通常是 0 或接近 0）
                
                # 扩展到最大攻击步数
                for step in range(last_step + 1, max_steps + 1):
                    steps_list.append(step)
                    y_list.append(last_y)  # 保持最后的 y 值不变
            
            Steps.append(steps_list)
            Y.append(y_list)
            # 在图例中显示重新计算的 R 值
            labels.append(f"{method} (R={r_value:.4f})")
        
        if not Steps:
            continue
        
        # 使用seaborn设置绘图风格（取消网格）
        sns.set_style("white", {
            'axes.facecolor': '#ffffff'
        })
        sns.set_context("notebook", font_scale=1.2)
        
        # 调整图像比例为1:1
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 循环绘制每一条曲线，使用配置的颜色和样式
        for i, method in enumerate(fixed_order):
            if method not in [m for m, _, _, _ in temp_data]:
                continue
            
            # 找到对应的数据索引
            method_idx = None
            for idx, (m, _, _, _) in enumerate(temp_data):
                if m == method:
                    method_idx = idx
                    break
            
            if method_idx is None:
                continue
            
            # 获取颜色和样式（如果方法不在配置中，尝试匹配 BiT-HyRL-* 格式）
            base_method = method
            if method.startswith('BiT-HyRL-'):
                base_method = 'BiT-HyRL'  # 使用 BiT-HyRL 的颜色和样式
            
            color = colors_dict.get(base_method, colors_dict.get(method, '#888888'))
            linestyle = linestyles_dict.get(base_method, linestyles_dict.get(method, '-'))
            marker = markers_dict.get(base_method, markers_dict.get(method, 'o'))
            
            # BiT-HyRL 及其变体使用更粗的线宽
            linewidth = 3.0 if (method == "BiT-HyRL" or method.startswith('BiT-HyRL-')) else 2.0
            markersize = 6 if (method == "BiT-HyRL" or method.startswith('BiT-HyRL-')) else 4
            
            ax.plot(Steps[method_idx], Y[method_idx],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    label=labels[method_idx],
                    markersize=markersize,
                    linewidth=linewidth)
        
        # 设置坐标轴标签（使用缩写）
        ax.set_xlabel("Attack Steps", fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_abbr, fontsize=14, fontweight='bold')
        # 删除标题
        
        # 创建图例（必须在setup_plot_style之前）
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        
        # 应用统一的样式设置（会更新图例样式）
        setup_plot_style(ax)
        
        # 添加临界值横线（最大指标的20%处）
        add_critical_line(ax, y_data_list=Y, threshold_ratio=0.2)
        
        # 构建保存路径（图像名为metric-attack_step）
        filename = f"{metric_type.upper()}-attack_step - Batch{batch_idx:02d}.png"
        filepath = os.path.join(save_dir, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info(f"Metric vs attack steps (raw) plot saved: {filepath}")


def plot_collapse_r_charts(r_values_dict, save_dir, metric_type, attack_mode):
    """
    绘制R值的bar图和箱线图（统计curve中的R值的均值和方差）。
    
    Args:
        r_values_dict (dict): {method: [r_value1, r_value2, ...]} 每个方法对应多个batch的R值
        save_dir (str): 保存目录
        metric_type (str): 指标类型
        attack_mode (str): 攻击模式
    """
    # 过滤掉没有数据的方法
    filtered_data = {k: v for k, v in r_values_dict.items() if v}
    
    if not filtered_data:
        return
    
    # 绘制bar图和箱线图
    plot_bar_chart(filtered_data, f"R Value ({metric_type.upper()})", save_dir, f"collapse_R_bar_{attack_mode}_{metric_type}.png")
    plot_box_chart(filtered_data, f"R Value ({metric_type.upper()})", save_dir, f"collapse_R_box_{attack_mode}_{metric_type}.png")
