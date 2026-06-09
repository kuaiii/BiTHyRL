import csv
import os
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_x_values_to_csv(x_values, name, filename="x_values_results.csv", save_dir="results"):
    """
    将 x_values 数据保存到 CSV 文件。

    参数:
        x_values (dict): 包含每种网络类型对应的 y=0 时 x 值列表。
        name (str): 文件名前缀。
        filename (str): 保存的 CSV 文件名，默认是 "x_values_results.csv"。
        save_dir (str): 保存目录，默认为 "results"。
    """
    # 按照固定的顺序保存，确保一致性，BiT-HyRL 放到最后
    fixed_order = ["Baseline", "RCP", "Bimodal+Random", "Random+RL", "GA+RL", "Onion+Ra", "Onion+RL", "SOLO+Ra", "SOLO+RL", "ROMEN+Ra", "ROMEN+RL", "BiT-HyRL"]
    
    # 只保存存在于 x_values 中的列，但保持固定顺序
    headers = [h for h in fixed_order if h in x_values]
    # 添加剩余的列（如果有其他方法）
    for k in x_values.keys():
        if k not in headers:
            headers.append(k)
    
    # 获取数据的最大长度（因为可能某些列表长度不同）
    max_length = 0
    if x_values:
        max_length = max(len(values) for values in x_values.values())
    
    # 创建一个二维列表，按列对齐数据，不足的部分填充为空值
    rows = []
    for i in range(max_length):
        row = [x_values[header][i] if i < len(x_values[header]) else "" for header in headers]
        rows.append(row)
    
    # 确保 save_dir 目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = os.path.join(save_dir, name + "_" + filename)
    # 写入 CSV 文件
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(headers)
        # 写入数据
        writer.writerows(rows)
    
    logger.info(f"x_values data saved to file: {path}")


def save_statistics_to_csv(x_values, save_dir, filename="statistics.csv"):
    """
    计算并保存平均值和标准差到 CSV。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    path = os.path.join(save_dir, filename)
    abs_path = os.path.abspath(path)
    
    headers = ["Method", "Mean", "Std"]
    rows = []
    
    # 同样使用固定顺序，确保与绘图一致，BiT-HyRL 放到最后
    fixed_order = ["Baseline", "RCP", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL", "BiT-HyRL"]
    
    # 处理固定顺序的方法
    for method in fixed_order:
        if method in x_values:
            values = x_values[method]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                rows.append([method, f"{mean_val:.4f}", f"{std_val:.4f}"])
            else:
                rows.append([method, "N/A", "N/A"])
                
    # 处理其他可能的方法
    for method, values in x_values.items():
        if method not in fixed_order:
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                rows.append([method, f"{mean_val:.4f}", f"{std_val:.4f}"])
            else:
                rows.append([method, "N/A", "N/A"])
            
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)
        
    logger.info(f"Statistics saved to file: {abs_path}")

def save_comprehensive_metrics(metrics_summary, save_dir, attack_mode):
    """
    保存综合指标 (CSA, H_C, WCP, R) 到 CSV 文件。
    计算所有迭代的平均值。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = "comprehensive_metrics.csv"
    path = os.path.join(save_dir, filename)
    
    # 定义表头
    headers = ["Method", "Nodes", "Edges", "Top-10 Degrees", "CSA", "Control Entropy (H_C)", "Weighted Control Potential (WCP)", f"Robustness (R - {attack_mode})"]
    
    rows = []
    
    # 按照固定顺序输出，BiT-HyRL 放到最后
    fixed_order = ["Baseline", "RCP", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL", "BiT-HyRL"]
    
    # 处理每个方法
    for method in fixed_order:
        if method in metrics_summary:
            data_list = metrics_summary[method]
            if not data_list:
                continue
            
            # 计算初始指标的平均值（从曲线的第一个点）
            avg_csa = np.mean([d['CSA_Curve'][0][0] if d['CSA_Curve'] else 0 for d in data_list])
            avg_hc = np.mean([d['CCE_Curve'][0][0] if d['CCE_Curve'] else 0 for d in data_list])
            avg_wcp = np.mean([d['WCP_Curve'][0][0] if d['WCP_Curve'] else 0 for d in data_list])
            avg_r = np.mean([d['R'] for d in data_list])
            
            # 获取拓扑属性（假设每次迭代拓扑结构相似，取最后一次的值）
            last_entry = data_list[-1]
            nodes = last_entry.get('Nodes', 0)
            edges = last_entry.get('Edges', 0)
            top10 = last_entry.get('Top10_Degrees', "[]")
            
            rows.append([
                method,
                nodes,
                edges,
                top10,
                f"{avg_csa:.4f}",
                f"{avg_hc:.4f}",
                f"{avg_wcp:.4f}",
                f"{avg_r:.4f}"
            ])
            
    # 处理其他可能的方法
    for method, data_list in metrics_summary.items():
        if method not in fixed_order:
             if not data_list: continue
             avg_csa = np.mean([d['CSA_Curve'][0][0] if d['CSA_Curve'] else 0 for d in data_list])
             avg_hc = np.mean([d['CCE_Curve'][0][0] if d['CCE_Curve'] else 0 for d in data_list])
             avg_wcp = np.mean([d['WCP_Curve'][0][0] if d['WCP_Curve'] else 0 for d in data_list])
             avg_r = np.mean([d['R'] for d in data_list])
             
             last_entry = data_list[-1]
             nodes = last_entry.get('Nodes', 0)
             edges = last_entry.get('Edges', 0)
             top10 = last_entry.get('Top10_Degrees', "[]")
             
             rows.append([method, nodes, edges, top10, f"{avg_csa:.4f}", f"{avg_hc:.4f}", f"{avg_wcp:.4f}", f"{avg_r:.4f}"])
            
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)
        
    logger.info(f"Comprehensive metrics saved to: {path}")

def save_execution_times(execution_times, save_dir):
    """
    保存算法运行时间统计到 CSV 文件。
    
    参数:
        execution_times (dict): key是阶段名称, value是时间列表(每次迭代的时间)。
        save_dir (str): 保存目录。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    path = os.path.join(save_dir, "execution_times.csv")
    
    headers = ["Stage", "Mean Time (s)", "Total Time (s)", "Min Time (s)", "Max Time (s)", "Iterations"]
    rows = []
    
    for stage, times in execution_times.items():
        if not times:
            continue
            
        mean_time = np.mean(times)
        total_time = np.sum(times)
        min_time = np.min(times)
        max_time = np.max(times)
        count = len(times)
        
        rows.append([
            stage,
            f"{mean_time:.4f}",
            f"{total_time:.4f}",
            f"{min_time:.4f}",
            f"{max_time:.4f}",
            count
        ])
        
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)
        
    logger.info(f"Execution times saved to: {path}")

def save_curve_data(curves_data, save_dir, metric_type, attack_mode):
    """
    保存曲线数据（不同batch_size的x, y值）到CSV文件。
    
    参数:
        curves_data (dict): {method: [(x_list, y_list, r_value), ...]} 每个方法对应多个batch的曲线数据
        save_dir (str): 保存目录
        metric_type (str): 指标类型
        attack_mode (str): 攻击模式
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 固定顺序，BiT-HyRL 放到最后
    fixed_order = ["Baseline", "RCP", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL", "BiT-HyRL"]
    
    # 为每个方法保存曲线数据
    for method in fixed_order:
        if method not in curves_data or not curves_data[method]:
            continue
        
        # 保存每个batch的曲线数据
        for batch_idx, (x_list, y_list, r_value) in enumerate(curves_data[method]):
            filename = f"{method}_batch{batch_idx}_{attack_mode}_{metric_type}.csv"
            path = os.path.join(save_dir, filename)
            
            with open(path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["x_val", "y_val", "r_value"])
                for x, y in zip(x_list, y_list):
                    writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{r_value:.6f}"])
    
    logger.info(f"Curve data saved to: {save_dir}")

def save_collapse_point_data(collapse_points, save_dir, metric_type, attack_mode):
    """
    保存崩溃点数据（每个batch_size下下降到20%时的移除节点比例）到CSV文件。
    
    参数:
        collapse_points (dict): {method: [collapse_point_values]} 每个方法对应多个batch的崩溃点
        save_dir (str): 保存目录
        metric_type (str): 指标类型
        attack_mode (str): 攻击模式
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    path = os.path.join(save_dir, f"collapse_points_{attack_mode}_{metric_type}.csv")
    
    # 固定顺序，BiT-HyRL 放到最后
    fixed_order = ["Baseline", "RCP", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL", "BiT-HyRL"]
    
    # 计算最大batch数量
    max_batches = max(len(v) for v in collapse_points.values()) if collapse_points else 0
    headers = ["Method"] + [f"Batch_{i}" for i in range(max_batches)]
    rows = []
    
    for method in fixed_order:
        if method not in collapse_points or not collapse_points[method]:
            continue
        
        row = [method] + [f"{val:.6f}" for val in collapse_points[method]]
        rows.append(row)
    
    if rows:
        with open(path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)
        
        logger.info(f"Collapse point data saved to: {path}")

def save_area_data(area_values, save_dir, metric_type, attack_mode):
    """
    保存面积数据（曲线下面积值）到CSV文件，用于比较不同算法的平均性能。
    
    参数:
        area_values (dict): {method: [area_values]} 每个方法对应多个batch的面积值
        save_dir (str): 保存目录
        metric_type (str): 指标类型
        attack_mode (str): 攻击模式
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    path = os.path.join(save_dir, f"area_values_{attack_mode}_{metric_type}.csv")
    
    # 固定顺序，BiT-HyRL 放到最后；未在列表中的方法追加到末尾（如 WGCC 的 GA+RCP, FRED-ABL, QDLM）
    fixed_order = ["Baseline", "RCP", "GA+RCP", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL", "BiT-HyRL", "FRED-ABL", "QDLM"]
    order = fixed_order + [k for k in (area_values or {}) if k not in fixed_order]
    
    # 计算最大batch数量
    max_batches = max(len(v) for v in area_values.values()) if area_values else 0
    headers = ["Method", "Mean_Area", "Std_Area"] + [f"Batch_{i}" for i in range(max_batches)]
    rows = []
    
    for method in order:
        if method not in area_values or not area_values[method]:
            continue
        
        values = area_values[method]
        mean_area = np.mean(values)
        std_area = np.std(values)
        
        row = [method, f"{mean_area:.6f}", f"{std_area:.6f}"] + [f"{val:.6f}" for val in values]
        rows.append(row)
    
    if rows:
        with open(path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)
        
        logger.info(f"Area data saved to: {path}")

def save_r_values_data(r_values, save_dir, metric_type, attack_mode):
    """
    保存R值数据到CSV文件（用于collapse_R文件夹）。
    
    参数:
        r_values (dict): {method: [r_values]} 每个方法对应多个batch的R值
        save_dir (str): 保存目录
        metric_type (str): 指标类型
        attack_mode (str): 攻击模式
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    path = os.path.join(save_dir, f"r_values_{attack_mode}_{metric_type}.csv")
    
    # 固定顺序，BiT-HyRL 放到最后；未在列表中的方法追加到末尾（如 WGCC 的 GA+RCP, FRED-ABL, QDLM）
    fixed_order = ["Baseline", "RCP", "GA+RCP", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL", "BiT-HyRL", "FRED-ABL", "QDLM"]
    order = fixed_order + [k for k in (r_values or {}) if k not in fixed_order]
    
    # 计算最大batch数量
    max_batches = max(len(v) for v in r_values.values()) if r_values else 0
    headers = ["Method", "Mean_R", "Std_R"] + [f"Batch_{i}" for i in range(max_batches)]
    rows = []
    
    for method in order:
        if method not in r_values or not r_values[method]:
            continue
        
        values = r_values[method]
        mean_r = np.mean(values)
        std_r = np.std(values)
        
        row = [method, f"{mean_r:.6f}", f"{std_r:.6f}"] + [f"{val:.6f}" for val in values]
        rows.append(row)
    
    if rows:
        with open(path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)
        
        logger.info(f"R values data saved to: {path}")

def save_decomposition_results(attack_steps_data, save_dir, metric_type, attack_mode, initial_nodes):
    """
    保存每种算法的分解结果（攻击过程中的详细数据）到CSV文件。
    
    Args:
        attack_steps_data (dict): {method: [(x_list, y_list, steps_list, r_value), ...]} 
                                 每个方法对应多个batch的攻击步数数据
        save_dir (str): 保存目录
        metric_type (str): 指标类型 (gcc, csa, cce, wcp)
        attack_mode (str): 攻击模式
        initial_nodes (int): 初始节点数
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 为每个方法保存分解结果
    for method_name, batch_data_list in attack_steps_data.items():
        if not batch_data_list:
            continue
        
        # 创建CSV文件：每个方法一个文件
        filename = f"{method_name}_{metric_type}_{attack_mode}_decomposition.csv"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['Batch', 'Attack_Step', 'Removed_Ratio', 'Metric_Value', 'R_Value'])
            
            # 写入每个batch的数据
            for batch_idx, batch_data in enumerate(batch_data_list):
                if len(batch_data) < 3:
                    continue
                
                x_list, y_list, steps_list = batch_data[0], batch_data[1], batch_data[2]
                r_value = batch_data[3] if len(batch_data) > 3 else 0.0
                
                # 计算removed_ratio
                removed_ratio_list = [step / initial_nodes for step in steps_list] if steps_list else x_list
                
                # 写入每一行的数据
                for i in range(len(x_list)):
                    attack_step = steps_list[i] if i < len(steps_list) else int(x_list[i] * initial_nodes)
                    removed_ratio = removed_ratio_list[i] if i < len(removed_ratio_list) else x_list[i]
                    metric_value = y_list[i] if i < len(y_list) else 0.0
                    
                    writer.writerow([
                        batch_idx,
                        attack_step,
                        f"{removed_ratio:.6f}",
                        f"{metric_value:.6f}",
                        f"{r_value:.6f}"
                    ])
        
        logger.info(f"保存 {method_name} 的分解结果到: {filepath}")


def save_attack_steps_data(attack_steps_data, save_dir, metric_type, attack_mode):
    """
    保存攻击步数数据到CSV文件，用于分析不同算法在不同指标下的攻击步数。
    
    参数:
        attack_steps_data (dict): {method: [(x_list, y_list, steps_list, r_value), ...]} 每个方法对应多个batch的攻击步数数据
        save_dir (str): 保存目录
        metric_type (str): 指标类型
        attack_mode (str): 攻击模式
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 固定顺序，BiT-HyRL 放到最后
    fixed_order = ["Baseline", "RCP", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL", "BiT-HyRL"]

    # 为每个方法保存攻击步数数据
    for method in fixed_order:
        if method not in attack_steps_data or not attack_steps_data[method]:
            continue
        
        # 保存每个batch的攻击步数数据
        for batch_idx, batch_data in enumerate(attack_steps_data[method]):
            # 支持新格式 (x_list, y_list, steps_list, r_value) 和旧格式 (x_list, y_list, steps_list)
            if len(batch_data) == 4:
                x_list, y_list, steps_list, r_value = batch_data
            else:
                x_list, y_list, steps_list = batch_data
                r_value = None
            filename = f"{method}_batch{batch_idx}_{attack_mode}_{metric_type}_attack_steps.csv"
            path = os.path.join(save_dir, filename)
            
            with open(path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["x_val", "y_val", "attack_step", "r_value"])
                for x, y, step in zip(x_list, y_list, steps_list):
                    r_val_str = f"{r_value:.6f}" if r_value is not None else ""
                    writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{step}", r_val_str])
    
    # 保存汇总数据（每个方法在不同batch下的崩溃点攻击步数）
    summary_path = os.path.join(save_dir, f"attack_steps_summary_{attack_mode}_{metric_type}.csv")
    
    # 计算每个方法在崩溃点（20%阈值）时的攻击步数
    summary_rows = []
    summary_headers = ["Method"]
    
    # 计算最大batch数量
    max_batches = max(len(v) for v in attack_steps_data.values()) if attack_steps_data else 0
    summary_headers.extend([f"Batch_{i}_CollapseStep" for i in range(max_batches)])
    summary_headers.append("Mean_CollapseStep")
    summary_headers.append("Std_CollapseStep")
    
    for method in fixed_order:
        if method not in attack_steps_data or not attack_steps_data[method]:
            continue
        
        collapse_steps = []
        for batch_idx, batch_data in enumerate(attack_steps_data[method]):
            # 支持新格式 (x_list, y_list, steps_list, r_value) 和旧格式 (x_list, y_list, steps_list)
            if len(batch_data) == 4:
                x_list, y_list, steps_list, _ = batch_data
            else:
                x_list, y_list, steps_list = batch_data
            if not y_list or not steps_list:
                continue
            
            # 找到崩溃点（y降到初始值的20%）
            initial_value = y_list[0] if y_list[0] > 0 else 1.0
            threshold = initial_value * 0.2
            
            collapse_step = None
            for i, y_val in enumerate(y_list):
                if y_val <= threshold:
                    if i < len(steps_list):
                        collapse_step = steps_list[i]
                    break
            
            if collapse_step is None and steps_list:
                collapse_step = steps_list[-1]
            
            if collapse_step is not None:
                collapse_steps.append(collapse_step)
        
        if collapse_steps:
            mean_step = np.mean(collapse_steps)
            std_step = np.std(collapse_steps)
            row = [method] + [f"{step}" if step is not None else "N/A" for step in collapse_steps]
            # 补齐到最大batch数
            while len(row) < max_batches + 1:
                row.append("N/A")
            row.extend([f"{mean_step:.2f}", f"{std_step:.2f}"])
            summary_rows.append(row)
    
    if summary_rows:
        with open(summary_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(summary_headers)
            writer.writerows(summary_rows)
        
        logger.info(f"Attack steps summary saved to: {summary_path}")
    
    logger.info(f"Attack steps data saved to: {save_dir}")

def save_metric_log(gcc_value, wcp_value, method_name, iteration_idx, x_remove_ratio, 
                    base_dir, dataset_name, attack_mode, experiment_id):
    """
    保存WCP和GCC指标计算结果的日志到工作目录中。
    
    参数:
        gcc_value (float): GCC指标值
        wcp_value (float): WCP指标值
        method_name (str): 方法名称
        iteration_idx (int): 迭代索引
        x_remove_ratio (float): 节点移除比例
        base_dir (str): 基础目录路径 (results/{dataset_name}/{attack_mode})
        dataset_name (str): 数据集名称
        attack_mode (str): 攻击模式
        experiment_id (int): 实验ID
    """
    # 构建指标目录路径: results/{dataset_name}/{attack_mode}/{experiment_id}/{metric_type}/
    experiment_dir = os.path.join(base_dir, str(experiment_id))
    
    # 为GCC和WCP分别创建日志文件
    for metric_type, metric_value in [('gcc', gcc_value), ('wcp', wcp_value)]:
        metric_dir = os.path.join(experiment_dir, metric_type)
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)
        
        # 日志文件名格式: metric_log_{method_name}.txt
        log_filename = f"metric_log_{method_name}.txt"
        log_path = os.path.join(metric_dir, log_filename)
        
        # 追加模式写入日志
        with open(log_path, mode="a", encoding="utf-8") as f:
            # 格式: iteration_idx, x_remove_ratio, metric_value
            f.write(f"Iteration_{iteration_idx}, x={x_remove_ratio:.6f}, {metric_type.upper()}={metric_value:.6f}\n")
