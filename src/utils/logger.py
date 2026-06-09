import logging
import sys
import os
import time
from contextlib import contextmanager

def setup_logger(log_dir="logs", log_filename="simulation.log", level=logging.WARNING, console_level=logging.WARNING):
    """
    配置全局日志记录器。
    
    Args:
        log_dir (str): 日志文件目录。
        log_filename (str): 日志文件名。
        level (int): 文件日志级别 (默认 WARNING)。
        console_level (int): 控制台日志级别 (默认 WARNING)。
    
    Note:
        日志级别: DEBUG < INFO < WARNING < ERROR < CRITICAL
        - WARNING: 只输出警告、错误和关键信息
        - INFO: 输出一般信息
        - DEBUG: 输出调试信息（详细）
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_path = os.path.join(log_dir, log_filename)
    
    # 获取 root logger
    logger = logging.getLogger()
    # 设置为最低级别，以便 handlers 可以自行过滤
    logger.setLevel(min(level, console_level))
    
    # 清除旧的 handlers 避免重复
    if logger.handlers:
        logger.handlers = []
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. 文件 Handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 2. 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level) 
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logging.info(f"Logger initialized. Log file: {log_path}")

def get_logger(name):
    return logging.getLogger(name)


@contextmanager
def algorithm_timer(algorithm_name, verbose=True, batch_info=None):
    """
    算法运行时间计时器上下文管理器，统一日志格式。
    
    Args:
        algorithm_name: 算法名称
        verbose: 是否打印日志
        batch_info: 批次信息字符串（如 "Batch 1/10"）
    
    Yields:
        elapsed_time: 运行时间（秒）
    
    Usage:
        with algorithm_timer("FRED-ABL", verbose=True, batch_info="Batch 1/10") as elapsed:
            result = construct_fred_abl(...)
    """
    start_time = time.time()
    if verbose:
        batch_str = f" [{batch_info}]" if batch_info else ""
        print(f"[开始] {algorithm_name}{batch_str}...", end="", flush=True)
    
    try:
        yield None
    finally:
        elapsed_time = time.time() - start_time
        if verbose:
            print(f" 完成 (耗时: {elapsed_time:.2f}秒)", flush=True)


def log_algorithm_progress(algorithm_name, current, total, additional_info="", verbose=True):
    """
    记录算法运行进度。
    
    Args:
        algorithm_name: 算法名称
        current: 当前进度
        total: 总进度
        additional_info: 额外信息（如当前最佳值）
        verbose: 是否打印
    """
    if not verbose:
        return
    
    percentage = (current / total * 100) if total > 0 else 0
    info_str = f" | {additional_info}" if additional_info else ""
    print(f"  [{algorithm_name}] 进度: {current}/{total} ({percentage:.1f}%){info_str}", flush=True)


def log_algorithm_iteration(algorithm_name, iteration, total_iterations, best_value=None, verbose=True, print_freq=10):
    """
    记录算法迭代信息（减少打印频率）。
    
    Args:
        algorithm_name: 算法名称
        iteration: 当前迭代
        total_iterations: 总迭代数
        best_value: 当前最佳值
        verbose: 是否打印
        print_freq: 打印频率（每N次打印一次）
    """
    if not verbose:
        return
    
    # 只在关键迭代打印：开始、结束、每N次、或找到新的最佳值
    should_print = (
        iteration == 0 or 
        iteration == total_iterations - 1 or 
        iteration % print_freq == 0
    )
    
    if should_print:
        best_str = f" | Best: {best_value:.4f}" if best_value is not None else ""
        print(f"  [{algorithm_name}] Iter {iteration + 1}/{total_iterations}{best_str}", flush=True)
