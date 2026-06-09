# -*- coding: utf-8 -*-
"""
BiT-HyRL 训练进度可视化模块。

功能:
  - 实时训练进度条
  - 实时训练曲线可视化
  - 训练统计信息展示
"""
import os
import sys
import time
from datetime import timedelta
from typing import Optional, List, Dict, Any

# 尝试导入 rich（高级终端美化）
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# tqdm 总是可用
from tqdm import tqdm


class TrainingProgress:
    """训练进度管理器"""
    
    def __init__(self, total_epochs: int, total_graphs: int = 0, 
                 use_rich: bool = True, show_live_plot: bool = False):
        """
        初始化训练进度管理器。
        
        Args:
            total_epochs: 总训练轮数
            total_graphs: 每轮训练的图数量
            use_rich: 是否使用 rich 库（更美观的进度条）
            show_live_plot: 是否显示实时训练曲线（需要图形界面）
        """
        self.total_epochs = total_epochs
        self.total_graphs = total_graphs
        self.use_rich = use_rich and RICH_AVAILABLE
        self.show_live_plot = show_live_plot
        
        # 训练历史
        self.history: List[Dict[str, Any]] = []
        self.start_time = None
        self.current_epoch = 0
        self.best_reward = -float('inf')
        
        # Rich 组件
        self.console = Console() if self.use_rich else None
        self.progress = None
        self.epoch_task = None
        self.graph_task = None
        
        # 实时绘图
        self.fig = None
        self.ax = None
        
    def start(self):
        """开始训练计时"""
        self.start_time = time.time()
        
        if self.use_rich:
            self._print_header()
            
    def _print_header(self):
        """打印训练头部信息"""
        if self.use_rich:
            self.console.print(Panel.fit(
                "[bold cyan]BiT-HyRL Training[/bold cyan]\n"
                f"Total Epochs: {self.total_epochs} | Graphs per Epoch: {self.total_graphs}",
                title="Training Started",
                border_style="cyan"
            ))
        else:
            print("=" * 60)
            print("BiT-HyRL Training")
            print(f"Total Epochs: {self.total_epochs} | Graphs per Epoch: {self.total_graphs}")
            print("=" * 60)
    
    def create_epoch_progress(self):
        """创建 epoch 级别的进度条"""
        if self.use_rich:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self.console,
                transient=False
            )
            self.epoch_task = self.progress.add_task(
                "[cyan]Training", 
                total=self.total_epochs
            )
            return self.progress
        else:
            return tqdm(
                total=self.total_epochs,
                desc="Training",
                unit="epoch",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
    
    def create_graph_progress(self, desc: str = "Graphs"):
        """创建图级别的进度条（用于每个 epoch 内）"""
        if self.use_rich and self.progress:
            self.graph_task = self.progress.add_task(
                f"[green]{desc}",
                total=self.total_graphs,
                visible=True
            )
        else:
            return tqdm(
                total=self.total_graphs,
                desc=desc,
                unit="graph",
                leave=False,
                ncols=80
            )
    
    def update_epoch(self, epoch: int, avg_reward: float, 
                     additional_info: Optional[Dict] = None):
        """
        更新 epoch 进度。
        
        Args:
            epoch: 当前 epoch
            avg_reward: 平均奖励
            additional_info: 额外信息
        """
        self.current_epoch = epoch
        
        # 更新最佳奖励
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            is_best = True
        else:
            is_best = False
        
        # 记录历史
        record = {
            'epoch': epoch,
            'avg_reward': avg_reward,
            'best_reward': self.best_reward,
            'is_best': is_best,
            'time': time.time() - self.start_time if self.start_time else 0
        }
        if additional_info:
            record.update(additional_info)
        self.history.append(record)
        
        # 更新进度条
        if self.use_rich and self.progress:
            self.progress.update(self.epoch_task, advance=1)
            
            # 每10轮或最后一轮打印详细信息
            if epoch % 10 == 0 or epoch == self.total_epochs or is_best:
                best_marker = " [bold green]★ BEST[/bold green]" if is_best else ""
                self.console.print(
                    f"  [dim]Epoch {epoch:3d}[/dim] | "
                    f"Reward: [cyan]{avg_reward:.4f}[/cyan] | "
                    f"Best: [green]{self.best_reward:.4f}[/green]{best_marker}"
                )
    
    def update_graph(self, count: int = 1):
        """更新图处理进度"""
        if self.use_rich and self.progress and self.graph_task is not None:
            self.progress.update(self.graph_task, advance=count)
    
    def reset_graph_progress(self):
        """重置图级别进度条"""
        if self.use_rich and self.progress and self.graph_task is not None:
            self.progress.reset(self.graph_task)
    
    def finish(self):
        """完成训练，打印总结"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        if self.use_rich:
            # 创建总结表格
            table = Table(title="Training Summary", show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="dim")
            table.add_column("Value", justify="right")
            
            table.add_row("Total Epochs", str(self.total_epochs))
            table.add_row("Total Time", str(timedelta(seconds=int(elapsed))))
            table.add_row("Best Reward", f"{self.best_reward:.4f}")
            table.add_row("Final Reward", f"{self.history[-1]['avg_reward']:.4f}" if self.history else "N/A")
            table.add_row("Avg Time/Epoch", f"{elapsed/self.total_epochs:.2f}s" if self.total_epochs > 0 else "N/A")
            
            self.console.print()
            self.console.print(table)
            self.console.print()
        else:
            print("\n" + "=" * 60)
            print("Training Summary")
            print("=" * 60)
            print(f"Total Epochs:    {self.total_epochs}")
            print(f"Total Time:      {timedelta(seconds=int(elapsed))}")
            print(f"Best Reward:     {self.best_reward:.4f}")
            if self.history:
                print(f"Final Reward:    {self.history[-1]['avg_reward']:.4f}")
            print("=" * 60)
    
    def get_history(self) -> List[Dict]:
        """获取训练历史"""
        return self.history


class SimpleProgress:
    """简化版进度条（使用 tqdm）"""
    
    def __init__(self, total: int, desc: str = "", unit: str = "it"):
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        self.start_time = time.time()
    
    def update(self, n: int = 1, postfix: Optional[Dict] = None):
        """更新进度"""
        self.pbar.update(n)
        if postfix:
            self.pbar.set_postfix(postfix)
    
    def set_description(self, desc: str):
        """设置描述"""
        self.pbar.set_description(desc)
    
    def close(self):
        """关闭进度条"""
        self.pbar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def create_progress_bar(total: int, desc: str = "", unit: str = "it", 
                       use_rich: bool = True) -> SimpleProgress:
    """
    创建简单进度条的工厂函数。
    
    Args:
        total: 总数
        desc: 描述
        unit: 单位
        use_rich: 是否使用 rich（当前版本统一使用 tqdm）
        
    Returns:
        SimpleProgress 实例
    """
    return SimpleProgress(total, desc, unit)


def print_training_status(epoch: int, total_epochs: int, 
                         avg_reward: float, best_reward: float,
                         elapsed_time: float, graphs_processed: int = 0):
    """
    打印训练状态（无进度条版本，用于日志）。
    
    Args:
        epoch: 当前 epoch
        total_epochs: 总 epochs
        avg_reward: 平均奖励
        best_reward: 最佳奖励
        elapsed_time: 已用时间（秒）
        graphs_processed: 已处理图数量
    """
    eta = (elapsed_time / epoch) * (total_epochs - epoch) if epoch > 0 else 0
    
    status = (
        f"Epoch {epoch:4d}/{total_epochs} | "
        f"Reward: {avg_reward:.4f} | "
        f"Best: {best_reward:.4f} | "
        f"Time: {timedelta(seconds=int(elapsed_time))} | "
        f"ETA: {timedelta(seconds=int(eta))}"
    )
    
    if graphs_processed > 0:
        status += f" | Graphs: {graphs_processed}"
    
    print(status)


# 导出
__all__ = [
    'TrainingProgress',
    'SimpleProgress', 
    'create_progress_bar',
    'print_training_status',
    'RICH_AVAILABLE'
]
