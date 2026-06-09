# -*- coding: utf-8 -*-
"""
在 dataset/testdata（或 dataset/testdata）下所有数据集上批量运行实验。

固定参数：attack=wgcc, batch_size=20, rate=0.1
日志简略，仅保留整体进度条与时间/预计时间。

用法:
  python alltest_main.py
  python alltest_main.py --data_dir dataset/testdata
  python alltest_main.py --skip BA-50 BA-100
  python alltest_main.py --only GtsCe Colt
  python alltest_main.py --model src/train/v1/checkpoints/curriculum_final_dynamic_gat.pth
"""
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import sys
import os
import argparse
import glob
import time

# 项目根目录加入 path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from main import main


DEFAULT_DATA_DIR = os.path.join(project_root, "dataset", "testdata")
ATTACK = "wgcc"
BATCH_SIZE = 20
RATE = 0.1


def _format_eta(seconds):
    if seconds is None or seconds < 0 or not float("inf") > seconds:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def list_datasets(data_dir):
    """列出 data_dir 下所有 .gml 文件名（不含路径与扩展名），排序后返回。"""
    if not os.path.isdir(data_dir):
        return []
    pattern = os.path.join(data_dir, "*.gml")
    paths = glob.glob(pattern)
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return sorted(names)


def run_one(dataset_name, test_mode=True, debug=False, model_path=None):
    """对单个数据集运行 main：attack=wgcc, batch_size=20, rate=0.1。"""
    main_params = [
        "main.py",
        "--dataset", dataset_name,
        "--attack", ATTACK,
        "--batch", str(BATCH_SIZE),
        "--rate", str(RATE),
    ]
    if test_mode:
        main_params.append("--test_mode")
    if debug:
        main_params.append("--debug")
    if model_path:
        main_params.extend(["--model", model_path])

    original_argv = sys.argv
    try:
        sys.argv = main_params
        main()
        return True
    except SystemExit as e:
        return (e.code == 0) if e.code is not None else True
    except Exception as e:
        print(f"运行异常 [{dataset_name}]: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = original_argv


def alltest_main():
    parser = argparse.ArgumentParser(
        description="在 dataset/testdata 所有数据集上运行 wgcc (batch_size=20, rate=0.1)，简略日志+整体进度条",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python alltest_main.py
  python alltest_main.py --data_dir dataset/testdata
  python alltest_main.py --skip BA-50 BA-100
  python alltest_main.py --only GtsCe Colt
  python alltest_main.py --model src/train/v1/checkpoints/curriculum_final_dynamic_gat.pth
        """
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="数据集目录")
    parser.add_argument("--skip", type=str, nargs="*", default=[], help="要跳过的数据集名称列表")
    parser.add_argument("--only", type=str, nargs="*", default=[], help="只运行这些数据集")
    parser.add_argument("--no_test_mode", action="store_true", help="不传 --test_mode 给 main.py")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="BiT-HyRL 使用的 GNN 模型路径 (.pth)，会传递给 main.py --model")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    all_names = list_datasets(data_dir)
    if not all_names:
        print(f"未在 {data_dir} 下找到任何 .gml 文件")
        return

    if args.only:
        names = [n for n in all_names if n in args.only]
        skipped = set(args.only) - set(names)
        if skipped:
            print(f"警告: --only 中以下数据集在目录中不存在，已忽略: {skipped}")
    else:
        names = [n for n in all_names if n not in args.skip]

    # 简略日志：由 alltest_main 调用 main 时启用
    os.environ["ALLTEST_QUIET"] = "1"

    total = len(names)
    start_all = time.time()
    ok_count = 0

    print("=" * 60)
    print("alltest_main: wgcc 全数据集 (batch=%d, rate=%.2f)" % (BATCH_SIZE, RATE))
    if args.model:
        print("BiT-HyRL 模型: %s" % args.model)
    print("数据集: %s" % (names if len(names) <= 12 else names[:6] + ["..."] + names[-3:]))
    print("=" * 60)

    if tqdm is not None:
        pbar = tqdm(
            enumerate(names, 1),
            total=total,
            unit="ds",
            desc="总进度",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            dynamic_ncols=True,
        )
        for i, dataset_name in pbar:
            pbar.set_postfix_str("当前: %s" % dataset_name)
            t0 = time.time()
            ok = run_one(dataset_name, test_mode=not args.no_test_mode, debug=args.debug, model_path=args.model)
            elapsed = time.time() - t0
            if ok:
                ok_count += 1
            # 预计剩余时间：按已用总时间 / 已完成数 估算
            done = i
            total_elapsed = time.time() - start_all
            if done > 0:
                eta = (total_elapsed / done) * (total - done)
                pbar.set_postfix_str("%s | %.0fs | 剩余~%s" % (dataset_name, elapsed, _format_eta(eta)))
    else:
        for i, dataset_name in enumerate(names, 1):
            t0 = time.time()
            ok = run_one(dataset_name, test_mode=not args.no_test_mode, debug=args.debug, model_path=args.model)
            elapsed = time.time() - t0
            if ok:
                ok_count += 1
            total_elapsed = time.time() - start_all
            eta = (total_elapsed / i) * (total - i) if i < total else 0
            print("[%d/%d] %s | %.1fs | 已用 %s | 剩余 ~%s"
                  % (i, total, dataset_name, elapsed, _format_eta(total_elapsed), _format_eta(eta)))

    total_elapsed = time.time() - start_all
    print("=" * 60)
    print("全部结束: 成功 %d/%d | 总用时 %s" % (ok_count, total, _format_eta(total_elapsed)))
    print("=" * 60)


if __name__ == "__main__":
    alltest_main()
