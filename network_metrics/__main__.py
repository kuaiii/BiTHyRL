# -*- coding: utf-8 -*-
"""
network_metrics 自测试脚本

运行方式:
    python -m network_metrics

当前仅测试 structural 指标（functional 指标待后续补充测试）。
"""
import networkx as nx
import random

from network_metrics import compute, list_metrics
from network_metrics.structural import gcc, robustness


def generate_test_graph(n=100, m=3, seed=42):
    """生成测试用的 BA 网络。"""
    return nx.barabasi_albert_graph(n, m, seed=seed)


def generate_attack_sequence(G, mode='degree', seed=42):
    """生成攻击序列。"""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if mode == 'random':
        rng.shuffle(nodes)
        return nodes
    elif mode == 'degree':
        # 动态高度数攻击：按初始度数降序
        degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        return [n for n, d in degrees]
    else:
        raise ValueError(f"Unknown attack mode: {mode}")


def test_gcc():
    """测试 GCC 相关指标。"""
    print("\n" + "=" * 60)
    print("[Test 1] GCC (Giant Connected Component)")
    print("=" * 60)

    G = generate_test_graph(n=100, m=3)
    n_init = G.number_of_nodes()

    gcc_val = compute('gcc_size', G)
    ratio = compute('gcc_ratio', G, initial_nodes=n_init)
    second = compute('second_lcc', G)
    all_sizes = compute('all_component_sizes', G)

    print(f"  初始节点数      : {n_init}")
    print(f"  GCC 大小        : {gcc_val}")
    print(f"  GCC 占比        : {ratio:.4f}")
    print(f"  次大 LCC 大小   : {second}")
    print(f"  连通分量数量    : {len(all_sizes)}")
    print(f"  前5个分量大小   : {all_sizes[:5]}")

    # 验证：BA 网络应该是连通的，所以 GCC = n_init
    assert gcc_val == n_init, "BA(100,3) 应该是连通的"
    assert ratio == 1.0, "连通图 GCC 占比应为 1.0"
    assert second == 0, "连通图不应有次大分量"
    print("  ✓ GCC 测试通过")


def test_attack_trajectory():
    """测试攻击轨迹（AUC 相关）。"""
    print("\n" + "=" * 60)
    print("[Test 2] Attack Trajectory")
    print("=" * 60)

    G = generate_test_graph(n=100, m=3)
    n_init = G.number_of_nodes()
    attack_seq = generate_attack_sequence(G, mode='degree')

    traj = compute('attack_trajectory', G, attack_sequence=attack_seq, stop_threshold=0.1)

    print(f"  初始节点数      : {n_init}")
    print(f"  攻击序列长度    : {len(attack_seq)}")
    print(f"  轨迹记录数      : {len(traj)}")
    print(f"  轨迹首条        : {traj[0]}")
    print(f"  轨迹末条        : {traj[-1]}")

    # 验证轨迹格式
    assert len(traj) >= 2, "轨迹应至少包含初始状态和一次攻击"
    assert traj[0][0] == 0, "首条应为初始状态 (removed=0)"
    assert traj[0][1] == 0.0, "首条移除比例应为 0.0"
    assert traj[0][2] == n_init, "首条 GCC 应为初始节点数"

    # 验证停止条件：最后一步的 GCC <= 10% 初始节点数
    final_gcc = traj[-1][2]
    stop_size = max(1, int(n_init * 0.1))
    assert final_gcc <= stop_size, f"应在 GCC<={stop_size} 时停止，实际 GCC={final_gcc}"

    # 打印中间几个点
    step = max(1, len(traj) // 5)
    print(f"  轨迹采样(每{step}步):")
    for i in range(0, len(traj), step):
        r = traj[i]
        print(f"    Step {r[0]:3d}: 移除比例={r[1]:.3f}, GCC={r[2]:3d}, 次大LCC={r[3]:3d}")

    print("  ✓ Attack Trajectory 测试通过")


def test_auc():
    """测试 AUC 计算。"""
    print("\n" + "=" * 60)
    print("[Test 3] AUC (Area Under Curve)")
    print("=" * 60)

    G = generate_test_graph(n=100, m=3)
    n_init = G.number_of_nodes()
    attack_seq = generate_attack_sequence(G, mode='degree')

    traj = compute('attack_trajectory', G, attack_sequence=attack_seq, stop_threshold=0.1)
    auc_val = compute('auc', trajectory=traj, initial_nodes=n_init)

    print(f"  AUC 值          : {auc_val:.6f}")

    # 完全鲁棒的图 AUC 接近 1.0，脆弱图较低
    # BA 网络在 HDA 下 AUC 通常在中等偏低水平
    assert 0.0 < auc_val < 1.0, "AUC 应在 (0, 1) 之间"
    print("  ✓ AUC 测试通过")


def test_r_value():
    """测试 R 值计算。"""
    print("\n" + "=" * 60)
    print("[Test 4] R-value (Robustness)")
    print("=" * 60)

    G = generate_test_graph(n=100, m=3)
    n_init = G.number_of_nodes()
    attack_seq = generate_attack_sequence(G, mode='degree')

    traj = compute('attack_trajectory', G, attack_sequence=attack_seq, stop_threshold=0.1)

    x_curve = [t[1] for t in traj]
    y_curve = [t[2] / n_init for t in traj]

    r_interp = compute('r_value', x_curve=x_curve, y_curve=y_curve, num_points=101)
    r_simple = compute('r_value_simple', x_curve=x_curve, y_curve=y_curve)

    print(f"  R-value (插值)  : {r_interp:.6f}")
    print(f"  R-value (简单)  : {r_simple:.6f}")

    assert 0.0 < r_interp < 1.0, "R-value 应在 (0, 1) 之间"
    assert 0.0 < r_simple < 1.0, "R-value 应在 (0, 1) 之间"
    print("  ✓ R-value 测试通过")


def test_gcc_after_removal():
    """测试节点移除后的 GCC 变化。"""
    print("\n" + "=" * 60)
    print("[Test 5] GCC after Node Removal")
    print("=" * 60)

    G = generate_test_graph(n=50, m=2)
    n_init = G.number_of_nodes()

    # 移除 20% 的节点
    nodes_to_remove = list(G.nodes())[:10]
    H = G.copy()
    H.remove_nodes_from(nodes_to_remove)

    gcc_before = compute('gcc_size', G)
    gcc_after = compute('gcc_size', H)
    ratio_after = compute('gcc_ratio', H, initial_nodes=n_init)

    print(f"  移除前 GCC      : {gcc_before}")
    print(f"  移除节点数      : {len(nodes_to_remove)}")
    print(f"  移除后 GCC      : {gcc_after}")
    print(f"  移除后 GCC占比  : {ratio_after:.4f}")

    assert gcc_after <= gcc_before, "移除节点后 GCC 不应增大"
    print("  ✓ GCC after Removal 测试通过")


def test_gcc_snapshot():
    """测试 GCC 快照（一次性返回四个值）。"""
    print("\n" + "=" * 60)
    print("[Test 6] GCC Snapshot (LCC, LCC_ratio, SLCC, SLCC_ratio)")
    print("=" * 60)

    G = generate_test_graph(n=100, m=3)
    n_init = G.number_of_nodes()

    snapshot = compute('gcc_snapshot', G, initial_nodes=n_init)
    lcc, lcc_ratio, slcc, slcc_ratio = snapshot

    print(f"  LCC 大小        : {lcc}")
    print(f"  LCC 占比        : {lcc_ratio:.4f}")
    print(f"  SLCC 大小       : {slcc}")
    print(f"  SLCC 占比       : {slcc_ratio:.4f}")

    assert lcc == n_init, "连通图 LCC 应为初始节点数"
    assert lcc_ratio == 1.0, "连通图 LCC 占比应为 1.0"
    assert slcc == 0, "连通图 SLCC 应为 0"
    assert slcc_ratio == 0.0, "连通图 SLCC 占比应为 0.0"
    print("  ✓ GCC Snapshot 测试通过")

    # 测试移除节点后的快照
    H = G.copy()
    H.remove_nodes_from(list(G.nodes())[:30])
    snapshot2 = compute('gcc_snapshot', H, initial_nodes=n_init)
    print(f"\n  移除30个节点后:")
    print(f"    LCC 大小      : {snapshot2[0]}")
    print(f"    LCC 占比      : {snapshot2[1]:.4f}")
    print(f"    SLCC 大小     : {snapshot2[2]}")
    print(f"    SLCC 占比     : {snapshot2[3]:.4f}")
    assert snapshot2[1] < 1.0, "移除节点后 LCC 占比应下降"
    print("  ✓ GCC Snapshot (after removal) 测试通过")


def test_attack_robustness():
    """测试 attack_robustness（一次性返回轨迹和AUC）。"""
    print("\n" + "=" * 60)
    print("[Test 7] Attack Robustness (trajectory + AUC)")
    print("=" * 60)

    G = generate_test_graph(n=100, m=3)
    n_init = G.number_of_nodes()
    attack_seq = generate_attack_sequence(G, mode='degree')

    result = compute('attack_robustness', G, attack_sequence=attack_seq, stop_threshold=0.1)

    traj = result['trajectory']
    auc_val = result['auc']
    final_step = result['final_step']

    print(f"  初始节点数      : {result['initial_nodes']}")
    print(f"  轨迹记录数      : {len(traj)}")
    print(f"  AUC 值          : {auc_val:.6f}")
    print(f"  最后一步        : {final_step}")
    print(f"  轨迹首条        : {traj[0]}")
    print(f"  轨迹末条        : {traj[-1]}")

    assert len(traj) >= 2, "轨迹应至少包含初始状态和一次攻击"
    assert 0.0 < auc_val < 1.0, "AUC 应在 (0, 1) 之间"
    assert final_step == traj[-1][0], "final_step 应与轨迹最后一步一致"

    # 验证与分别计算的结果一致
    traj2 = compute('attack_trajectory', G, attack_sequence=attack_seq, stop_threshold=0.1)
    auc2 = compute('auc', trajectory=traj2, initial_nodes=n_init)
    assert traj == traj2, "attack_robustness 的轨迹应与单独计算一致"
    assert abs(auc_val - auc2) < 1e-10, "attack_robustness 的 AUC 应与单独计算一致"

    print("  ✓ Attack Robustness 测试通过")


def test_functional_robustness():
    """测试功能指标组合接口（CSA/CCE/WCP 轨迹 + AUC）。"""
    print("\n" + "=" * 60)
    print("[Test 8] Functional Robustness (CSA / CCE / WCP)")
    print("=" * 60)

    G = generate_test_graph(n=100, m=3)
    n_init = G.number_of_nodes()
    attack_seq = generate_attack_sequence(G, mode='degree')
    centers = list(G.nodes())[:5]  # 选择前5个节点作为控制器

    # CSA Robustness
    csa_result = compute('csa_robustness', G, attack_sequence=attack_seq, centers=centers, stop_threshold=0.1)
    print(f"  CSA Robustness:")
    print(f"    轨迹记录数    : {len(csa_result['trajectory'])}")
    print(f"    CSA_AUC       : {csa_result['auc']:.6f}")
    print(f"    最后一步      : {csa_result['final_step']}")
    print(f"    首条 CSA      : {csa_result['trajectory'][0][2]:.4f}")
    print(f"    末条 CSA      : {csa_result['trajectory'][-1][2]:.4f}")
    assert len(csa_result['trajectory']) >= 2, "CSA 轨迹应至少包含初始状态"
    assert csa_result['auc'] >= 0.0, "CSA_AUC 应 >= 0"

    # CCE Robustness
    cce_result = compute('cce_robustness', G, attack_sequence=attack_seq, centers=centers, stop_threshold=0.1)
    print(f"\n  CCE Robustness:")
    print(f"    轨迹记录数    : {len(cce_result['trajectory'])}")
    print(f"    CCE_AUC       : {cce_result['auc']:.6f}")
    print(f"    最后一步      : {cce_result['final_step']}")
    print(f"    首条 CCE      : {cce_result['trajectory'][0][2]:.4f}")
    print(f"    末条 CCE      : {cce_result['trajectory'][-1][2]:.4f}")
    assert len(cce_result['trajectory']) >= 2, "CCE 轨迹应至少包含初始状态"
    assert cce_result['auc'] >= 0.0, "CCE_AUC 应 >= 0"

    # WCP Robustness
    wcp_result = compute('wcp_robustness', G, attack_sequence=attack_seq, centers=centers, stop_threshold=0.1)
    print(f"\n  WCP Robustness:")
    print(f"    轨迹记录数    : {len(wcp_result['trajectory'])}")
    print(f"    WCP_AUC       : {wcp_result['auc']:.6f}")
    print(f"    最后一步      : {wcp_result['final_step']}")
    print(f"    首条 WCP      : {wcp_result['trajectory'][0][2]:.4f}")
    print(f"    末条 WCP      : {wcp_result['trajectory'][-1][2]:.4f}")
    assert len(wcp_result['trajectory']) >= 2, "WCP 轨迹应至少包含初始状态"
    assert wcp_result['auc'] >= 0.0, "WCP_AUC 应 >= 0"

    # 验证与单独调用的一致性
    single_csa = compute('csa', G, centers=centers)
    assert abs(csa_result['trajectory'][0][2] - single_csa) < 1e-6, "组合接口初始 CSA 应与单次计算一致"

    print("\n  ✓ Functional Robustness (CSA/CCE/WCP) 测试通过")


def main():
    print("=" * 60)
    print("network_metrics 结构指标自测试")
    print("=" * 60)
    print(f"\n已注册指标: {list_metrics()}")

    test_gcc()
    test_attack_trajectory()
    test_auc()
    test_r_value()
    test_gcc_after_removal()
    test_gcc_snapshot()
    test_attack_robustness()
    test_functional_robustness()

    print("\n" + "=" * 60)
    print("全部 structural + functional 指标测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    main()
