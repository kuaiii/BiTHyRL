import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import rcParams

# ==========================================
# 1. 全局配置与样式 (您可以微调此处)
# ==========================================
# 颜色定义（按你给的 RGB 风格）
# 169, 233, 228 -> #A9E9E4
# 239, 148, 158 -> #EF949E
COLOR_RANDOM = '#A9E9E4'
COLOR_TARGETED = '#EF949E'

# 字体与绘图风格设置
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'], # 强制使用 Times New Roman
    "font.weight": 'bold',             # 全局加粗
    "axes.labelweight": 'bold',
    "axes.titleweight": 'bold',
    
    # --- 您可以在这里调整基础线条粗细 ---
    "axes.linewidth": 3,       # 坐标轴边框粗细
    "xtick.major.width": 3,    # x轴刻度线粗细
    "ytick.major.width": 3,    # y轴刻度线粗细
    "lines.linewidth": 4,      # 绘图曲线粗细
    
    "mathtext.fontset": 'stix', # 数学公式字体兼容
}
rcParams.update(config)

# 网络基础参数
N_NODES = 1000
M_EDGES = 4000
SEED_VAL = 42

np.random.seed(SEED_VAL)
random.seed(SEED_VAL)

# ==========================================
# 2. 网络生成器 (严格控制 N 和 M)
# ==========================================

def adjust_edges_strictly(G, target_m):
    """辅助函数：强制增删边以精确匹配目标边数"""
    current_m = G.number_of_edges()
    if current_m < target_m:
        nodes = list(G.nodes())
        while G.number_of_edges() < target_m:
            u, v = random.sample(nodes, 2)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
    elif current_m > target_m:
        edges = list(G.edges())
        random.shuffle(edges)
        G.remove_edges_from(edges[:current_m - target_m])
    return G

def create_random_strict(n, target_m):
    """Random (ER) Network"""
    G = nx.gnm_random_graph(n, target_m, seed=SEED_VAL)
    return adjust_edges_strictly(G, target_m)

def create_bimodal_theoretical(n, m, seed=42):
    """Bimodal Network (Based on k_max = A * N^(2/3))"""
    avg_k = 2 * m / n
    numerator = 2 * (avg_k**2) * ((avg_k - 1)**2)
    denominator = 2 * avg_k - 1
    A = (numerator / denominator) ** (1/3)
    k_max = int(round(A * (n ** (2/3))))
    
    n_high = 1
    n_low = n - n_high
    total_degree_needed = 2 * m
    degree_rem = total_degree_needed - k_max
    k_base = degree_rem // n_low
    remainder = degree_rem % n_low
    
    degree_seq = [k_max] * n_high + [k_base + 1] * remainder + [k_base] * (n_low - remainder)
    
    if nx.is_graphical(degree_seq):
        G = nx.havel_hakimi_graph(degree_seq)
        nx.double_edge_swap(G, nswap=5*m, max_tries=20*m, seed=seed)
    else:
        G = nx.configuration_model(degree_seq, seed=seed)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
    return adjust_edges_strictly(G, m)

def create_ba_strict(n, target_m):
    """Scale-free (BA) Network"""
    m_param = int(round(target_m / n))
    G = nx.barabasi_albert_graph(n, m_param, seed=SEED_VAL)
    return adjust_edges_strictly(G, target_m)

def create_ws_strict(n, target_m, p=0.05):
    """Small-world (WS) Network"""
    k_param = int(2 * target_m / n)
    G = nx.watts_strogatz_graph(n, k_param, p, seed=SEED_VAL)
    return adjust_edges_strictly(G, target_m)

class OnionOptimizerSimple:
    """Simple Onion Optimizer class"""
    def __init__(self, G):
        self.G = G
        self.nodes = list(G.nodes())
        self.N = len(self.nodes)
        self.edge_set = set(tuple(sorted((u, v))) for u, v in G.edges())
        deg = dict(G.degree())
        self.attack_order = sorted(self.nodes, key=lambda n: deg[n], reverse=True)
        self.K = max(3, int(self.N * 0.15)) 
    
    def get_R(self, edge_set):
        adj = {i: [] for i in range(self.N)}
        node_map = {n: i for i, n in enumerate(self.nodes)}
        edges_mapped = [(node_map[u], node_map[v]) for u, v in edge_set]
        attack_indices = [node_map[n] for n in self.attack_order[:self.K]]
        surviving = set(range(self.N)) - set(attack_indices)
        
        parent = list(range(self.N))
        size = [1] * self.N 
        active = [False] * self.N
        
        def find(i):
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for p in path: parent[p] = i
            return i
        
        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                if size[root_i] < size[root_j]: root_i, root_j = root_j, root_i
                parent[root_j] = root_i
                size[root_i] += size[root_j]
                return size[root_i]
            return size[root_i]
            
        adj_mapped = [[] for _ in range(self.N)]
        for u, v in edges_mapped:
            adj_mapped[u].append(v)
            adj_mapped[v].append(u)
            
        for i in surviving: active[i] = True
        
        current_max = 1 if surviving else 0
        
        for i in surviving:
            for neighbor in adj_mapped[i]:
                if active[neighbor] and neighbor < i:
                    union(i, neighbor)
        
        # Recalculate max after initial build
        max_s = 0
        for i in range(self.N):
            if active[i] and parent[i] == i:
                if size[i] > max_s: max_s = size[i]
        current_max = max_s

        lcc_history = [current_max]
        for i in reversed(attack_indices):
            active[i] = True
            s_i = 1
            for neighbor in adj_mapped[i]:
                if active[neighbor]:
                    s = union(i, neighbor)
                    s_i = max(s_i, s)
            current_max = max(current_max, s_i)
            lcc_history.append(current_max)
            
        return sum(lcc_history) / (self.N * (self.K + 1))

    def optimize(self, iterations=500):
        current_edges = list(self.edge_set)
        current_R = self.get_R(self.edge_set)
        for _ in range(iterations):
            idx1, idx2 = random.sample(range(len(current_edges)), 2)
            u, v = current_edges[idx1]
            x, y = current_edges[idx2]
            if len({u, v, x, y}) < 4: continue
            
            if random.random() < 0.5:
                e1, e2 = tuple(sorted((u, x))), tuple(sorted((v, y)))
            else:
                e1, e2 = tuple(sorted((u, y))), tuple(sorted((v, x)))
            if e1 in self.edge_set or e2 in self.edge_set: continue
            
            new_edge_set = self.edge_set.copy()
            new_edge_set.remove(current_edges[idx1])
            new_edge_set.remove(current_edges[idx2])
            new_edge_set.add(e1)
            new_edge_set.add(e2)
            new_R = self.get_R(new_edge_set)
            
            if new_R > current_R:
                self.edge_set = new_edge_set
                current_edges = list(self.edge_set)
                current_R = new_R
                
        G_opt = nx.Graph()
        G_opt.add_nodes_from(self.nodes)
        G_opt.add_edges_from(self.edge_set)
        return adjust_edges_strictly(G_opt, M_EDGES)

# ==========================================
# 3. 攻击模拟与辅助函数
# ==========================================

def simulate_attack(G, attack_type):
    G = G.copy()
    nodes = list(G.nodes())
    n_orig = len(nodes)
    
    if attack_type == 'targeted':
        deg = dict(G.degree())
        random.shuffle(nodes)
        nodes.sort(key=lambda x: deg[x], reverse=True)
    else:
        random.shuffle(nodes)
        
    lccs = [1.0]
    ratios = [0.0]
    
    for i in range(len(nodes)):
        G.remove_node(nodes[i])
        if G.number_of_nodes() > 0:
            lccs.append(len(max(nx.connected_components(G), key=len)) / n_orig)
        else:
            lccs.append(0)
        ratios.append((i+1)/n_orig)
    
    r_res = np.trapz(lccs, ratios)
    return ratios, lccs, r_res

def find_intersection_x(x_data, y_data, target_y=0.2):
    """计算 y=0.2 时的 x 坐标（线性插值）"""
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    idx = np.where(y_data < target_y)[0]
    if len(idx) == 0:
        return None
    
    first_idx = idx[0]
    if first_idx == 0:
        return x_data[0]
        
    x1, x2 = x_data[first_idx-1], x_data[first_idx]
    y1, y2 = y_data[first_idx-1], y_data[first_idx]
    
    x_target = x1 + (target_y - y1) * (x2 - x1) / (y2 - y1)
    return x_target


def _choose_annotation_offset_and_align(x_val, atk, xlim=(0.0, 1.0), ylim=(0.0, 1.05)):
    """
    为 y=0.2 的交点标注选择一个不压坐标轴的偏移与对齐方式。

    规则：
    - x 靠近左侧（y 轴）则向右偏；x 靠近右侧则向左偏
    - targeted 默认偏下，random 默认偏上；但在靠近底部时向上抬
    """
    x0, x1 = xlim
    # 归一化到 [0,1]
    denom = (x1 - x0) if (x1 - x0) != 0 else 1.0
    xn = (x_val - x0) / denom

    # --- 水平方向：避免压到 y 轴 / 右边框 ---
    # 交点靠左：文字放右边；靠右：文字放左边
    if xn <= 0.15:
        dx = 14
        ha = 'left'
    elif xn >= 0.85:
        dx = -14
        ha = 'right'
    else:
        # 中间区域：保留原先不同攻击类型的习惯方向
        dx = 11 if atk == 'random' else -11
        ha = 'left' if dx > 0 else 'right'

    # --- 垂直方向：避免压到 x 轴（底部） ---
    # 交点 y 固定在 0.2，但如果你后面改 target_y，这里仍然可复用
    y_val = 0.2
    y0, y1 = ylim
    yden = (y1 - y0) if (y1 - y0) != 0 else 1.0
    yn = (y_val - y0) / yden
    if yn <= 0.12:
        dy = 17  # 距离底部太近就强制上移（箭头长度减半）
        va = 'bottom'
    else:
        dy = 15 if atk == 'random' else -20
        va = 'bottom' if dy > 0 else 'top'

    # 连接弧度：与方向一致（更自然）
    if dx > 0:
        connectionstyle = "arc3,rad=.2"
    else:
        connectionstyle = "arc3,rad=-.2"

    return (dx, dy), ha, va, connectionstyle

# ==========================================
# 4. 主程序执行与绘图 (关键调整区域)
# ==========================================

if __name__ == "__main__":
    print("1. Generating Networks...")
    G_random = create_random_strict(N_NODES, M_EDGES)
    G_bimodal = create_bimodal_theoretical(N_NODES, M_EDGES)
    G_ba = create_ba_strict(N_NODES, M_EDGES)
    G_ws = create_ws_strict(N_NODES, M_EDGES)
    
    print("2. Optimizing Onion (this may take a few seconds)...")
    G_onion = OnionOptimizerSimple(G_ba).optimize(iterations=500)

    networks = {
        'Bimodal Network': G_bimodal,
        'BA Network': G_ba,
        'Random Network': G_random,
        'WS Network': G_ws,
        'Onion Network': G_onion
    }

    print("3. Simulating Attacks...")
    results = {}
    attack_types = ['random', 'targeted']
    for name in networks:
        results[name] = {}
        for atk in attack_types:
            results[name][atk] = simulate_attack(networks[name], atk)

    print("4. Plotting...")
    
    # --- 绘图循环 ---
    for name in networks:
        # 调整 figsize 可以改变字体的相对大小
        # (5.5, 4.5) 是当前字号下比较紧凑的比例
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        
        # 画水平虚线
        ax.axhline(y=0.2, color='black', linestyle='--', linewidth=2, alpha=0.5)
        
        for atk in attack_types:
            x, y, r = results[name][atk]
            c = COLOR_RANDOM if atk == 'random' else COLOR_TARGETED
            
            # 图例文本
            label = f"{'Random' if atk == 'random' else 'Targeted'} ($R_{{res}}={r:.4f}$)"
            
            # 绘制曲线
            ax.plot(x, y, label=label, color=c, alpha=1.0)
            ax.fill_between(x, y, color=c, alpha=0.1)
            
            # --- 箭头标注逻辑 ---
            x_int = find_intersection_x(x, y, 0.2)
            if x_int is not None:
                # 根据交点位置自动调整偏移，避免与坐标轴/边框重合
                xytext_offset, ha, va, connectionstyle = _choose_annotation_offset_and_align(
                    x_int,
                    atk,
                    xlim=ax.get_xlim(),
                    ylim=ax.get_ylim()
                )
                    
                ax.annotate(
                    f'{x_int:.2f}', 
                    xy=(x_int, 0.2), 
                    xytext=xytext_offset,
                    textcoords='offset points',
                    arrowprops=dict(
                        arrowstyle='->', 
                        color='black',
                        connectionstyle=connectionstyle,
                        linewidth=1.5
                    ),
                    bbox=dict(  # 给数字加底色，避免视觉上“压住”坐标轴线
                        boxstyle='round,pad=0.15',
                        facecolor='white',
                        edgecolor='none',
                        alpha=0.75
                    ),
                    fontsize=20,      # 【字体】箭头标注数字的大小
                    color='k',          # 字体颜色与曲线一致
                    fontweight='bold',
                    ha=ha, va=va
                )

        # 标题与标签设置
        ax.set_title(name, pad=20, fontsize=18) # 【字体】标题大小
        
        # 【字体】X轴和Y轴标签大小 (您要求大一倍，设为 40)
        ax.set_xlabel('The fraction of removed nodes : p', fontsize=20)   
        ax.set_ylabel('The Size of GCC : S /N', fontsize=20)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        
        # 去除网格
        ax.grid(False)
        
        # 【字体】图例大小 (您要求缩写1/3，设为 7)
        ax.legend(loc='best', fontsize=15) 

        # 保存图片 (无白边)
        filename = f"{name.replace(' ', '_')}_Final.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.0)
        print(f"Saved {filename}")
        
        # plt.show() # 如果在脚本中批量生成，可以注释掉此行