import numpy as np
import networkx as nx
import random
import math
import copy
import matplotlib.pyplot as plt
from collections import defaultdict

# --- 全局参数配置 (基于论文 Table II  复现) ---
CONF = {
    # 场景参数
    'AREA_WIDTH': 800,  # 监测区域 800m x 800m
    'NUM_NODES': 100,  # 节点总数 (论文为1000，为演示代码运行效率，此处设为100，可调整)
    'RATIO_SUPER_NODES': 0.1,  # 超级节点占比 10%
    'NUM_SINKS': 4,  # Sink节点数量 (多Sink场景)

    # 通信参数
    'R_COMM_ORDINARY': 50,  # 普通节点通信半径 (m)
    'R_COMM_SUPER': 150,  # 超级节点通信半径 (m)

    # 能量模型参数 (First Order Radio Model)
    'E_ELEC': 50e-9,  # 电路损耗 (J/bit)
    'E_FS': 10e-12,  # 自由空间放大系数 (J/bit/m^2)
    'E_MP': 0.0013e-12,  # 多径衰落放大系数 (J/bit/m^4)
    'D0': 87.7,  # 阈值距离 sqrt(E_fs/E_mp)
    'PACKET_SIZE': 1000,  # 数据包大小 (bits)
    'INIT_ENERGY_ORD': 0.5,  # 普通节点初始能量 (J)
    'INIT_ENERGY_SUPER': 1.5,  # 超级节点初始能量 (J)

    # 遗传算法参数 (TEAM算法特有)
    'MP': 4,  # 种群数量 (Multi-Populations)
    'POP_SIZE': 20,  # 每个种群的个体数
    'MAX_GEN': 50,  # 最大迭代次数
    'P_CROSSOVER': 0.7,  # 交叉概率
    'P_MUTATION': 0.03,  # 变异概率
    'TOURNAMENT_SIZE': 2  # 锦标赛选择规模
}


class Node:
    """ 节点类：定义节点的基本属性 """

    def __init__(self, node_id, x, y, node_type):
        self.id = node_id
        self.x = x
        self.y = y
        self.type = node_type  # 0: 普通节点, 1: 超级节点, 2: Sink节点
        # 根据节点类型分配初始能量
        if node_type == 2:
            self.initial_energy = float('inf')  # Sink能量无限
        elif node_type == 1:
            self.initial_energy = CONF['INIT_ENERGY_SUPER']
        else:
            self.initial_energy = CONF['INIT_ENERGY_ORD']
        self.current_energy = self.initial_energy


class IoTNetwork:
    """ 物联网环境类：管理拓扑结构 """

    def __init__(self, G=None):
        """
        初始化网络
        :param G: 可选的networkx图，如果提供则基于此图构建，否则自动生成
        """
        self.nodes = []
        self.graph = nx.Graph()  # 基础物理连接图
        self.super_nodes = []
        self.sink_nodes = []
        self.ordinary_nodes = []
        if G is not None:
            self._initialize_from_graph(G)
        else:
            self._initialize_nodes()

    def _initialize_nodes(self):
        """ 初始化节点位置与类型 """
        # 1. 部署 Sink 节点 (固定在四角，模拟多Sink覆盖)
        sink_positions = [(0, 0), (CONF['AREA_WIDTH'], 0), (0, CONF['AREA_WIDTH']), (CONF['AREA_WIDTH'], CONF['AREA_WIDTH'])]
        for i in range(CONF['NUM_SINKS']):
            pos = sink_positions[i % 4]
            n = Node(i, pos[0], pos[1], 2)
            self.nodes.append(n)
            self.sink_nodes.append(n)

        # 2. 随机部署 超级节点 和 普通节点
        num_others = CONF['NUM_NODES'] - CONF['NUM_SINKS']
        num_super = int(num_others * CONF['RATIO_SUPER_NODES'])

        for i in range(num_others):
            nid = len(self.nodes)
            x = random.uniform(0, CONF['AREA_WIDTH'])
            y = random.uniform(0, CONF['AREA_WIDTH'])
            # 简单起见，前 num_super 个随机节点设为超级节点
            # 实际论文中可能使用聚类中心确定超级节点位置，此处简化为随机分布
            ntype = 1 if i < num_super else 0
            n = Node(nid, x, y, ntype)
            self.nodes.append(n)
            if ntype == 1:
                self.super_nodes.append(n)
            else:
                self.ordinary_nodes.append(n)

        # 3. 构建基础物理连接图 (基于通信半径)
        self.build_distance_graph()

    def _initialize_from_graph(self, G):
        """
        从外部networkx图G初始化网络
        假设G的节点有'pos'属性，节点ID为0到n-1
        注意：会根据通信半径重新构建基础图，而不是直接使用原图的所有边
        """
        # 确保节点ID是连续的0到n-1
        n = G.number_of_nodes()
        if sorted(G.nodes()) != list(range(n)):
            # 重新映射节点ID
            mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(G.nodes()))}
            G = nx.relabel_nodes(G, mapping)
        
        # 确定节点类型：假设前NUM_SINKS个为Sink，然后按比例分配超级节点
        num_sinks = min(CONF['NUM_SINKS'], n)
        num_others = n - num_sinks
        num_super = int(num_others * CONF['RATIO_SUPER_NODES'])
        
        # 创建节点对象
        # 首先检查位置是否已经归一化（在[0,1]范围内）
        sample_pos = None
        for i in range(min(5, n)):
            if 'pos' in G.nodes[i]:
                sample_pos = G.nodes[i]['pos']
                break
        
        is_normalized = False
        if sample_pos is not None:
            # 检查位置是否在[0,1]范围内（允许小的误差）
            if isinstance(sample_pos, (tuple, list)) and len(sample_pos) >= 2:
                if 0 <= sample_pos[0] <= 1.1 and 0 <= sample_pos[1] <= 1.1:
                    is_normalized = True
        
        for i in range(n):
            if 'pos' in G.nodes[i]:
                pos = G.nodes[i]['pos']
                if is_normalized:
                    # 归一化位置转换为实际坐标（缩放到合适的范围，确保节点在通信范围内）
                    # 使用较小的缩放因子，确保节点间距离在通信半径内
                    scale_factor = min(CONF['AREA_WIDTH'], CONF['R_COMM_SUPER'] * 3)  # 确保有足够的连接
                    x, y = pos[0] * scale_factor, pos[1] * scale_factor
                else:
                    # 位置已经是实际坐标，直接使用
                    x, y = pos[0], pos[1]
            else:
                # 如果没有pos属性，使用随机位置
                x = random.uniform(0, CONF['AREA_WIDTH'])
                y = random.uniform(0, CONF['AREA_WIDTH'])
            
            # 分配节点类型
            if i < num_sinks:
                ntype = 2  # Sink节点
            elif i < num_sinks + num_super:
                ntype = 1  # 超级节点
            else:
                ntype = 0  # 普通节点
            
            node = Node(i, x, y, ntype)
            self.nodes.append(node)
            
            if ntype == 2:
                self.sink_nodes.append(node)
            elif ntype == 1:
                self.super_nodes.append(node)
            else:
                self.ordinary_nodes.append(node)
        
        # 根据通信半径重新构建基础图（而不是直接使用原图）
        # 这样可以确保图结构符合TEAM算法的通信模型
        self.build_distance_graph()

    def _update_edge_weights(self):
        """更新图中所有边的权重为实际距离"""
        positions = {n.id: (n.x, n.y) for n in self.nodes}
        for u, v in list(self.graph.edges()):
            if u in positions and v in positions:
                dist = np.linalg.norm(np.array(positions[u]) - np.array(positions[v]))
                self.graph[u][v]['weight'] = dist

    def build_distance_graph(self):
        """
        根据欧几里得距离和通信半径构建潜在链路图。
        这是拓扑进化的基础空间。
        """
        self.graph.clear()
        self.graph.add_nodes_from([n.id for n in self.nodes])

        positions = np.array([[n.x, n.y] for n in self.nodes])

        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                dist = np.linalg.norm(positions[i] - positions[j])
                node_i = self.nodes[i]
                node_j = self.nodes[j]

                # 确定通信半径：只要一方是超级节点或Sink，就按大半径计算（假设双向通信能力匹配）
                # 或者严格按照节点类型：普通节点只能听懂 50m 内的信号
                # 这里采用非对称链路的简化处理：取两者的最大通信半径作为判定标准
                r_i = CONF['R_COMM_SUPER'] if node_i.type in [1, 2] else CONF['R_COMM_ORDINARY']
                r_j = CONF['R_COMM_SUPER'] if node_j.type in [1, 2] else CONF['R_COMM_ORDINARY']

                # 如果距离小于最大通信半径，建立潜在连接
                if dist <= max(r_i, r_j):
                    self.graph.add_edge(i, j, weight=dist)

    def get_super_node_matrix_size(self):
        """ 计算超级节点间连接矩阵的大小 (用于染色体长度) """
        n = len(self.super_nodes)
        return (n * (n - 1)) // 2  # 上三角矩阵元素个数

    class Evaluator:
        """ 评估器：计算 FNDT, Robustness, Entropy """

        @staticmethod
        def calculate_energy_cost(dist):
            """
            实现一阶无线电能耗模型公式
            return: 传输 k bits 的能耗 (J)
            """
            k = CONF['PACKET_SIZE']
            if dist <= CONF['D0']:
                return k * CONF['E_ELEC'] + k * CONF['E_FS'] * (dist ** 2)
            else:
                return k * CONF['E_ELEC'] + k * CONF['E_MP'] * (dist ** 4)

        @staticmethod
        def evaluate(chromosome, network, sample_ratio=0.3):
            """
            核心评估函数（优化版本：使用采样加速）
            :param chromosome: 二进制列表，代表超级节点间的连接开关
            :param network: 基础网络对象
            :param sample_ratio: 采样比例（用于鲁棒性计算加速）
            :return: [entropy, fndt_score, r_value] (均为越大越好)
            """
            # 1. 解码：基于染色体构建当前演化的拓扑图
            # 复制基础图，在此基础上根据染色体断开某些超级节点间的连接
            temp_graph = network.graph.copy()
            sn_ids = [n.id for n in network.super_nodes]

            # 染色体映射到超级节点对 (上三角矩阵遍历)
            gene_idx = 0
            for i in range(len(sn_ids)):
                for j in range(i + 1, len(sn_ids)):
                    u, v = sn_ids[i], sn_ids[j]

                    # 如果基础图中存在这条边，且染色体对应位为 0，则断开连接
                    if temp_graph.has_edge(u, v):
                        if gene_idx < len(chromosome):
                            if chromosome[gene_idx] == 0:
                                temp_graph.remove_edge(u, v)
                        gene_idx += 1

            # --- 目标 1: 负载均衡 (Information Entropy)  ---
            # 优化版本：使用批量最短路径计算和采样加速
            sink_loads = defaultdict(int)
            sink_ids = [n.id for n in network.sink_nodes]
            ordinary_ids = [n.id for n in network.ordinary_nodes]

            # 优化：使用简化的负载分配策略（快速近似）
            # 对于大图，使用基于距离的简单分配，避免昂贵的路径计算
            total_packets = 0
            if len(ordinary_ids) > 50:
                # 大图：使用简化的距离分配（只计算欧氏距离，不计算图路径）
                # 从图中获取节点位置
                node_positions = {}
                for nid in temp_graph.nodes():
                    if 'pos' in temp_graph.nodes[nid]:
                        node_positions[nid] = temp_graph.nodes[nid]['pos']
                    elif hasattr(network, 'graph') and nid in network.graph.nodes():
                        if 'pos' in network.graph.nodes[nid]:
                            node_positions[nid] = network.graph.nodes[nid]['pos']
                
                for oid in ordinary_ids:
                    try:
                        if oid not in node_positions:
                            continue
                        o_pos = node_positions[oid]
                        min_dist = float('inf')
                        chosen_sink = None
                        for sid in sink_ids:
                            if sid not in node_positions:
                                continue
                            s_pos = node_positions[sid]
                            # 计算欧氏距离（快速近似）
                            dist = math.sqrt((o_pos[0] - s_pos[0])**2 + (o_pos[1] - s_pos[1])**2)
                            if dist < min_dist:
                                min_dist = dist
                                chosen_sink = sid
                        if chosen_sink:
                            sink_loads[chosen_sink] += 1
                            total_packets += 1
                    except:
                        pass
            else:
                # 小图：使用批量BFS计算最短路径
                try:
                    sink_distances = {}
                    for sid in sink_ids:
                        try:
                            distances = nx.single_source_dijkstra_path_length(
                                temp_graph, sid, weight='weight', cutoff=1000)  # 设置cutoff加速
                            sink_distances[sid] = distances
                        except:
                            sink_distances[sid] = {}
                    
                    for oid in ordinary_ids:
                        min_dist = float('inf')
                        chosen_sink = None
                        for sid in sink_ids:
                            if sid in sink_distances and oid in sink_distances[sid]:
                                dist = sink_distances[sid][oid]
                                if dist < min_dist:
                                    min_dist = dist
                                    chosen_sink = sid
                        if chosen_sink:
                            sink_loads[chosen_sink] += 1
                            total_packets += 1
                except Exception:
                    # 如果失败，使用简单分配
                    for oid in ordinary_ids[:min(20, len(ordinary_ids))]:
                        sink_loads[sink_ids[0] if sink_ids else None] = len(ordinary_ids)
                        total_packets = len(ordinary_ids)

            # 计算熵
            entropy = 0
            if total_packets > 0:
                for sid in sink_ids:
                    if sink_loads[sid] > 0:
                        p = sink_loads[sid] / total_packets
                        entropy += -1 * p * math.log(p)

            # --- 目标 2: 能量效率 (FNDT) ---
            # 优化版本：使用度中心性替代介数中心性（快得多）
            # 或者使用采样介数中心性
            try:
                # 方法1：使用度中心性（快速近似）
                if len(temp_graph) > 100:
                    # 对于大图，使用度中心性作为近似
                    degrees = dict(temp_graph.degree())
                    max_degree = 0
                    for sn_id in sn_ids:
                        if sn_id in degrees:
                            max_degree = max(max_degree, degrees[sn_id])
                    # 归一化度中心性
                    max_load = max_degree / len(temp_graph) if len(temp_graph) > 0 else 0
                else:
                    # 对于小图，可以使用采样介数中心性
                    # 只计算部分节点的介数中心性
                    sample_nodes = min(30, len(temp_graph))
                    sampled_nodes = random.sample(list(temp_graph.nodes()), sample_nodes)
                    bet_cent = nx.betweenness_centrality(temp_graph, weight='weight', k=sample_nodes)
                    max_load = 0
                    for sn_id in sn_ids:
                        if sn_id in bet_cent:
                            max_load = max(max_load, bet_cent[sn_id])
                    # 如果没有找到，使用度中心性作为后备
                    if max_load == 0:
                        degrees = dict(temp_graph.degree())
                        for sn_id in sn_ids:
                            if sn_id in degrees:
                                max_load = max(max_load, degrees[sn_id] / len(temp_graph))

                # 加上一个小量防止除零
                fndt_score = 1.0 / (max_load + 1e-9)
            except:
                fndt_score = 0

            # --- 目标 3: 鲁棒性 (R-Value)  ---
            # 优化版本：使用采样方法加速计算
            attack_graph = temp_graph.copy()
            n_nodes = len(attack_graph)
            # 使用采样：只移除部分节点
            num_to_remove = max(1, int(n_nodes * sample_ratio * 0.1))
            if num_to_remove < n_nodes:
                nodes_to_remove = random.sample(list(attack_graph.nodes()), num_to_remove)
                attack_graph.remove_nodes_from(nodes_to_remove)

            if len(attack_graph) > 0:
                try:
                    # 获取所有连通分量
                    components = list(nx.connected_components(attack_graph))
                    if components:
                        largest_cc_size = len(max(components, key=len))
                        r_value = largest_cc_size / n_nodes  # 使用原始节点数归一化
                    else:
                        r_value = 0
                except:
                    r_value = 0
            else:
                r_value = 0

            return [entropy, fndt_score, r_value]


class TEAM_Engine:
    """ TEAM 算法核心引擎 """

    def __init__(self, network):
        self.network = network
        self.chromo_len = network.get_super_node_matrix_size()
        self.populations = []  # 存储多个种群
        if self.chromo_len == 0:
            raise ValueError("没有超级节点，无法进行拓扑演化。请确保网络中有超级节点。")

    def init_populations(self):
        """ 初始化 MP 个种群 """
        for _ in range(CONF['MP']):
            pop = []
            for _ in range(CONF['POP_SIZE']):
                # 随机生成二进制染色体
                genome = [random.randint(0, 1) for _ in range(self.chromo_len)]
                # 个体结构：基因，目标函数值，Rank，拥挤距离
                ind = {'genome': genome, 'objs': None, 'rank': 0, 'dist': 0}
                pop.append(ind)
            self.populations.append(pop)

    def fast_non_dominated_sort(self, population):
        """
        NSGA-II 快速非支配排序
        将种群中的个体划分到不同的 Pareto Fronts
        """
        if not population:
            return
        fronts = [[]]
        for p in population:
            p['n'] = 0  # 支配 p 的个体数量
            p['S'] = []  # p 支配的个体集合

            for q in population:
                if p == q:
                    continue
                # 检查 p 是否支配 q (所有目标最大化)
                # objs = [entropy, fndt, r_value]
                dominate = False
                dominated = False
                for k in range(3):
                    if p['objs'][k] > q['objs'][k]:
                        dominate = True
                    elif p['objs'][k] < q['objs'][k]:
                        dominated = True

                if dominate and not dominated:  # p 支配 q
                    p['S'].append(q)
                elif dominated and not dominate:  # q 支配 p
                    p['n'] += 1

            if p['n'] == 0:
                p['rank'] = 1
                fronts[0].append(p)

        # 生成后续 Fronts
        i = 0
        while i < len(fronts) and len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p['S']:
                    q['n'] -= 1
                    if q['n'] == 0:
                        q['rank'] = i + 2
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        # 计算拥挤距离 (Same-Layer Election 的基础)
        for front in fronts:
            if not front: continue
            # 初始化距离
            for p in front: p['dist'] = 0

            # 对每个目标维度计算距离
            for m in range(3):
                # 根据该目标值排序
                front.sort(key=lambda x: x['objs'][m])

                # 边界个体距离设为无穷大
                if len(front) > 0:
                    front[0]['dist'] = float('inf')
                    if len(front) > 1:
                        front[-1]['dist'] = float('inf')

                    obj_min = front[0]['objs'][m]
                    obj_max = front[-1]['objs'][m]

                    if obj_max - obj_min == 0: 
                        continue

                    # 中间个体累加归一化距离
                    for k in range(1, len(front) - 1):
                        front[k]['dist'] += (front[k + 1]['objs'][m] - front[k - 1]['objs'][m]) / (obj_max - obj_min)

    def layered_cooperation(self):
        """
        [复现核心] 分层协作机制
        在相邻种群间交换精英与劣汰个体
        """
        num_pops = len(self.populations)
        for i in range(num_pops):
            target_idx = (i + 1) % num_pops

            # 当前种群：按 Rank 升序 (1是最好), Dist 降序 (越大越好)
            # 取出最好的个体 (Elite)
            pop_curr = sorted(self.populations[i], key=lambda x: (x['rank'], -x['dist']))
            if not pop_curr:
                continue
            elite = copy.deepcopy(pop_curr[0])

            # 目标种群：同样排序
            # 取出最差的个体 (Worst)，通常是 Rank 最大且 Dist 最小的
            pop_next = sorted(self.populations[target_idx], key=lambda x: (x['rank'], -x['dist']))
            if not pop_next:
                continue

            # 替换操作：将 Elite 的基因赋予 Target 种群的最差个体
            # 注意：实际操作中可能需要保留 elite 的完整属性，这里仅做基因替换
            # 下一轮评估时会重新计算目标值
            worst_idx = self.populations[target_idx].index(pop_next[-1])
            self.populations[target_idx][worst_idx]['genome'] = elite['genome']
            self.populations[target_idx][worst_idx]['objs'] = None  # 标记为脏数据，需重算

    def run(self, verbose=True):
        """ 主循环（优化版本） """
        import time
        start_time = time.time()
        
        if verbose:
            print(f"  [TEAM] 初始化种群...")
        
        self.init_populations()
        history_best = []
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"  [TEAM] 初始化完成 (耗时: {elapsed:.2f}s)")
            print(f"  [TEAM] 开始演化 ({CONF['MAX_GEN']} 代)...")

        for gen in range(CONF['MAX_GEN']):
            gen_start = time.time()
            
            # 显示当前代进度
            if verbose and gen % 2 == 0:  # 每2代显示一次
                elapsed = time.time() - start_time
                print(f"  [TEAM] 正在评估 Gen {gen + 1}/{CONF['MAX_GEN']}... (已耗时: {elapsed:.2f}s)", end="\r", flush=True)

            # 1. 评估所有个体 (如果未评估，使用采样加速)
            total_to_eval = sum(1 for pop in self.populations for ind in pop if ind['objs'] is None)
            evaluated = 0
            for pop in self.populations:
                for ind in pop:
                    if ind['objs'] is None:
                        ind['objs'] = IoTNetwork.Evaluator.evaluate(ind['genome'], self.network, sample_ratio=0.3)
                        evaluated += 1
                        # 每评估10个个体显示一次进度
                        if verbose and total_to_eval > 10 and evaluated % 10 == 0:
                            print(f"  [TEAM] 评估进度: {evaluated}/{total_to_eval} (Gen {gen + 1})", end="\r", flush=True)

            # 2. 对每个种群执行 NSGA-II 排序
            for pop in self.populations:
                self.fast_non_dominated_sort(pop)

            # 3. 执行分层协作 (交换基因)
            self.layered_cooperation()

            # 4. 生成下一代 (选择、交叉、变异)
            for i in range(CONF['MP']):
                pop = self.populations[i]
                new_pop = []

                # 精英保留策略：直接保留 Front 1 的一部分
                # 这里简化为锦标赛选择生成全量新后代
                while len(new_pop) < CONF['POP_SIZE']:
                    # 锦标赛选择 (Tournament Selection)
                    parents = []
                    for _ in range(2):
                        candidates = random.sample(pop, min(CONF['TOURNAMENT_SIZE'], len(pop)))
                        # 胜出逻辑：Rank 小者胜；Rank 相同则 Dist 大者胜
                        winner = min(candidates, key=lambda x: (x['rank'], -x['dist']))
                        parents.append(winner)

                    p1, p2 = parents[0]['genome'], parents[1]['genome']

                    # 交叉 (Crossover)
                    if random.random() < CONF['P_CROSSOVER'] and self.chromo_len > 1:
                        cp = random.randint(1, self.chromo_len - 1)
                        c1 = p1[:cp] + p2[cp:]
                        c2 = p2[:cp] + p1[cp:]
                    else:
                        c1, c2 = p1[:], p2[:]

                    # 变异 (Mutation)
                    for child_g in [c1, c2]:
                        for g_idx in range(len(child_g)):
                            if random.random() < CONF['P_MUTATION']:
                                child_g[g_idx] = 1 - child_g[g_idx]

                        new_pop.append({'genome': child_g, 'objs': None, 'rank': 0, 'dist': 0})

                # 截断保留 POP_SIZE 个体
                self.populations[i] = new_pop[:CONF['POP_SIZE']]

            # 记录本代全局最优 (基于 FNDT 简单记录用于展示)
            # 实际应记录 Pareto Front
            all_inds = [ind for pop in self.populations for ind in pop if ind['objs']]
            if all_inds:
                best_fndt = max(all_inds, key=lambda x: x['objs'][1])  # 基于FNDT选择
                history_best.append(best_fndt['objs'])
                self.best_individual = best_fndt  # 保存最佳个体
                if verbose and (gen == 0 or gen % 2 == 0 or gen == CONF['MAX_GEN'] - 1):
                    elapsed = time.time() - start_time
                    gen_elapsed = time.time() - gen_start
                    print(f"\n  [TEAM] Gen {gen + 1}/{CONF['MAX_GEN']}: Entropy={best_fndt['objs'][0]:.4f}, "
                          f"FNDT={best_fndt['objs'][1]:.4f}, R={best_fndt['objs'][2]:.4f} "
                          f"(本代耗时: {gen_elapsed:.2f}s, 总耗时: {elapsed:.2f}s)", flush=True)

        if verbose:
            total_time = time.time() - start_time
            if self.best_individual and self.best_individual.get('objs'):
                best_objs = self.best_individual['objs']
                print(f"  [TEAM] 优化完成 | Entropy: {best_objs[0]:.4f}, FNDT: {best_objs[1]:.4f}, "
                      f"R: {best_objs[2]:.4f} | 总耗时: {total_time:.2f}s")

        return history_best

    def get_reconstructed_graph(self):
        """
        获取重构后的图
        :return: networkx图对象
        """
        if not hasattr(self, 'best_individual') or self.best_individual is None:
            # 如果没有最佳个体，从当前种群中选择
            all_inds = [ind for pop in self.populations for ind in pop if ind['objs']]
            if not all_inds:
                return self.network.graph.copy()
            self.best_individual = max(all_inds, key=lambda x: x['objs'][1])
        
        # 基于最佳个体的染色体重构图
        temp_graph = self.network.graph.copy()
        sn_ids = [n.id for n in self.network.super_nodes]
        chromosome = self.best_individual['genome']
        
        # 染色体映射到超级节点对 (上三角矩阵遍历)
        gene_idx = 0
        for i in range(len(sn_ids)):
            for j in range(i + 1, len(sn_ids)):
                u, v = sn_ids[i], sn_ids[j]
                if temp_graph.has_edge(u, v):
                    if gene_idx < len(chromosome):
                        if chromosome[gene_idx] == 0:
                            temp_graph.remove_edge(u, v)
                    gene_idx += 1
        
        return temp_graph


if __name__ == "__main__":
    # 1. 创建仿真环境
    print("Initializing IoT Network...")
    iot_net = IoTNetwork()
    print(f"Network Created: {len(iot_net.nodes)} nodes.")
    print(f"Super Nodes: {len(iot_net.super_nodes)}, Sinks: {len(iot_net.sink_nodes)}")

    # 2. 初始化算法引擎
    team_solver = TEAM_Engine(iot_net)

    # 3. 运行演化
    print("Starting TEAM Evolution...")
    results = team_solver.run()

    # 4. 结果可视化 (绘制收敛曲线)
    entropy_hist = [x[0] for x in results]
    fndt_hist = [x[1] for x in results]
    r_hist = [x[2] for x in results]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(entropy_hist, 'r')
    plt.title("Load Balancing (Entropy)")

    plt.subplot(1, 3, 2)
    plt.plot(fndt_hist, 'g')
    plt.title("Energy Efficiency (FNDT Score)")

    plt.subplot(1, 3, 3)
    plt.plot(r_hist, 'b')
    plt.title("Robustness (R-Value)")

    plt.tight_layout()
    plt.show()
    print("Simulation Complete.")