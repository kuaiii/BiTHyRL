# -*- coding: utf-8 -*-
"""
依赖控制器部署的网络功能指标：覆盖率、效率、最大受控连通分量
"""
import networkx as nx


def get_shortest_dist(G, source, dest):
    """计算带权最短路径距离，失败则返回 inf。"""
    try:
        try:
            dist = nx.shortest_path_length(G, source, dest, weight='weight')
        except TypeError:
            dist = nx.shortest_path_length(G, source, dest)
        return dist
    except nx.NetworkXNoPath:
        return float('inf')


def coverage_compute1(G, centers, assignments):
    """
    覆盖率计算 v1：基于 assignments 映射判断每个节点是否被覆盖。
    """
    count = 0
    center_num = len(centers)
    for node in G.nodes():
        if assignments[node] < center_num:
            if nx.has_path(G, node, centers[assignments[node]]):
                count += 1
    return count / G.number_of_nodes()


def coverage_compute2(G, centers):
    """
    覆盖率计算 v2（推荐）：基于连通分量。
    只要连通分量中包含至少一个控制器，该分量内所有节点都被视为已覆盖。
    返回被覆盖的绝对节点数（便于主程序统一除以初始节点数进行归一化）。
    """
    current_total = G.number_of_nodes()
    if current_total == 0:
        return 0
    communities = list(nx.connected_components(G))
    centers_set = set(centers)
    count = 0
    for community in communities:
        if not centers_set.isdisjoint(community):
            count += len(community)
    return count


def weight_efficiency_compute(G, centers):
    """
    网络加权效率：节点-控制器通信效率 + 控制器-控制器同步效率。
    返回累加值，未归一化。
    """
    efficiency = 0.0
    for node in G.nodes():
        if node not in centers:
            min_dis = float('inf')
            for center in centers:
                try:
                    if nx.has_path(G, node, center):
                        temp_dis = get_shortest_dist(G, node, center)
                        if temp_dis < min_dis:
                            min_dis = temp_dis
                except Exception:
                    pass
            if min_dis != 0 and min_dis != float('inf'):
                efficiency += 1 / min_dis

    center_num = len(centers)
    temp = 0.0
    for i in range(center_num):
        source = centers[i]
        for j in range(i + 1, center_num):
            dest = centers[j]
            if nx.has_path(G, source, dest):
                d = get_shortest_dist(G, source, dest)
                if d != 0:
                    temp += 1 / d
    return efficiency + temp


def max_component_size_compute(G, centers):
    """
    计算包含控制器的最大连通分量大小。
    """
    if G.number_of_nodes() == 0:
        return 0
    communities = list(sorted(nx.connected_components(G), key=len, reverse=True))
    centers_set = set(centers)
    for community in communities:
        if not community.isdisjoint(centers_set):
            return len(community)
    return 0


def calculate_efficiency(G, centers):
    """统一接口：加权效率。"""
    return weight_efficiency_compute(G, centers)


def calculate_coverage(G, centers):
    """统一接口：覆盖率（基于连通分量）。"""
    return coverage_compute2(G, centers)
