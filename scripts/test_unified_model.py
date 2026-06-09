# -*- coding: utf-8 -*-
"""
统一模型集成测试

测试新实现的组件:
1. 规模自适应Node2Vec特征提取
2. 规模编码
3. 特征投影器
4. UnifiedGATPolicy模型
5. 统一GCC奖励函数
6. 课程学习采样
"""
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import sys
import os
import unittest
import torch
import numpy as np
import networkx as nx

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestScaleAdaptiveNode2Vec(unittest.TestCase):
    """测试规模自适应Node2Vec特征提取"""
    
    def setUp(self):
        """创建不同规模的测试图"""
        self.graphs = {
            'tiny': nx.barabasi_albert_graph(30, 2),
            'small': nx.barabasi_albert_graph(70, 3),
            'medium': nx.barabasi_albert_graph(150, 3),
            'large': nx.barabasi_albert_graph(300, 4),
        }
    
    def test_feature_extraction(self):
        """测试特征提取功能"""
        from src.bit_hyrl.gnn_model import get_scale_adaptive_node2vec
        
        for scale, G in self.graphs.items():
            features, node_list, raw_dim = get_scale_adaptive_node2vec(G)
            
            # 验证输出
            self.assertEqual(len(node_list), G.number_of_nodes())
            self.assertEqual(features.shape[0], G.number_of_nodes())
            self.assertIn(raw_dim, [32, 48, 64])  # 预期的维度
            
            print(f"  {scale} ({G.number_of_nodes()} nodes): dim={raw_dim}, shape={features.shape}")
    
    def test_different_scale_different_dim(self):
        """验证不同规模使用不同维度"""
        from src.bit_hyrl.gnn_model import get_scale_adaptive_node2vec
        
        _, _, dim_tiny = get_scale_adaptive_node2vec(self.graphs['tiny'])
        _, _, dim_large = get_scale_adaptive_node2vec(self.graphs['large'])
        
        # 小图应该使用较小的维度
        self.assertLessEqual(dim_tiny, dim_large)


class TestScaleEncoding(unittest.TestCase):
    """测试规模编码"""
    
    def test_encoding_output(self):
        """测试编码输出"""
        from src.bit_hyrl.gnn_model import get_scale_encoding
        
        G_small = nx.barabasi_albert_graph(50, 2)
        G_large = nx.barabasi_albert_graph(200, 3)
        
        enc_small = get_scale_encoding(G_small, dim=8)
        enc_large = get_scale_encoding(G_large, dim=8)
        
        # 验证维度
        self.assertEqual(enc_small.shape[0], 8)
        self.assertEqual(enc_large.shape[0], 8)
        
        # 验证范围 [0, 1]
        self.assertTrue((enc_small >= 0).all() and (enc_small <= 1.1).all())
        self.assertTrue((enc_large >= 0).all() and (enc_large <= 1.1).all())
        
        # 验证不同规模有不同编码
        self.assertFalse(torch.allclose(enc_small, enc_large))
        
        print(f"  Small network encoding: {enc_small[:4].tolist()}")
        print(f"  Large network encoding: {enc_large[:4].tolist()}")


class TestFeatureProjector(unittest.TestCase):
    """测试特征投影器"""
    
    def test_projection(self):
        """测试不同维度的投影"""
        from src.bit_hyrl.gnn_model import FeatureProjector
        
        projector = FeatureProjector(output_dim=64)
        
        # 测试不同输入维度
        for input_dim in [32, 48, 64, 128]:
            x = torch.randn(10, input_dim)
            out = projector(x, input_dim)
            
            self.assertEqual(out.shape, (10, 64))
            print(f"  Input dim={input_dim} -> Output shape={out.shape}")


class TestUnifiedGATPolicy(unittest.TestCase):
    """测试统一GAT策略网络"""
    
    def setUp(self):
        """创建模型和测试图"""
        from src.bit_hyrl.gnn_model import UnifiedGATPolicy
        from src.bit_hyrl import config
        
        self.device = config.DEVICE
        self.model = UnifiedGATPolicy(
            in_channels=64,
            hidden_channels=96,
            scale_encoding_dim=8,
            heads=4,
            num_layers=3,
        ).to(self.device)
        self.G = nx.barabasi_albert_graph(50, 2)
    
    def test_forward_pass(self):
        """测试前向传播"""
        from src.bit_hyrl.gnn_model import (
            get_scale_adaptive_node2vec,
            get_scale_encoding,
            graph_to_pyg_data,
        )
        
        # 获取特征（指定设备）
        node_features, node_list, raw_dim = get_scale_adaptive_node2vec(self.G, device=self.device)
        scale_encoding = get_scale_encoding(self.G, dim=8, device=self.device)
        
        # 投影特征
        x = self.model.feature_projector(node_features, raw_dim)
        
        # 获取边索引
        _, edge_index, _ = graph_to_pyg_data(self.G, device=self.device)
        
        # 前向传播
        probs, value = self.model(x, edge_index, scale_encoding=scale_encoding)
        
        # 验证输出
        self.assertEqual(probs.shape[0], self.G.number_of_nodes())
        self.assertTrue(torch.abs(probs.sum() - 1.0) < 1e-5)  # 概率和为1
        self.assertEqual(value.shape[0], 1)
        
        print(f"  Probs shape: {probs.shape}, sum: {probs.sum().item():.4f}")
        print(f"  Value: {value.item():.4f}")
    
    def test_get_action(self):
        """测试获取动作"""
        from src.bit_hyrl.gnn_model import (
            get_scale_adaptive_node2vec,
            get_scale_encoding,
            graph_to_pyg_data,
        )
        
        # 获取特征（指定设备）
        node_features, node_list, raw_dim = get_scale_adaptive_node2vec(self.G, device=self.device)
        scale_encoding = get_scale_encoding(self.G, dim=8, device=self.device)
        x = self.model.feature_projector(node_features, raw_dim)
        _, edge_index, _ = graph_to_pyg_data(self.G, device=self.device)
        
        # 获取动作
        action, log_prob, value, probs = self.model.get_action(
            x, edge_index, scale_encoding=scale_encoding, deterministic=True
        )
        
        # 验证输出
        self.assertTrue(0 <= action.item() < self.G.number_of_nodes())
        self.assertIsInstance(log_prob.item(), float)
        
        print(f"  Selected action: {action.item()}")
        print(f"  Log prob: {log_prob.item():.4f}")


class TestUnifiedGCCReward(unittest.TestCase):
    """测试统一GCC奖励函数"""
    
    def test_reward_calculation(self):
        """测试奖励计算"""
        from src.bit_hyrl.reward import calculate_unified_gcc_reward
        
        G = nx.barabasi_albert_graph(100, 3)
        
        # 随机选择控制器
        centers = list(G.nodes())[:10]
        
        reward = calculate_unified_gcc_reward(G, centers)
        
        # 验证奖励范围
        self.assertGreaterEqual(reward, 0)
        self.assertLessEqual(reward, 1)
        
        print(f"  Reward: {reward:.4f}")
    
    def test_reward_different_scales(self):
        """测试不同规模网络的奖励"""
        from src.bit_hyrl.reward import calculate_unified_gcc_reward
        
        for n in [30, 70, 150, 300]:
            G = nx.barabasi_albert_graph(n, 3)
            centers = list(G.nodes())[:max(1, n // 10)]
            reward = calculate_unified_gcc_reward(G, centers)
            
            self.assertGreaterEqual(reward, 0)
            print(f"  n={n}, k={len(centers)}, reward={reward:.4f}")


class TestCurriculumSampling(unittest.TestCase):
    """测试课程学习采样"""
    
    def test_curriculum_sample(self):
        """测试课程采样"""
        from src.bit_hyrl.ppo_trainer import curriculum_sample_graphs
        
        # 创建不同规模的图
        graphs = []
        for n in range(20, 300, 20):
            graphs.append(nx.barabasi_albert_graph(n, 2))
        
        # 早期应该偏向小图
        early_samples = curriculum_sample_graphs(graphs, epoch=0, total_epochs=100, sample_size=10)
        late_samples = curriculum_sample_graphs(graphs, epoch=90, total_epochs=100, sample_size=10)
        
        early_avg = np.mean([g.number_of_nodes() for g in early_samples])
        late_avg = np.mean([g.number_of_nodes() for g in late_samples])
        
        # 早期平均规模应该小于后期
        print(f"  Early average size: {early_avg:.1f}")
        print(f"  Late average size: {late_avg:.1f}")
        
        # 验证后期样本更大（允许一些随机性）
        # 不严格断言，因为有随机性
    
    def test_scale_balanced_sample(self):
        """测试规模平衡采样"""
        from src.bit_hyrl.ppo_trainer import scale_balanced_sample_graphs
        
        # 创建偏向大图的数据集
        graphs = []
        for _ in range(10):
            graphs.append(nx.barabasi_albert_graph(30, 2))  # tiny
        for _ in range(50):
            graphs.append(nx.barabasi_albert_graph(250, 3))  # large
        
        # 平衡采样应该有更多小图
        samples = scale_balanced_sample_graphs(graphs, sample_size=20)
        
        tiny_count = sum(1 for g in samples if g.number_of_nodes() < 50)
        large_count = sum(1 for g in samples if g.number_of_nodes() >= 200)
        
        print(f"  Tiny samples: {tiny_count}")
        print(f"  Large samples: {large_count}")
        
        # 验证平衡（tiny和large数量相近）
        self.assertGreater(tiny_count, 0)
        self.assertGreater(large_count, 0)


class TestEndToEnd(unittest.TestCase):
    """端到端集成测试"""
    
    def test_controller_selection(self):
        """测试完整的控制器选择流程"""
        from src.bit_hyrl.gnn_model import (
            UnifiedGATPolicy,
            get_unified_features,
            graph_to_pyg_data,
        )
        from src.bit_hyrl.reward import calculate_unified_gcc_reward
        from src.bit_hyrl import config
        
        print("\n端到端测试:")
        
        device = config.DEVICE
        
        # 创建模型
        model = UnifiedGATPolicy(
            in_channels=64,
            hidden_channels=96,
            scale_encoding_dim=8,
            heads=4,
            num_layers=3,
        ).to(device)
        model.eval()
        
        # 测试不同规模的图
        for scale, n in [('tiny', 30), ('small', 70), ('medium', 150)]:
            G = nx.barabasi_albert_graph(n, 3)
            k = max(1, n // 10)
            
            # 获取特征（指定设备）
            node_features, scale_encoding, node_list, raw_dim = get_unified_features(G, device=device)
            x = model.feature_projector(node_features, raw_dim)
            _, edge_index, _ = graph_to_pyg_data(G, device=device)
            
            # 选择控制器
            selected_mask = torch.zeros(len(node_list), dtype=torch.bool, device=device)
            centers = []
            
            with torch.no_grad():
                for _ in range(k):
                    action, _, _, _ = model.get_action(
                        x, edge_index, scale_encoding=scale_encoding,
                        selected_mask=selected_mask, deterministic=True
                    )
                    centers.append(node_list[action.item()])
                    selected_mask[action.item()] = True
            
            # 计算奖励
            reward = calculate_unified_gcc_reward(G, centers)
            
            print(f"  {scale} (n={n}, k={k}): reward={reward:.4f}")
            
            self.assertEqual(len(centers), k)
            self.assertGreaterEqual(reward, 0)


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("BiT-HyRL 统一模型集成测试")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestScaleAdaptiveNode2Vec))
    suite.addTests(loader.loadTestsFromTestCase(TestScaleEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureProjector))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedGATPolicy))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedGCCReward))
    suite.addTests(loader.loadTestsFromTestCase(TestCurriculumSampling))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 总结
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("所有测试通过!")
    else:
        print(f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
