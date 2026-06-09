# -*- coding: utf-8 -*-
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import os
import networkx as nx
from src.topology.generators import load_graph
from src.controller.strategies import weight_improved_GN, fast_GN_partition
from src.controller.rl_optimizer import train_offline

def main():
    """
    Offline training script for GNN-RL Agent.
    """
    datasets = ['Chinanet', 'GtsCe', 'UsCarrier', 'Colt', 'Cogentco']
    training_subgraphs = []
    
    print("Preparing training data...")
    for dataset_name in datasets:
        gml_path = f'dataset/all/{dataset_name}.gml'
        if not os.path.exists(gml_path):
            # Try alternate path
            gml_path = f'dataset/testdata/{dataset_name}.gml'
            if not os.path.exists(gml_path):
                continue
            
        G, _ = load_graph(gml_path)
        if G.number_of_nodes() == 0:
            continue
            
        edge_num = G.number_of_edges()
        # Use Fast GN to generate communities
        edge_to_remove = min(40, int(edge_num * 0.2))
        if edge_to_remove < 1: edge_to_remove = 1
        
        _, communities = fast_GN_partition(G, remove_count=edge_to_remove)
        
        # Add community subgraphs to training set
        for comm in communities:
            if len(comm) > 3: # Only train on meaningful subgraphs
                subgraph = G.subgraph(comm).copy()
                training_subgraphs.append(subgraph)
                
    print(f"Collected {len(training_subgraphs)} subgraphs for training.")
    
    # Train
    # Increase episodes for better convergence since we do it once
    train_offline(training_subgraphs, episodes_per_graph=20, save_path='src/train/v1/checkpoints/gnn_agent.pth')
    
if __name__ == "__main__":
    main()
