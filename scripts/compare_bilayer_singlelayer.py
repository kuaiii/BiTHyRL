# -*- coding: utf-8 -*-
"""
Compare bilayer network dismantling vs single-layer network dismantling

Bilayer network: BA-200 + randomly selected controller nodes (with cascade failure)
Single-layer network: BA-200 (no controllers, no cascade failure)

Attack modes: degree (targeted attack) and random (random attack)
Metric: GCC (Giant Connected Component)
"""
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from math import ceil

from src.simulation.dismantling import simulate_dismantling
import network_dismantling as nd  # 统一拆解接口
from network_metrics import compute as _nm_compute
from src.controller.strategies import random_select_controllers

# Fix random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def get_gcc_size_single_layer(G):
    """Calculate the size of the largest connected component in a single-layer network (without considering controllers)"""
    if G.number_of_nodes() == 0:
        return 0
    components = list(nx.connected_components(G))
    if not components:
        return 0
    return len(max(components, key=len))

def simulate_single_layer_dismantling(G, sequence, nodes_num):
    """
    Simulate single-layer network dismantling (no controllers, no cascade failure)
    
    Args:
        G (nx.Graph): Network graph (will be modified)
        sequence (list): Node removal sequence
        nodes_num (int): Initial total number of nodes (for normalization)
        
    Returns:
        tuple: (metrics_dict, state_history)
            - metrics_dict: Contains GCC curve data
                - 'gcc': [(gcc_normalized, remove_ratio), ...]
            - state_history: List of intermediate state information
                Each entry is a dict with keys: 'remove_ratio', 'total_nodes', 'total_edges', 
                'gcc_size', 'num_components', 'component_sizes'
    """
    metrics_dict = {'gcc': []}
    state_history = []
    
    # Initial state
    initial_gcc = get_gcc_size_single_layer(G)
    initial_gcc_normalized = initial_gcc / nodes_num
    metrics_dict['gcc'].append((initial_gcc_normalized, 0.0))
    
    # Record initial state with detailed information
    components = list(nx.connected_components(G))
    component_sizes = sorted([len(comp) for comp in components], reverse=True)
    initial_nodes = G.number_of_nodes()
    initial_edges = G.number_of_edges()
    network_density = (2 * initial_edges) / (initial_nodes * (initial_nodes - 1)) if initial_nodes > 1 else 0.0
    avg_degree = (2 * initial_edges) / initial_nodes if initial_nodes > 0 else 0.0
    max_component_size = component_sizes[0] if component_sizes else 0
    avg_component_size = sum(component_sizes) / len(component_sizes) if component_sizes else 0.0
    
    # Calculate detailed component information for initial state
    component_details = []
    for comp in components:
        comp_size = len(comp)
        component_details.append({
            'size': comp_size,
            'switches': comp_size,  # Single-layer: all nodes are switches
            'controllers': 0,
            'has_controller': False,
            'node_ids': sorted(list(comp))[:10]  # First 10 nodes for reference
        })
    
    state_history.append({
        'remove_ratio': 0.0,
        'attack_step': 0,  # Initial state is step 0
        'attacked_node': 'N/A',  # No attack yet
        'is_attacked_node_controller': False,
        'edges_removed_by_attack': 0,
        'total_nodes': initial_nodes,
        'total_edges': initial_edges,
        'network_density': network_density,
        'avg_degree': avg_degree,
        'switch_nodes': initial_nodes,  # Single-layer: all nodes are switches
        'controller_nodes': 0,  # Single-layer: no controllers
        'gcc_size': initial_gcc,
        'gcc': initial_gcc_normalized,
        'num_components': len(components),
        'num_controlled_components': 0,  # Single-layer: no controllers
        'num_uncontrolled_components': len(components),  # All components are uncontrolled
        'max_component_size': max_component_size,
        'avg_component_size': avg_component_size,
        'component_sizes': component_sizes,
        'controlled_component_sizes': [],
        'uncontrolled_component_sizes': component_sizes,
        'switches_in_controlled_components': 0,  # Single-layer: no controllers
        'switches_in_uncontrolled_components': initial_nodes,  # All nodes are in uncontrolled components
        'component_details': component_details,
        'cascade_removed_nodes': 0,  # Single-layer: no cascade failure
        'cascade_removed_node_ids': [],
        'nodes_removed_this_step': 0  # Initial state: no nodes removed
    })
    
    removed_count = 0
    attack_step = 0
    
    # Iterate through attack sequence
    for node in sequence:
        if not G.has_node(node):
            continue
        
        # Remove node (no cascade failure)
        G.remove_node(node)
        removed_count += 1
        attack_step += 1  # Increment attack step
        
        # Calculate current state
        current_nodes = G.number_of_nodes()
        x_val = (nodes_num - current_nodes) / nodes_num
        
        if current_nodes == 0:
            # Network completely collapsed
            metrics_dict['gcc'].append((0.0, x_val))
            state_history.append({
                'remove_ratio': x_val,
                'attack_step': attack_step,
                'attacked_node': node if 'node' in locals() else 'N/A',
                'is_attacked_node_controller': False,  # Single-layer: no controllers
                'edges_removed_by_attack': 0,
                'total_nodes': 0,
                'total_edges': 0,
                'switch_nodes': 0,
                'controller_nodes': 0,
                'gcc_size': 0,
                'gcc': 0.0,
                'num_components': 0,
                'num_controlled_components': 0,
                'num_uncontrolled_components': 0,
                'component_sizes': [],
                'controlled_component_sizes': [],
                'uncontrolled_component_sizes': [],
                'component_details': [],
                'cascade_removed_nodes': 0,  # Single-layer: no cascade failure
                'cascade_removed_node_ids': [],
                'nodes_removed_this_step': 1
            })
            break
        
        # Calculate GCC (single-layer network, directly calculate largest connected component)
        gcc_value = get_gcc_size_single_layer(G)
        gcc_normalized = gcc_value / nodes_num
        
        metrics_dict['gcc'].append((gcc_normalized, x_val))
        
        # Record intermediate state with detailed information
        components = list(nx.connected_components(G))
        component_sizes = sorted([len(comp) for comp in components], reverse=True)
        
        # Calculate detailed component information
        component_details = []
        for comp in components:
            comp_size = len(comp)
            component_details.append({
                'size': comp_size,
                'nodes': sorted(list(comp))[:10]  # First 10 nodes for reference
            })
        
        state_history.append({
            'remove_ratio': x_val,
            'attack_step': attack_step,
            'attacked_node': node,  # Record which node was attacked
            'total_nodes': current_nodes,
            'total_edges': G.number_of_edges(),
            'switch_nodes': current_nodes,  # Single-layer: all nodes are switches
            'controller_nodes': 0,  # Single-layer: no controllers
            'gcc_size': gcc_value,
            'gcc': gcc_normalized,
            'num_components': len(components),
            'component_sizes': component_sizes,
            'component_details': component_details,  # Detailed component information
            'cascade_removed_nodes': 0,  # Single-layer: no cascade failure
            'nodes_removed_this_step': 1  # Only the attacked node
        })
    
    return metrics_dict, state_history

def simulate_bilayer_dismantling(G, sequence, nodes_num, centers):
    """
    Simulate bilayer network dismantling (with controllers, with cascade failure)
    
    Args:
        G (nx.Graph): Network graph (will be modified)
        sequence (list): Node removal sequence
        nodes_num (int): Initial total number of nodes (for normalization)
        centers (set): Set of controller nodes (will be modified)
        
    Returns:
        tuple: (metrics_dict, state_history)
            - metrics_dict: Contains GCC curve data
                - 'gcc': [(gcc_normalized, remove_ratio), ...]
            - state_history: List of intermediate state information
                Each entry is a dict with keys: 'remove_ratio', 'total_nodes', 'total_edges',
                'switch_nodes', 'controller_nodes', 'gcc_size', 'num_components', 'component_sizes'
    """
    metrics_dict = {'gcc': []}
    state_history = []
    centers_set = centers.copy()  # Work with a copy to track changes
    
    # Initial state
    from network_metrics import compute as _nm_compute
    initial_gcc = _nm_compute('gcc_with_controllers', G, centers=list(centers_set))
    initial_gcc_normalized = initial_gcc / nodes_num
    metrics_dict['gcc'].append((initial_gcc_normalized, 0.0))
    
    # Record initial state with detailed information
    components = list(nx.connected_components(G))
    component_sizes = sorted([len(comp) for comp in components], reverse=True)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    initial_nodes = G.number_of_nodes()
    initial_edges = G.number_of_edges()
    network_density = (2 * initial_edges) / (initial_nodes * (initial_nodes - 1)) if initial_nodes > 1 else 0.0
    avg_degree = (2 * initial_edges) / initial_nodes if initial_nodes > 0 else 0.0
    max_component_size = component_sizes[0] if component_sizes else 0
    avg_component_size = sum(component_sizes) / len(component_sizes) if component_sizes else 0.0
    
    # Calculate detailed component information for initial state
    component_details = []
    controlled_components = []
    uncontrolled_components = []
    total_switches_in_controlled = 0
    total_switches_in_uncontrolled = 0
    
    for comp in components:
        comp_size = len(comp)
        comp_switches = [n for n in comp if n not in centers_set]
        comp_controllers = [n for n in comp if n in centers_set]
        has_controller = len(comp_controllers) > 0
        
        comp_detail = {
            'size': comp_size,
            'switches': len(comp_switches),
            'controllers': len(comp_controllers),
            'has_controller': has_controller,
            'node_ids': sorted(list(comp))[:10]  # First 10 nodes for reference
        }
        component_details.append(comp_detail)
        
        if has_controller:
            controlled_components.append(comp_size)
            total_switches_in_controlled += len(comp_switches)
        else:
            uncontrolled_components.append(comp_size)
            total_switches_in_uncontrolled += len(comp_switches)
    
    state_history.append({
        'remove_ratio': 0.0,
        'attack_step': 0,  # Initial state is step 0
        'attacked_node': 'N/A',  # No attack yet
        'is_attacked_node_controller': False,
        'edges_removed_by_attack': 0,
        'total_nodes': initial_nodes,
        'total_edges': initial_edges,
        'network_density': network_density,
        'avg_degree': avg_degree,
        'switch_nodes': len(switch_nodes),
        'controller_nodes': len(centers_set),
        'gcc_size': initial_gcc,
        'gcc': initial_gcc_normalized,
        'num_components': len(components),
        'num_controlled_components': len(controlled_components),
        'num_uncontrolled_components': len(uncontrolled_components),
        'max_component_size': max_component_size,
        'avg_component_size': avg_component_size,
        'component_sizes': component_sizes,
        'controlled_component_sizes': sorted(controlled_components, reverse=True),
        'uncontrolled_component_sizes': sorted(uncontrolled_components, reverse=True),
        'switches_in_controlled_components': total_switches_in_controlled,
        'switches_in_uncontrolled_components': total_switches_in_uncontrolled,
        'component_details': component_details,
        'cascade_removed_nodes': 0,  # Initial state: no cascade failure
        'cascade_removed_node_ids': [],
        'nodes_removed_this_step': 0  # Initial state: no nodes removed
    })
    
    removed_count = 0
    attack_step = 0
    
    # Iterate through attack sequence
    for node in sequence:
        if not G.has_node(node):
            continue
        
        # 1. Attack removal
        attacked_node = node
        is_attacked_node_controller = node in centers_set
        edges_before_removal = list(G.edges(node))
        num_edges_removed = len(edges_before_removal)
        
        G.remove_edges_from(edges_before_removal)
        G.remove_node(node)
        removed_count += 1
        attack_step += 1  # Increment attack step
        
        # Check if removed node is a controller
        if is_attacked_node_controller:
            centers_set.remove(node)
        
        # 2. Cascade failure logic
        cascade_removed_nodes = []
        if G.number_of_nodes() > 0:
            if len(centers_set) == 0:
                # No controllers left, all nodes cascade fail
                nodes_to_cascade_remove = list(G.nodes())
                for inode in nodes_to_cascade_remove:
                    if G.has_node(inode):
                        edges = list(G.edges(inode))
                        G.remove_edges_from(edges)
                        G.remove_node(inode)
                        cascade_removed_nodes.append(inode)
            else:
                # Check components without controllers
                components = list(nx.connected_components(G))
                nodes_to_cascade_remove = []
                
                for comp in components:
                    # Check if this component contains any controller
                    if centers_set.isdisjoint(comp):
                        # This component has no controller, cascade failure
                        nodes_to_cascade_remove.extend(list(comp))
                
                if nodes_to_cascade_remove:
                    for inode in nodes_to_cascade_remove:
                        if G.has_node(inode):
                            edges = list(G.edges(inode))
                            G.remove_edges_from(edges)
                            G.remove_node(inode)
                            cascade_removed_nodes.append(inode)
        
        # 3. Calculate metrics and record state
        current_nodes = G.number_of_nodes()
        x_val = (nodes_num - current_nodes) / nodes_num
        
        if current_nodes == 0:
            # Network completely collapsed
            metrics_dict['gcc'].append((0.0, x_val))
            state_history.append({
                'remove_ratio': x_val,
                'attack_step': attack_step,
                'attacked_node': attacked_node if 'attacked_node' in locals() else 'N/A',
                'is_attacked_node_controller': is_attacked_node_controller if 'is_attacked_node_controller' in locals() else False,
                'edges_removed_by_attack': num_edges_removed if 'num_edges_removed' in locals() else 0,
                'total_nodes': 0,
                'total_edges': 0,
                'switch_nodes': 0,
                'controller_nodes': 0,
                'gcc_size': 0,
                'gcc': 0.0,
                'num_components': 0,
                'num_controlled_components': 0,
                'num_uncontrolled_components': 0,
                'component_sizes': [],
                'controlled_component_sizes': [],
                'uncontrolled_component_sizes': [],
                'component_details': [],
                'cascade_removed_nodes': len(cascade_removed_nodes) if 'cascade_removed_nodes' in locals() else 0,
                'cascade_removed_node_ids': cascade_removed_nodes[:20] if 'cascade_removed_nodes' in locals() else [],
                'nodes_removed_this_step': 1 + (len(cascade_removed_nodes) if 'cascade_removed_nodes' in locals() else 0)
            })
            break
        else:
            # Calculate GCC
            from network_metrics import compute as _nm_compute
            centers_list = list(centers_set)
            gcc_value = _nm_compute('gcc_with_controllers', G, centers=centers_list)
            gcc_normalized = gcc_value / nodes_num
            
            metrics_dict['gcc'].append((gcc_normalized, x_val))
            
            # Record intermediate state with detailed information
            components = list(nx.connected_components(G))
            component_sizes = sorted([len(comp) for comp in components], reverse=True)
            switch_nodes = [n for n in G.nodes() if n not in centers_set]
            
            # Calculate network statistics
            current_edges = G.number_of_edges()
            network_density = (2 * current_edges) / (current_nodes * (current_nodes - 1)) if current_nodes > 1 else 0.0
            avg_degree = (2 * current_edges) / current_nodes if current_nodes > 0 else 0.0
            max_component_size = component_sizes[0] if component_sizes else 0
            avg_component_size = sum(component_sizes) / len(component_sizes) if component_sizes else 0.0
            
            # Calculate detailed component information
            component_details = []
            controlled_components = []
            uncontrolled_components = []
            total_switches_in_controlled = 0
            total_switches_in_uncontrolled = 0
            
            for comp in components:
                comp_size = len(comp)
                comp_switches = [n for n in comp if n not in centers_set]
                comp_controllers = [n for n in comp if n in centers_set]
                has_controller = len(comp_controllers) > 0
                
                comp_detail = {
                    'size': comp_size,
                    'switches': len(comp_switches),
                    'controllers': len(comp_controllers),
                    'has_controller': has_controller,
                    'node_ids': sorted(list(comp))[:10]  # First 10 nodes for reference
                }
                component_details.append(comp_detail)
                
                if has_controller:
                    controlled_components.append(comp_size)
                    total_switches_in_controlled += len(comp_switches)
                else:
                    uncontrolled_components.append(comp_size)
                    total_switches_in_uncontrolled += len(comp_switches)
            
            state_history.append({
                'remove_ratio': x_val,
                'attack_step': attack_step,
                'attacked_node': attacked_node,
                'is_attacked_node_controller': is_attacked_node_controller,
                'edges_removed_by_attack': num_edges_removed,
                'total_nodes': current_nodes,
                'total_edges': current_edges,
                'network_density': network_density,
                'avg_degree': avg_degree,
                'switch_nodes': len(switch_nodes),
                'controller_nodes': len(centers_set),
                'gcc_size': gcc_value,
                'gcc': gcc_normalized,
                'num_components': len(components),
                'num_controlled_components': len(controlled_components),
                'num_uncontrolled_components': len(uncontrolled_components),
                'max_component_size': max_component_size,
                'avg_component_size': avg_component_size,
                'component_sizes': component_sizes,
                'controlled_component_sizes': sorted(controlled_components, reverse=True),
                'uncontrolled_component_sizes': sorted(uncontrolled_components, reverse=True),
                'switches_in_controlled_components': total_switches_in_controlled,
                'switches_in_uncontrolled_components': total_switches_in_uncontrolled,
                'component_details': component_details,
                'cascade_removed_nodes': len(cascade_removed_nodes),
                'cascade_removed_node_ids': cascade_removed_nodes[:20],  # First 20 for reference
                'nodes_removed_this_step': 1 + len(cascade_removed_nodes)  # Attacked node + cascade nodes
            })
    
    return metrics_dict, state_history

def save_results_to_csv(results_dict, state_history, output_dir, attack_mode, network_type):
    """
    Save results to CSV file
    
    Args:
        results_dict (dict): Results dictionary, format: {'gcc': [(value, x), ...]}
        state_history (list): List of intermediate state information
        output_dir (str): Output directory
        attack_mode (str): Attack mode ('degree' or 'random')
        network_type (str): Network type ('bilayer' or 'singlelayer')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save GCC data
    gcc_data = results_dict['gcc']
    df = pd.DataFrame(gcc_data, columns=['GCC', 'Remove_Ratio'])
    filename = f"{network_type}_{attack_mode}_gcc_results.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"Saved: {filepath}")
    
    # Save intermediate state information
    if state_history:
        # Convert state_history to DataFrame with all detailed information
        state_data = []
        for state in state_history:
            row = {
                'Attack_Step': state.get('attack_step', 0),
                'Remove_Ratio': state['remove_ratio'],
                'Attacked_Node': state.get('attacked_node', 'N/A'),
                'Is_Attacked_Node_Controller': state.get('is_attacked_node_controller', False),
                'Edges_Removed_By_Attack': state.get('edges_removed_by_attack', 0),
                'Total_Nodes': state['total_nodes'],
                'Total_Edges': state['total_edges'],
                'Network_Density': state.get('network_density', 0.0),
                'Avg_Degree': state.get('avg_degree', 0.0),
                'Switch_Nodes': state['switch_nodes'],
                'Controller_Nodes': state['controller_nodes'],
                'GCC_Size': state['gcc_size'],
                'GCC_Normalized': state.get('gcc', state['gcc_size'] / (state.get('total_nodes', 1) or 1)),
                'Num_Components': state['num_components'],
                'Num_Controlled_Components': state.get('num_controlled_components', 0),
                'Num_Uncontrolled_Components': state.get('num_uncontrolled_components', 0),
                'Max_Component_Size': state.get('max_component_size', 0),
                'Avg_Component_Size': state.get('avg_component_size', 0.0),
                'Component_Sizes': str(state['component_sizes']),
                'Controlled_Component_Sizes': str(state.get('controlled_component_sizes', [])),
                'Uncontrolled_Component_Sizes': str(state.get('uncontrolled_component_sizes', [])),
                'Switches_In_Controlled_Components': state.get('switches_in_controlled_components', 0),
                'Switches_In_Uncontrolled_Components': state.get('switches_in_uncontrolled_components', 0),
                'Cascade_Removed_Nodes': state.get('cascade_removed_nodes', 0),
                'Cascade_Removed_Node_Ids': str(state.get('cascade_removed_node_ids', [])),
                'Nodes_Removed_This_Step': state.get('nodes_removed_this_step', 1),
                'Component_Details': str(state.get('component_details', []))  # Detailed component info
            }
            state_data.append(row)
        
        df_state = pd.DataFrame(state_data)
        filename_state = f"{network_type}_{attack_mode}_state_history.csv"
        filepath_state = os.path.join(output_dir, filename_state)
        df_state.to_csv(filepath_state, index=False, encoding='utf-8-sig')
        print(f"Saved detailed state history: {filepath_state}")
        
        # Also save a simplified version for quick analysis
        simplified_data = []
        for state in state_history:
            row = {
                'Attack_Step': state.get('attack_step', 0),
                'Remove_Ratio': state['remove_ratio'],
                'Total_Nodes': state['total_nodes'],
                'Total_Edges': state['total_edges'],
                'Network_Density': state.get('network_density', 0.0),
                'Avg_Degree': state.get('avg_degree', 0.0),
                'Switch_Nodes': state['switch_nodes'],
                'Controller_Nodes': state['controller_nodes'],
                'GCC_Size': state.get('gcc_size', 0),
                'GCC_Normalized': state.get('gcc', 0.0),
                'Num_Components': state['num_components'],
                'Num_Controlled_Components': state.get('num_controlled_components', 0),
                'Num_Uncontrolled_Components': state.get('num_uncontrolled_components', 0),
                'Max_Component_Size': state.get('max_component_size', 0),
                'Avg_Component_Size': state.get('avg_component_size', 0.0),
                'Switches_In_Controlled': state.get('switches_in_controlled_components', 0),
                'Switches_In_Uncontrolled': state.get('switches_in_uncontrolled_components', 0),
                'Cascade_Removed_Nodes': state.get('cascade_removed_nodes', 0),
                'Nodes_Removed_This_Step': state.get('nodes_removed_this_step', 1)
            }
            simplified_data.append(row)
        
        df_simplified = pd.DataFrame(simplified_data)
        filename_simplified = f"{network_type}_{attack_mode}_state_summary.csv"
        filepath_simplified = os.path.join(output_dir, filename_simplified)
        df_simplified.to_csv(filepath_simplified, index=False, encoding='utf-8-sig')
        print(f"Saved simplified state summary: {filepath_simplified}")

def plot_comparison(state_history_bd, state_history_br,
                    state_history_sd, state_history_sr,
                    output_dir):
    """
    Plot comparison line charts using attack steps as x-axis
    
    Args:
        state_history_bd: Bilayer network degree attack state history
        state_history_br: Bilayer network random attack state history
        state_history_sd: Single-layer network degree attack state history
        state_history_sr: Single-layer network random attack state history
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from state history
    def extract_data_from_history(state_history):
        x_vals = [state['attack_step'] for state in state_history]
        # Calculate normalized GCC from state history
        y_vals = []
        for state in state_history:
            if 'gcc' in state:
                y_vals.append(state['gcc'])
            elif 'gcc_size' in state and 'total_nodes' in state:
                # Calculate normalized GCC if not present
                total_nodes = state.get('total_nodes', 1)
                if total_nodes > 0:
                    y_vals.append(state['gcc_size'] / total_nodes)
                else:
                    y_vals.append(0.0)
            else:
                y_vals.append(0.0)
        return x_vals, y_vals
    
    # Extract data
    x_bd, y_bd = extract_data_from_history(state_history_bd)
    x_br, y_br = extract_data_from_history(state_history_br)
    x_sd, y_sd = extract_data_from_history(state_history_sd)
    x_sr, y_sr = extract_data_from_history(state_history_sr)
    
    # Main comparison plot
    plt.figure(figsize=(14, 10))
    
    # Bilayer network - degree attack
    plt.plot(x_bd, y_bd, 'b-', linewidth=2.5, marker='o', markersize=4, 
             label='Bilayer Network (Degree Attack)', alpha=0.8)
    
    # Bilayer network - random attack
    plt.plot(x_br, y_br, 'b--', linewidth=2.5, marker='s', markersize=4, 
             label='Bilayer Network (Random Attack)', alpha=0.8)
    
    # Single-layer network - degree attack
    plt.plot(x_sd, y_sd, 'r-', linewidth=2.5, marker='^', markersize=4, 
             label='Single-Layer Network (Degree Attack)', alpha=0.8)
    
    # Single-layer network - random attack
    plt.plot(x_sr, y_sr, 'r--', linewidth=2.5, marker='d', markersize=4, 
             label='Single-Layer Network (Random Attack)', alpha=0.8)
    
    plt.xlabel('Attack Step (Number of Targeted Nodes)', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized GCC', fontsize=14, fontweight='bold')
    plt.title('Bilayer Network vs Single-Layer Network Dismantling Comparison (BA-200)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='best')
    max_steps = max(max(x_bd) if x_bd else 0, max(x_br) if x_br else 0, 
                    max(x_sd) if x_sd else 0, max(x_sr) if x_sr else 0)
    plt.xlim(0, max_steps)
    plt.ylim(0, 1.05)
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"comparison_BA200_steps_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {plot_path}")
    
    # Plot degree and random comparisons separately
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Degree attack comparison
    ax1.plot(x_bd, y_bd, 'b-', linewidth=2.5, marker='o', markersize=4, 
             label='Bilayer Network', alpha=0.8)
    ax1.plot(x_sd, y_sd, 'r-', linewidth=2.5, marker='^', markersize=4, 
             label='Single-Layer Network', alpha=0.8)
    ax1.set_xlabel('Attack Step (Number of Targeted Nodes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized GCC', fontsize=12, fontweight='bold')
    ax1.set_title('Degree Attack Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    max_steps_degree = max(max(x_bd) if x_bd else 0, max(x_sd) if x_sd else 0)
    ax1.set_xlim(0, max_steps_degree)
    ax1.set_ylim(0, 1.05)
    
    # Random attack comparison
    ax2.plot(x_br, y_br, 'b--', linewidth=2.5, marker='s', markersize=4, 
             label='Bilayer Network', alpha=0.8)
    ax2.plot(x_sr, y_sr, 'r--', linewidth=2.5, marker='d', markersize=4, 
             label='Single-Layer Network', alpha=0.8)
    ax2.set_xlabel('Attack Step (Number of Targeted Nodes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized GCC', fontsize=12, fontweight='bold')
    ax2.set_title('Random Attack Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11)
    max_steps_random = max(max(x_br) if x_br else 0, max(x_sr) if x_sr else 0)
    ax2.set_xlim(0, max_steps_random)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, f"comparison_separated_BA200_steps_{timestamp}.png")
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    print(f"Saved separated comparison plots: {plot_path2}")
    
    plt.close('all')

def main():
    """Main function"""
    print("=" * 80)
    print("Bilayer Network vs Single-Layer Network Dismantling Comparison Experiment")
    print("=" * 80)
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # 1. Generate BA-200 network
    print("1. Generating BA-200 network...")
    N = 200
    m = 3
    G_original = nx.barabasi_albert_graph(N, m, seed=RANDOM_SEED)
    for u, v in G_original.edges():
        G_original[u][v]['weight'] = 1.0
    
    print(f"   Number of nodes: {G_original.number_of_nodes()}")
    print(f"   Number of edges: {G_original.number_of_edges()}")
    print()
    
    # 2. Bilayer network: randomly select controllers
    print("2. Bilayer network: randomly selecting controllers...")
    controller_rate = 0.1
    centers_bilayer = random_select_controllers(G_original, controller_rate)
    centers_set_bilayer = set(centers_bilayer)
    controller_num = len(centers_bilayer)
    print(f"   Number of controllers: {controller_num}")
    print(f"   Controller nodes: {centers_bilayer}")
    print()
    
    # 3. Generate attack sequences
    print("3. Generating attack sequences...")
    print("   - Degree attack sequence...")
    sequence_degree = get_dismantling_sequence(G_original.copy(), "degree")
    print(f"     Sequence length: {len(sequence_degree)}")
    
    print("   - Random attack sequence...")
    sequence_random = get_dismantling_sequence(G_original.copy(), "random")
    print(f"     Sequence length: {len(sequence_random)}")
    print()
    
    # 4. Create output directory
    output_dir = "results/comparison_bilayer_singlelayer"
    os.makedirs(output_dir, exist_ok=True)
    print(f"4. Output directory: {output_dir}")
    print()
    
    # 5. Bilayer network dismantling - Degree attack
    print("5. Bilayer network dismantling - Degree attack...")
    G_bd = G_original.copy()
    results_bilayer_degree, state_history_bd = simulate_bilayer_dismantling(
        G_bd, sequence_degree, N, centers_set_bilayer.copy()
    )
    save_results_to_csv(results_bilayer_degree, state_history_bd, output_dir, "degree", "bilayer")
    print(f"   Completed, {len(results_bilayer_degree['gcc'])} data points")
    print()
    
    # 6. Bilayer network dismantling - Random attack
    print("6. Bilayer network dismantling - Random attack...")
    G_br = G_original.copy()
    results_bilayer_random, state_history_br = simulate_bilayer_dismantling(
        G_br, sequence_random, N, centers_set_bilayer.copy()
    )
    save_results_to_csv(results_bilayer_random, state_history_br, output_dir, "random", "bilayer")
    print(f"   Completed, {len(results_bilayer_random['gcc'])} data points")
    print()
    
    # 7. Single-layer network dismantling - Degree attack
    print("7. Single-layer network dismantling - Degree attack...")
    G_sd = G_original.copy()
    results_single_degree, state_history_sd = simulate_single_layer_dismantling(
        G_sd, sequence_degree, N
    )
    save_results_to_csv(results_single_degree, state_history_sd, output_dir, "degree", "singlelayer")
    print(f"   Completed, {len(results_single_degree['gcc'])} data points")
    print()
    
    # 8. Single-layer network dismantling - Random attack
    print("8. Single-layer network dismantling - Random attack...")
    G_sr = G_original.copy()
    results_single_random, state_history_sr = simulate_single_layer_dismantling(
        G_sr, sequence_random, N
    )
    save_results_to_csv(results_single_random, state_history_sr, output_dir, "random", "singlelayer")
    print(f"   Completed, {len(results_single_random['gcc'])} data points")
    print()
    
    # 9. Plot comparison charts
    print("9. Plotting comparison charts...")
    plot_comparison(
        state_history_bd, state_history_br,
        state_history_sd, state_history_sr,
        output_dir
    )
    print()
    
    # 10. Output statistics
    print("10. Statistics:")
    print("=" * 80)
    
    def get_collapse_point_from_history(state_history):
        """Get attack step when GCC drops to 0.2"""
        for state in state_history:
            gcc_val = state.get('gcc', 0.0)
            if gcc_val <= 0.2:
                return state.get('attack_step', 0)
        # Return last attack step if never reached 0.2
        if state_history:
            return state_history[-1].get('attack_step', 0)
        return 0
    
    def calculate_r_value_from_history(state_history):
        """Calculate R value (area under curve) using attack steps as x-axis"""
        if not state_history:
            return 0.0
        
        x_vals = [state.get('attack_step', 0) for state in state_history]
        y_vals = []
        for state in state_history:
            if 'gcc' in state:
                y_vals.append(state['gcc'])
            elif 'gcc_size' in state:
                # Calculate normalized GCC if not present
                total_nodes = state.get('total_nodes', 1)
                if total_nodes > 0:
                    y_vals.append(state['gcc_size'] / total_nodes)
                else:
                    y_vals.append(0.0)
            else:
                y_vals.append(0.0)
        
        if len(x_vals) < 2:
            return 0.0
        
        return np.trapz(y_vals, x_vals)
    
    collapse_step_bd = get_collapse_point_from_history(state_history_bd)
    collapse_step_br = get_collapse_point_from_history(state_history_br)
    collapse_step_sd = get_collapse_point_from_history(state_history_sd)
    collapse_step_sr = get_collapse_point_from_history(state_history_sr)
    
    r_value_bd = calculate_r_value_from_history(state_history_bd)
    r_value_br = calculate_r_value_from_history(state_history_br)
    r_value_sd = calculate_r_value_from_history(state_history_sd)
    r_value_sr = calculate_r_value_from_history(state_history_sr)
    
    print("Bilayer Network - Degree Attack:")
    print(f"  Collapse Step (GCC=0.2): {collapse_step_bd}")
    print(f"  R Value: {r_value_bd:.4f}")
    print()
    
    print("Bilayer Network - Random Attack:")
    print(f"  Collapse Step (GCC=0.2): {collapse_step_br}")
    print(f"  R Value: {r_value_br:.4f}")
    print()
    
    print("Single-Layer Network - Degree Attack:")
    print(f"  Collapse Step (GCC=0.2): {collapse_step_sd}")
    print(f"  R Value: {r_value_sd:.4f}")
    print()
    
    print("Single-Layer Network - Random Attack:")
    print(f"  Collapse Step (GCC=0.2): {collapse_step_sr}")
    print(f"  R Value: {r_value_sr:.4f}")
    print()
    
    # Save statistics to CSV
    stats_data = {
        'Network_Type': ['Bilayer Network', 'Bilayer Network', 'Single-Layer Network', 'Single-Layer Network'],
        'Attack_Mode': ['Degree', 'Random', 'Degree', 'Random'],
        'Collapse_Step': [
            collapse_step_bd,
            collapse_step_br,
            collapse_step_sd,
            collapse_step_sr
        ],
        'R_Value': [
            r_value_bd,
            r_value_br,
            r_value_sd,
            r_value_sr
        ]
    }
    df_stats = pd.DataFrame(stats_data)
    stats_path = os.path.join(output_dir, "statistics_summary.csv")
    df_stats.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"Saved statistics: {stats_path}")
    print()
    
    print("=" * 80)
    print("Experiment completed!")
    print(f"All results saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
