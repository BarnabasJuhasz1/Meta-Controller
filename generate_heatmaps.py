#!/usr/bin/env python3
"""
Generate state visitation heatmaps for all algorithms.

For each algorithm, creates a single PNG showing:
- The environment grid as background
- Overlaid heatmap of all cells visited by all skills combined
- Shows which areas of the state space each algorithm's skills explore

Output: One PNG per algorithm (LSD.png, METRA.png, DIAYN.png, DADS.png, RSD.png)
"""

import os
import sys

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'oesd/ours/unified_baseline_utils'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from oesd.ours.unified_baseline_utils.SingleLoader import load_config, load_model_from_config
from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry
from minigrid.envs import DoorKeyEnv

def collect_state_visits(adapter, skill_registry, episodes=10, horizon=400, stochastic=True, env_seed=42):
    """
    Collect all state visits for all skills of an algorithm.
    Returns: tuple of (visit_counts, env_info) where env_info contains object positions
    """
    env = DoorKeyEnv(size=8, max_steps=horizon, render_mode='rgb_array')
    # Reset with fixed seed to keep door/key/goal positions consistent
    env.reset(seed=env_seed)
    
    grid_size = 8
    visit_counts = np.zeros((grid_size, grid_size), dtype=int)
    position_extraction_failed = False  # Track if we've warned about position issues
    
    # Extract object positions from the environment
    env_info = {
        'door_pos': None,
        'key_pos': None,
        'goal_pos': None,
        'door_opened': False
    }
    
    # Find door, key, and goal positions in the grid
    for x in range(grid_size):
        for y in range(grid_size):
            cell = env.grid.get(x, y)
            if cell is not None:
                cell_type = cell.type
                if cell_type == 'door':
                    env_info['door_pos'] = (x, y)
                elif cell_type == 'key':
                    env_info['key_pos'] = (x, y)
                elif cell_type == 'goal':
                    env_info['goal_pos'] = (x, y)
    
    try:
        skills = skill_registry.get_skills_belonging_to_algo(adapter.algo_name)
    except:
        print(f"  ⚠️  Could not get skills for {adapter.algo_name}")
        return visit_counts
    
    print(f"  Collecting visits for {len(skills)} skills...")
    
    for skill_idx, skill_vec in enumerate(skills):
        for ep in range(episodes):
            # Use fixed seed to keep environment layout consistent
            obs, info = env.reset(seed=env_seed)
            
            # Process observation if needed
            if hasattr(adapter, 'process_obs'):
                proc_obs = adapter.process_obs(obs, env)
            else:
                proc_obs = obs
            
            for t in range(horizon):
                # Record position
                try:
                    if hasattr(env, 'agent_pos') and env.agent_pos is not None:
                        x, y = env.agent_pos
                        # Ensure valid coordinates
                        if 0 <= x < grid_size and 0 <= y < grid_size:
                            visit_counts[y, x] += 1
                except Exception as e:
                    # Silently skip if position extraction fails
                    pass
                
                # Get action (stochastic by default)
                try:
                    act = adapter.get_action(proc_obs, skill_vec, deterministic=(not stochastic))
                except:
                    # Fallback for adapters with different signatures
                    try:
                        act = adapter.get_action(proc_obs, deterministic=(not stochastic))
                    except:
                        act = env.action_space.sample()
                
                act_arr = np.asarray(act).reshape(-1)
                
                if act_arr.size > 1:
                    if stochastic:
                        # Ensure non-negative probabilities (clip negative values)
                        probs = np.clip(act_arr, 0, None)
                        probs_sum = probs.sum()
                        
                        # Handle edge cases: all zeros or NaN
                        if probs_sum < 1e-10 or np.isnan(probs_sum):
                            # Fall back to uniform distribution
                            probs = np.ones_like(probs) / len(probs)
                        else:
                            probs = probs / probs_sum
                        
                        # Final safety check
                        if not np.all(np.isfinite(probs)) or not np.all(probs >= 0):
                            probs = np.ones_like(probs) / len(probs)
                        
                        chosen = int(np.random.choice(len(act_arr), p=probs))
                    else:
                        chosen = int(np.argmax(act_arr))
                else:
                    chosen = int(np.round(act_arr[0]))
                
                chosen = int(np.clip(chosen, 0, adapter.action_dim - 1))
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(chosen)
                
                # Check if door was opened
                if env_info['door_pos']:
                    dx, dy = env_info['door_pos']
                    door_cell = env.grid.get(dx, dy)
                    if door_cell and door_cell.type == 'door' and door_cell.is_open:
                        env_info['door_opened'] = True
                
                if hasattr(adapter, 'process_obs'):
                    proc_obs = adapter.process_obs(obs, env)
                else:
                    proc_obs = obs
                
                if terminated or truncated:
                    break
    
    # Warn once if position extraction failed throughout
    if not position_extraction_failed and np.sum(visit_counts) == 0:
        print(f"  ⚠️  Warning: Could not extract agent positions (adapter may use different observation format)")
    
    env.close()
    return visit_counts, env_info


def plot_heatmap(visit_counts, algo_name, algo_color, output_path, env_info=None):
    """
    Create a heatmap visualization for one algorithm.
    env_info: dict with 'door_pos', 'key_pos', 'goal_pos' (x, y) tuples
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Create heatmap
    sns.heatmap(
        visit_counts,
        annot=False,
        fmt='d',
        cmap='YlOrRd',
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        cbar_kws={'label': 'Visit Count'}
    )
    
    # Add environment structure overlay
    # Draw border walls
    for i in range(8):
        ax.add_patch(Rectangle((0, i), 1, 1, fill=False, edgecolor='black', linewidth=3))
        ax.add_patch(Rectangle((7, i), 1, 1, fill=False, edgecolor='black', linewidth=3))
        ax.add_patch(Rectangle((i, 0), 1, 1, fill=False, edgecolor='black', linewidth=3))
        ax.add_patch(Rectangle((i, 7), 1, 1, fill=False, edgecolor='black', linewidth=3))
    
    # Draw vertical wall (DoorKeyEnv with seed 42: wall at x=2, door at (2,3))
    if env_info and env_info.get('door_pos'):
        door_x, door_y = env_info['door_pos']
        # Draw wall segments except where the door is
        for i in range(1, 7):  # Skip borders
            if i != door_y:
                ax.add_patch(Rectangle((door_x, i), 1, 1, fill=False, edgecolor='purple', linewidth=2, linestyle='--'))
    
    # Mark actual object positions if available
    if env_info:
        if env_info.get('door_pos'):
            dx, dy = env_info['door_pos']
            door_color = 'lightgreen' if env_info.get('door_opened') else 'green'
            ax.add_patch(Rectangle((dx, dy), 1, 1, fill=False, edgecolor=door_color, linewidth=3))
            ax.text(dx + 0.5, dy + 0.5, 'D', ha='center', va='center', fontsize=16, fontweight='bold', color=door_color)
        
        if env_info.get('key_pos'):
            kx, ky = env_info['key_pos']
            ax.add_patch(Rectangle((kx, ky), 1, 1, fill=False, edgecolor='blue', linewidth=3))
            ax.text(kx + 0.5, ky + 0.5, 'K', ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
        
        if env_info.get('goal_pos'):
            gx, gy = env_info['goal_pos']
            ax.add_patch(Rectangle((gx, gy), 1, 1, fill=False, edgecolor='gold', linewidth=3))
            ax.text(gx + 0.5, gy + 0.5, 'G', ha='center', va='center', fontsize=16, fontweight='bold', color='gold')
    
    ax.set_title(f'{algo_name} - State Visitation Heatmap\n(All Skills Combined)', 
                 fontsize=16, fontweight='bold', color=algo_color)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    
    plt.tight_layout()
    # Save as both PNG and PDF
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    print(f"  ✓ Saved: {pdf_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate state visitation heatmaps for all algorithms')
    parser.add_argument('--config', type=str, default='oesd/ours/configs/config1.py',
                        help='Path to config file')
    parser.add_argument('--episodes', type=int, default=8,
                        help='Number of episodes per skill')
    parser.add_argument('--horizon', type=int, default=300,
                        help='Max steps per episode')
    parser.add_argument('--outdir', type=str, default='./heatmaps',
                        help='Output directory for heatmap PNGs')
    parser.add_argument('--stochastic', action='store_true', default=True,
                        help='Use stochastic action sampling')
    parser.add_argument('--seed', type=int, default=42,
                        help='Environment seed for reproducibility')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load config
    config = load_config(args.config)
    
    # Get colors from config and map to algorithm names
    color_map = {}
    if hasattr(config, 'COLORS'):
        # Map algo names to color names
        algo_to_color_name = {
            'RSD': 'green',
            'LSD': 'pink',
            'DIAYN': 'red',
            'DADS': 'gold',
            'METRA': 'blue'
        }
        
        for algo_name, color_name in algo_to_color_name.items():
            if color_name in config.COLORS:
                color_map[algo_name] = config.COLORS[color_name]
        
        print(f"Using colors from config: {color_map}")
    
    print("\n" + "="*70)
    print("GENERATING STATE VISITATION HEATMAPS")
    print("="*70 + "\n")
    print(f"Seed: {args.seed}")
    print(f"Episodes per skill: {args.episodes}")
    print(f"Horizon: {args.horizon}")
    print(f"Stochastic: {args.stochastic}")
    print(f"Output directory: {args.outdir}")
    print()
    
    # Get unique algorithms
    algos = {}
    for m in config.model_cfgs:
        if m.algo_name.upper() not in algos:
            algos[m.algo_name.upper()] = m
    
    print(f"Found {len(algos)} algorithms: {list(algos.keys())}")
    print()
    
    # Process each algorithm
    for algo_name, model_cfg in algos.items():
        print(f"{'─'*70}")
        print(f"Processing: {algo_name}")
        print(f"{'─'*70}")
        print(f"  Checkpoint: {model_cfg.checkpoint_path}")
        
        try:
            # Load model
            skill_registry = SkillRegistry(model_cfg.skill_dim)
            adapter = load_model_from_config(model_cfg, skill_registry=skill_registry)
            
            # Collect state visits
            visit_counts, env_info = collect_state_visits(
                adapter, 
                skill_registry, 
                episodes=args.episodes,
                horizon=args.horizon,
                stochastic=args.stochastic,
                env_seed=args.seed
            )
            
            # Calculate coverage stats
            total_cells = 8 * 8
            visited_cells = np.sum(visit_counts > 0)
            coverage = visited_cells / total_cells * 100
            total_visits = np.sum(visit_counts)
            max_visits = np.max(visit_counts)
            
            print(f"  Coverage: {visited_cells}/{total_cells} cells ({coverage:.1f}%)")
            print(f"  Total visits: {total_visits:,}")
            print(f"  Max visits (single cell): {max_visits}")
            
            # Get color from config color_map, fallback to model_cfg.algo_color
            title_color = color_map.get(algo_name, model_cfg.algo_color)
            
            # Generate heatmap
            output_path = os.path.join(args.outdir, f"{algo_name}.png")
            plot_heatmap(visit_counts, algo_name, title_color, output_path, env_info)
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print()
    
    print("="*70)
    print(f"✓ All heatmaps saved to: {args.outdir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
