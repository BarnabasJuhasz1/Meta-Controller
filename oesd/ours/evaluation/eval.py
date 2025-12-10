import argparse
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from collections import defaultdict, Counter

# Add project root to path if needed (though running as module usually handles this)
import sys
# Assuming running from root like `python oesd/ours/evaluation/eval.py`
# sys.path.append(os.getcwd()) 

from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry
from oesd.ours.contoller.meta_env_wrapper import MetaControllerEnv
from oesd.ours.unified_baseline_utils.SingleLoader import load_model_from_config, load_config
from minigrid.core.world_object import Door, Key

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Meta Controller")
    parser.add_argument("--env_name", type=str, default="minigrid")
    parser.add_argument("--skill_count_per_algo", type=int, default=10)
    parser.add_argument("--skill_duration", type=int, default=10)
    parser.add_argument("--config_path", type=str, default="oesd/ours/configs/config1.py")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained meta-controller model .zip file")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--output_dir", type=str, default="oesd/ours/evaluation/results", help="Directory to save results")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def evaluate(args):
    # 1. Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading Configuration from {args.config_path}...")
    skill_registry = SkillRegistry(args.skill_count_per_algo)
    config = load_config(args.config_path)
    
    print("Loading Sub-policies (Adapters)...")
    adapters = [load_model_from_config(m, skill_registry=skill_registry) for m in config.model_cfgs]
    model_interfaces = {adapter.algo_name: adapter for adapter in adapters}
    
    print("Initializing Environment...")
    # Using rgb_array for potential recording, 'human' if rendering requested
    render_mode = "human" if args.render else "rgb_array"
    meta_env = MetaControllerEnv(
        skill_registry, 
        model_interfaces, 
        env_name=args.env_name, 
        skill_duration=args.skill_duration, 
        render_mode=render_mode
    )
    
    print(f"Loading Meta-Controller from {args.model_path}...")
    model = PPO.load(args.model_path)
    
    # 2. Evaluation Loop
    episode_metrics = []
    
    # For HRL Timeline (only keeping a few for visualization)
    timelines = [] 
    
    print(f"Starting Evaluation for {args.num_episodes} episodes...")
    
    for ep in range(args.num_episodes):
        obs, info = meta_env.reset(seed=args.seed + ep)
        terminated = False
        truncated = False
        
        ep_reward = 0
        step_count = 0
        
        # Tracking per episode
        skill_history = []  # List of (step, skill_id)
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            skill_id = int(action)
            
            # Record skill usage
            skill_history.append({
                "step_start": step_count * args.skill_duration,
                "step_end": (step_count + 1) * args.skill_duration, 
                "skill_id": skill_id,
                "algo": meta_env.registry.get_algo_and_skill_from_skill_idx(skill_id)[0]
            })
            
            # Step
            obs, reward, terminated, truncated, info = meta_env.step(action, render=args.render)
            
            ep_reward += reward
            step_count += 1
        
        # Check Success (Heuristic: access the underlying env to see if goal reached)
        # In DoorKey, usually success is reaching the goal. 
        # rewards > 0 usually imply some progress, but let's check exact success condition if possible.
        # Or just rely on reward. For DoorKey, reward > 0 typically means solved (reward = 1 - 0.9*(step/max_steps))
        is_success = ep_reward > 0 
        
        episode_metrics.append({
            "episode": ep,
            "reward": float(ep_reward),
            "steps": step_count, # Meta-steps
            "total_timesteps": step_count * args.skill_duration,
            "success": bool(is_success),
            "skill_history": skill_history
        })
        
        timelines.append(skill_history)
        
        print(f"Episode {ep+1}: Reward={ep_reward:.4f}, Steps={step_count}, Success={is_success}")

    # 3. Analyze Results
    analyze_and_visualize(episode_metrics, args.output_dir, skill_registry)

def analyze_and_visualize(metrics, output_dir, registry):
    # Aggregated Metrics
    rewards = [m["reward"] for m in metrics]
    successes = [m["success"] for m in metrics]
    success_rate = np.mean(successes)
    avg_reward = np.mean(rewards)
    
    print("\n--- Evaluation Results ---")
    print(f"Success Rate: {success_rate*100:.2f}%")
    print(f"Average Reward: {avg_reward:.4f}")
    
    # Skill Usage Analysis
    all_skills = []
    for m in metrics:
        for usage in m["skill_history"]:
            all_skills.append(usage["skill_id"])
            
    total_skills_used = len(all_skills)
    skill_counts = Counter(all_skills)
    
    unique_skills_count = len(skill_counts)
    active_skill_ratio = unique_skills_count / len(registry.bag_of_skills)
    
    print(f"Unique Skills Used: {unique_skills_count} / {len(registry.bag_of_skills)}")
    print(f"Active Skill Ratio: {active_skill_ratio:.2f}")

    # Save Metrics
    results = {
        "success_rate": success_rate,
        "average_reward": avg_reward,
        "active_skill_ratio": active_skill_ratio,
        "skill_usage": dict(skill_counts),
        "episodes": metrics
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # --- Visualizations ---
    
    # 1. Skill Usage Histogram
    plt.figure(figsize=(10, 6))
    # Fill gaps for skills not used
    x = range(len(registry.bag_of_skills))
    y = [skill_counts.get(i, 0) for i in x]
    
    plt.bar(x, y, color='skyblue', edgecolor='black')
    plt.xlabel("Global Skill ID")
    plt.ylabel("Frequency")
    plt.title(f"Skill Usage Distribution (Succ. Rate: {success_rate:.2%})")
    plt.savefig(os.path.join(output_dir, "skill_usage.png"))
    plt.close()
    
    # 2. HRL Timeline (for the first 5 episodes or best success)
    # Let's plot the first successful one and the first failed one if available, or just first few.
    num_timelines_to_plot = min(3, len(metrics))
    
    fig, axes = plt.subplots(num_timelines_to_plot, 1, figsize=(12, 4 * num_timelines_to_plot), sharex=True)
    if num_timelines_to_plot == 1:
        axes = [axes]
        
    for i in range(num_timelines_to_plot):
        ax = axes[i]
        ep_data = metrics[i]
        history = ep_data["skill_history"]
        
        # Plot bars
        # X axis: timesteps
        # Y axis: Skill ID
        
        # We can use broken_barh for this
        # [(start, width), (start, width)...]
        
        # We also want to color code by Algorithm if possible, or just distinct colors for skills
        
        # Simplify: Y-axis is Skill ID, X-axis is time. 
        # Plot points or short lines?
        # Better: Step plot or Gantt chart style
        
        times = []
        skills = []
        for h in history:
            times.append(h["step_start"])
            skills.append(h["skill_id"])
            # Add end point for step plot
            times.append(h["step_end"])
            skills.append(h["skill_id"])
            
        ax.plot(times, skills, drawstyle="steps-post", linewidth=2, label=f"Ep {i+1}")
        ax.set_ylabel("Skill ID")
        ax.set_ylim(-1, len(registry.bag_of_skills))
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Episode {i+1} Timeline (Success: {ep_data['success']}, Reward: {ep_data['reward']:.2f})")
        
        # Annotate algo names roughly? Maybe too cluttered.
        
    plt.xlabel("Timesteps")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hrl_timeline.png"))
    plt.close()
    
    # 3. Strategy Analysis (Textual)
    print("\n--- Strategy Analysis ---")
    # Entropy
    probs = np.array(list(skill_counts.values())) / total_skills_used
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    print(f"Skill Distribution Entropy: {entropy:.4f}")
    
    analysis_text = ""
    if entropy > 2.0 and success_rate < 0.2:
        analysis_text = "Scenario: Potential 'Jack of All Trades' / Random exploration. High entropy but low success."
    elif entropy < 1.0 and success_rate > 0.8:
        analysis_text = "Scenario: 'Specialist'. Low entropy (focused skills) with high success."
    elif success_rate > 0.8:
        analysis_text = "Scenario: 'Diverse Solver'. Using multiple skills effectively."
    else:
        analysis_text = "Scenario: Mixed / Learning in progress."
        
    print(analysis_text)

    # --- Save Report MD ---
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(f"# Evaluation Report\n\n")
        f.write(f"- **Success Rate**: {success_rate*100:.2f}%\n")
        f.write(f"- **Average Reward**: {avg_reward:.4f}\n")
        f.write(f"- **Active Skill Ratio**: {active_skill_ratio:.2f} ({unique_skills_count}/{len(registry.bag_of_skills)})\n")
        f.write(f"- **Entropy**: {entropy:.4f}\n\n")
        f.write(f"### Strategy Analysis\n")
        f.write(f"> {analysis_text}\n\n")
        f.write(f"![Skill Usage](skill_usage.png)\n")
        f.write(f"![Timeline](hrl_timeline.png)\n")
    
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
