import argparse
import os
import time
import json
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

import gymnasium as gym
from stable_baselines3 import PPO
from collections import defaultdict, Counter
from pathlib import Path

import glob
from PIL import Image

# Add project root to path if needed (though running as module usually handles this)
import sys
# Assuming running from root like `python oesd/ours/evaluation/eval.py`
# sys.path.append(os.getcwd()) 

from oesd.ours.unified_baseline_utils.skill_registry import SkillRegistry
from oesd.ours.contoller.meta_env_wrapper import MetaControllerEnv
from oesd.ours.unified_baseline_utils.SingleLoader import load_model_from_config, load_config
from minigrid.core.world_object import Door, Key

import re


# This key function finds all sequences of digits and converts them to integers
def natural_key(path):
    # This turns "model_100_.zip" into the list ['model_', 100, '_.zip']
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', path.name)]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Meta Controller")
    parser.add_argument("--env_name", type=str, default="minigrid")
    parser.add_argument("--skill_count_per_algo", type=int, default=10)
    parser.add_argument("--skill_duration", type=int, default=10)
    parser.add_argument("--config_path", type=str, default="oesd/ours/configs/config1.py")
    parser.add_argument("--model_path_or_dir", type=str, required=True, help="Path to the directory containing trained meta-controller model .zip files")
    # parser.add_argument("--model_path", type=str, required=False, help="Path to the trained meta-controller model .zip file")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--output_dir", type=str, default="oesd/ours/evaluation/results", help="Directory to save results")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main(args):
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
    
    print(f"Loading Meta-Controller from {args.model_path_or_dir}...")

    path = Path(args.model_path_or_dir)
    # Check if the input is a specific file or a directory
    if path.is_file() and path.suffix == '.zip':
        # If it's a file, create a list with just that one file
        zip_files = [path]
    else:
        # If it's a directory, search for all zip files inside
        zip_files = sorted(path.rglob('*.zip'), key=natural_key)

    for zip_file in zip_files:
        # load the model
        model = PPO.load(zip_file, device="cpu")
        # get the file name without extension
        filename = zip_file.stem
        # evaluate the model
        evaluate(args, meta_env, model, skill_registry, model_interfaces, filename)
        
    make_gif(args.output_dir)
    
    print(f"\nGIF saved to {args.output_dir}/skill_usage_over_time.gif")

def evaluate(args, meta_env, model, skill_registry, model_interfaces, filename):
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
        frames = [] # List to store frames for GIF
        
        # Capture initial frame
        initial_frame = meta_env.render()
        if initial_frame is not None:
            frames.append(Image.fromarray(initial_frame))

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
            
            # Step - Force render=True to capture frames
            obs, reward, terminated, truncated, info = meta_env.step(action, render=True)
            
            # Capture frame from info (it's in C, H, W format from wrapper)
            if 'render' in info:
                rendered_frames = info['render']
                # If it's a single frame (old behavior or just in case), make it a list
                if not isinstance(rendered_frames, list):
                    rendered_frames = [rendered_frames]
                
                for frame_chw in rendered_frames:
                    # Transpose back to H, W, C for PIL
                    frame_arr = frame_chw.transpose(1, 2, 0)
                    # Ensure it's uint8
                    if frame_arr.dtype != np.uint8:
                        frame_arr = frame_arr.astype(np.uint8)
                    frames.append(Image.fromarray(frame_arr))
            
            ep_reward += reward
            step_count += 1
        
        # Check Success (Heuristic: access the underlying env to see if goal reached)
        # In DoorKey, usually success is reaching the goal. 
        # rewards > 0 usually imply some progress, but let's check exact success condition if possible.
        # Or just rely on reward. For DoorKey, reward > 0 typically means solved (reward = 1 - 0.9*(step/max_steps))
        is_success = ep_reward > 0 
        
        if is_success:
            gif_path = os.path.join(args.output_dir, f"{filename}_ep{ep+1}_success.gif")
            if frames:
                frames[0].save(
                    gif_path, 
                    save_all=True, 
                    append_images=frames[1:], 
                    duration=25, # 100ms per frame = 10fps
                    loop=0
                )
            print(f"Episode {ep+1} Solved! GIF saved to {gif_path}")

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
    analyze_and_visualize(episode_metrics, args.output_dir, skill_registry, model_interfaces, filename)


def analyze_and_visualize(metrics, output_dir, registry, model_interfaces, filename):
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
    
    with open(os.path.join(output_dir, f"{filename}_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # --- Visualizations ---
    
    # 1. Skill Usage Histogram
    # plt.figure(figsize=(10, 6))
    # # Fill gaps for skills not used
    # x = range(len(registry.bag_of_skills))
    # y = [skill_counts.get(i, 0) for i in x]
    # colors = [model_interfaces[registry.get_algo_from_skill_idx(i)].algo_color for i in x]

    # # Create legend patches
    # handles = [mpatches.Patch(color=model_interfaces[i].algo_color, label=model_interfaces[i].algo_name) for i in model_interfaces.keys()]

    # plt.bar(x, y, color=colors, edgecolor='black', linewidth=1)
    
    # plt.xlabel("Global Skill ID")
    # plt.ylabel("Frequency")
    # plt.legend(handles=handles, loc="upper right")
    # plt.title(f"Skill Usage Distribution (Steps: {filename}, Succ. Rate: {success_rate:.2%}, Num. of Skills: {unique_skills_count})")
    # plt.savefig(os.path.join(output_dir, f"{filename}_skill_usage.png"))
    # plt.close()
    make_skill_usage_plot(output_dir, filename, registry, model_interfaces, skill_counts, success_rate, unique_skills_count)
    
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
    plt.savefig(os.path.join(output_dir, f"{filename}_hrl_timeline.png"))
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
    report_path = os.path.join(output_dir, f"{filename}_report.md")
    with open(report_path, "w") as f:
        f.write(f"# Evaluation Report\n\n")
        f.write(f"- **Success Rate**: {success_rate*100:.2f}%\n")
        f.write(f"- **Average Reward**: {avg_reward:.4f}\n")
        f.write(f"- **Active Skill Ratio**: {active_skill_ratio:.2f} ({unique_skills_count}/{len(registry.bag_of_skills)})\n")
        f.write(f"- **Entropy**: {entropy:.4f}\n\n")
        f.write(f"### Strategy Analysis\n")
        f.write(f"> {analysis_text}\n\n")
        f.write(f"![Skill Usage]({filename}_skill_usage.png)\n")
        f.write(f"![Timeline]({filename}_hrl_timeline.png)\n")
    
    print(f"\nReport saved to {report_path}")

def make_skill_usage_plot(output_dir, filename, registry, model_interfaces, skill_counts, success_rate, unique_skills_count):

    # 1. Skill Usage Histogram
    plt.figure(figsize=(10, 6))

    # --- Data Preparation for Seaborn ---
    # We build a list of dictionaries to convert into a DataFrame
    data = []
    palette_map = {}

    for i in range(len(registry.bag_of_skills)):
        # Retrieve algorithm info for this skill index
        algo_idx = registry.get_algo_from_skill_idx(i)
        interface = model_interfaces[algo_idx]
        
        # Store data for the plot
        data.append({
            "Global Skill ID": i,
            "Frequency": skill_counts.get(i, 0),
            # "Frequency": np.random.randint(0, 100),
            "Algorithm": interface.algo_name
        })
        
        # Build the color palette dict (Name -> Color)
        palette_map[interface.algo_name] = interface.algo_color

    df = pd.DataFrame(data)

    # --- Plotting ---
    ax = sns.barplot(
        data=df,
        x="Global Skill ID",
        y="Frequency",
        hue="Algorithm",      # Automatically colors bars based on Algorithm
        palette=palette_map,  # Uses your specific colors
        # edgecolor="black",
        # linewidth=0.5,
        edgecolor=None,
        linewidth=0,
        dodge=False,           # Keeps bars full width (prevents thinning when using hue)
        alpha=1.0
    )

    # --- Formatting ---
    # If you have many skills, the x-labels might get crowded. 
    # This helps keep the x-axis readable (optional, depends on skill count):
    if len(df) > 20:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.title(f"Skill Usage Distribution (Steps: {filename}, Succ. Rate: {success_rate:.2%}, Num. of Skills: {unique_skills_count})")

    # Move legend to upper right to match original placement
    plt.legend(loc="upper right", title="Algorithm")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_skill_usage.png"))
    plt.close()

def make_gif(frame_folder):
    # 1. Create the file pattern (e.g., all pngs starting with "frame_")
    # Change 'frame_*.png' to match your specific naming convention
    files = glob.glob(f"{frame_folder}/*_skill_usage.png")
    
    # 2. Sort the files
    # Standard python sort(). See "Handling Sorting" below if you have issues.
    files.sort() 

    # 3. Load the images
    frames = [Image.open(image) for image in files]
    
    # 4. Save the GIF
    # duration is in milliseconds (e.g., 500 = 0.5 seconds per frame)
    # loop=0 means loop forever
    frame_one = frames[0]
    frame_one.save(os.path.join(frame_folder, "skill_usage_over_time.gif"), format="GIF", append_images=frames[1:],
               save_all=True, duration=500, loop=0)

if __name__ == "__main__":
    args = parse_args()
    main(args)
