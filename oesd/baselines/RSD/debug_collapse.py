#!/usr/bin/env python3
"""
Debugging script for identifying model collapse in RSD training.
This script helps diagnose which component (skill generator, policy, encoder) is collapsing.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import wandb
import os


class CollapseDebugger:
    """
    Debugger to track and visualize potential model collapse in RSD.
    
    This monitors:
    1. Skill Generator (SampleZPolicy) - checks if it's producing diverse skills
    2. Trajectory Encoder - checks if encoded states are diverse
    3. Policy Network - checks if actions are diverse
    4. Learned Skills - checks if final skills are distinct
    """
    
    def __init__(self, log_to_wandb=True, save_dir='./debug_logs'):
        self.log_to_wandb = log_to_wandb
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # History tracking
        self.history = defaultdict(list)
        
    def compute_diversity_metrics(self, samples, name=""):
        """
        Compute various diversity metrics for a set of samples.
        
        Args:
            samples: torch.Tensor or np.ndarray of shape [N, D]
            name: str, name of the component being analyzed
            
        Returns:
            dict of metrics
        """
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
            
        if len(samples.shape) == 1:
            samples = samples.reshape(-1, 1)
            
        metrics = {}
        
        # 1. Standard deviation (per dimension and overall)
        std_per_dim = np.std(samples, axis=0)
        metrics[f'{name}/std_mean'] = np.mean(std_per_dim)
        metrics[f'{name}/std_min'] = np.min(std_per_dim)
        metrics[f'{name}/std_max'] = np.max(std_per_dim)
        
        # 2. Pairwise distances
        from scipy.spatial.distance import pdist
        if samples.shape[0] > 1:
            pairwise_dists = pdist(samples, metric='euclidean')
            metrics[f'{name}/pairwise_dist_mean'] = np.mean(pairwise_dists)
            metrics[f'{name}/pairwise_dist_std'] = np.std(pairwise_dists)
            metrics[f'{name}/pairwise_dist_min'] = np.min(pairwise_dists)
            
            # 3. Effective rank (measures diversity)
            # Higher rank = more diverse
            U, S, Vh = np.linalg.svd(samples - samples.mean(axis=0), full_matrices=False)
            normalized_S = S / np.sum(S)
            entropy = -np.sum(normalized_S * np.log(normalized_S + 1e-10))
            effective_rank = np.exp(entropy)
            metrics[f'{name}/effective_rank'] = effective_rank
            metrics[f'{name}/max_rank'] = min(samples.shape[0], samples.shape[1])
            metrics[f'{name}/rank_ratio'] = effective_rank / min(samples.shape[0], samples.shape[1])
        
        # 4. Coefficient of variation (CV) - robustness metric
        mean_per_dim = np.mean(samples, axis=0)
        cv_per_dim = std_per_dim / (np.abs(mean_per_dim) + 1e-10)
        metrics[f'{name}/cv_mean'] = np.mean(cv_per_dim)
        
        return metrics
    
    def check_skill_generator(self, skill_generator, input_token, discrete=True, dim_option=8):
        """
        Check if the skill generator is producing diverse skills.
        
        Args:
            skill_generator: The SampleZPolicy network
            input_token: Input to the skill generator
            discrete: bool, whether skills are discrete
            dim_option: int, skill dimension
        """
        with torch.no_grad():
            # Sample multiple times to check diversity
            num_samples = 100
            device = next(skill_generator.parameters()).device
            
            if discrete:
                # For discrete skills, check the distribution over skills
                z_values = skill_generator(input_token).mean
                probabilities = torch.nn.functional.softmax(z_values, dim=-1)
                
                # Sample skills
                z_indices = []
                for _ in range(num_samples):
                    z_idx = torch.multinomial(probabilities, 1).squeeze(-1)
                    z_indices.append(z_idx.cpu().numpy())
                z_indices = np.array(z_indices)
                
                # Check distribution uniformity
                unique, counts = np.unique(z_indices, return_counts=True)
                skill_probs = probabilities[0].cpu().numpy()
                
                metrics = {
                    'skill_generator/num_unique_skills': len(unique),
                    'skill_generator/max_skill_prob': np.max(skill_probs),
                    'skill_generator/min_skill_prob': np.min(skill_probs),
                    'skill_generator/entropy': -np.sum(skill_probs * np.log(skill_probs + 1e-10)),
                    'skill_generator/max_entropy': np.log(dim_option),
                }
                
                # Save probability distribution plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(dim_option), skill_probs)
                ax.set_xlabel('Skill Index')
                ax.set_ylabel('Probability')
                ax.set_title('Skill Generator Distribution')
                ax.axhline(y=1.0/dim_option, color='r', linestyle='--', label='Uniform')
                ax.legend()
                
            else:
                # For continuous skills
                z_samples = []
                for _ in range(num_samples):
                    dist_z = skill_generator(input_token)
                    z = dist_z.sample()
                    z_samples.append(z.cpu().numpy())
                z_samples = np.concatenate(z_samples, axis=0)
                
                metrics = self.compute_diversity_metrics(z_samples, 'skill_generator')
                
                # Visualize skill distribution in 2D (PCA)
                from sklearn.decomposition import PCA
                if z_samples.shape[1] > 2:
                    pca = PCA(n_components=2)
                    z_2d = pca.fit_transform(z_samples)
                else:
                    z_2d = z_samples[:, :2]
                    
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.5)
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_title('Skill Generator Output Distribution (PCA)')
                ax.grid(True)
            
            save_path = os.path.join(self.save_dir, 'skill_generator_dist.png')
            plt.savefig(save_path)
            plt.close()
            
            if self.log_to_wandb and wandb.run is not None:
                wandb.log(metrics)
                wandb.log({'skill_generator/distribution': wandb.Image(save_path)})
                
            return metrics
    
    def check_encoder(self, encoder, observations, skill_options):
        """
        Check if the trajectory encoder is producing diverse embeddings.
        
        Args:
            encoder: The trajectory encoder network
            observations: Batch of observations
            skill_options: The skills used to generate these observations
        """
        with torch.no_grad():
            encoded = encoder(observations).mean
            
            # Overall diversity
            metrics = self.compute_diversity_metrics(encoded, 'encoder')
            
            # Per-skill diversity (are different skills producing different encodings?)
            if skill_options is not None:
                unique_skills = torch.unique(skill_options, dim=0)
                per_skill_variance = []
                
                for skill in unique_skills:
                    mask = (skill_options == skill).all(dim=1)
                    if mask.sum() > 1:
                        skill_encodings = encoded[mask]
                        variance = torch.var(skill_encodings, dim=0).mean().item()
                        per_skill_variance.append(variance)
                
                if per_skill_variance:
                    metrics['encoder/within_skill_variance'] = np.mean(per_skill_variance)
                
                # Between-skill variance (should be high)
                skill_means = []
                for skill in unique_skills:
                    mask = (skill_options == skill).all(dim=1)
                    if mask.sum() > 0:
                        skill_means.append(encoded[mask].mean(dim=0))
                
                if len(skill_means) > 1:
                    skill_means = torch.stack(skill_means)
                    between_metrics = self.compute_diversity_metrics(skill_means, 'encoder_between_skills')
                    metrics.update(between_metrics)
            
            if self.log_to_wandb and wandb.run is not None:
                wandb.log(metrics)
                
            return metrics
    
    def check_policy(self, policy, observations, skills, num_samples=50):
        """
        Check if the policy is producing diverse actions for different skills.
        
        Args:
            policy: The option policy network
            observations: Batch of observations
            skills: Batch of skill options
            num_samples: Number of action samples per obs-skill pair
        """
        with torch.no_grad():
            # Process observations
            processed_obs = policy.process_observations(observations)
            
            # Concatenate with skills
            if skills.dim() == 1:
                skills = skills.unsqueeze(1)
            concat_obs = torch.cat([processed_obs, skills.float()], dim=-1)
            
            # Sample actions
            dist, _ = policy(concat_obs)
            actions = dist.sample((num_samples,))  # [num_samples, batch, action_dim]
            
            # Compute diversity
            actions_flat = actions.view(-1, actions.shape[-1])
            metrics = self.compute_diversity_metrics(actions_flat, 'policy_actions')
            
            # Per-skill action diversity
            unique_skills = torch.unique(skills, dim=0)
            if len(unique_skills) > 1:
                per_skill_action_means = []
                for skill in unique_skills:
                    mask = (skills == skill).all(dim=1)
                    if mask.sum() > 0:
                        skill_actions = actions[:, mask, :].reshape(-1, actions.shape[-1])
                        per_skill_action_means.append(skill_actions.mean(dim=0))
                
                if len(per_skill_action_means) > 1:
                    per_skill_action_means = torch.stack(per_skill_action_means)
                    between_metrics = self.compute_diversity_metrics(
                        per_skill_action_means, 'policy_between_skills'
                    )
                    metrics.update(between_metrics)
            
            if self.log_to_wandb and wandb.run is not None:
                wandb.log(metrics)
                
            return metrics
    
    def check_learned_skills(self, trajectories, skill_labels):
        """
        Analyze if learned skills produce distinct behaviors.
        
        Args:
            trajectories: List of trajectory observations [num_traj, traj_len, obs_dim]
            skill_labels: Skill used for each trajectory [num_traj, skill_dim]
        """
        # Compute trajectory features (e.g., final position, trajectory variance)
        traj_features = []
        for traj in trajectories:
            if isinstance(traj, torch.Tensor):
                traj = traj.detach().cpu().numpy()
            
            # Features: start pos, end pos, mean pos, std pos
            features = np.concatenate([
                traj[0],  # start
                traj[-1],  # end
                traj.mean(axis=0),  # mean
                traj.std(axis=0),  # std
            ])
            traj_features.append(features)
        
        traj_features = np.array(traj_features)
        
        # Overall diversity
        metrics = self.compute_diversity_metrics(traj_features, 'learned_skills')
        
        # Per-skill analysis
        if skill_labels is not None:
            if isinstance(skill_labels, torch.Tensor):
                skill_labels = skill_labels.detach().cpu().numpy()
            
            unique_skills = np.unique(skill_labels, axis=0)
            per_skill_traj_means = []
            
            for skill in unique_skills:
                mask = (skill_labels == skill).all(axis=1)
                if mask.sum() > 1:
                    skill_trajs = traj_features[mask]
                    per_skill_traj_means.append(skill_trajs.mean(axis=0))
            
            if len(per_skill_traj_means) > 1:
                per_skill_traj_means = np.array(per_skill_traj_means)
                between_metrics = self.compute_diversity_metrics(
                    per_skill_traj_means, 'learned_skills_between'
                )
                metrics.update(between_metrics)
                
                # Visualize skill separation
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                traj_2d = pca.fit_transform(traj_features)
                
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Color by skill
                for i, skill in enumerate(unique_skills):
                    mask = (skill_labels == skill).all(axis=1)
                    ax.scatter(traj_2d[mask, 0], traj_2d[mask, 1], 
                              label=f'Skill {i}', alpha=0.6)
                
                ax.set_xlabel('PC 1')
                ax.set_ylabel('PC 2')
                ax.set_title('Trajectory Features by Skill (PCA)')
                ax.legend()
                ax.grid(True)
                
                save_path = os.path.join(self.save_dir, 'skill_separation.png')
                plt.savefig(save_path)
                plt.close()
                
                if self.log_to_wandb and wandb.run is not None:
                    wandb.log({'learned_skills/separation': wandb.Image(save_path)})
        
        if self.log_to_wandb and wandb.run is not None:
            wandb.log(metrics)
            
        return metrics
    
    def diagnose_collapse(self, all_metrics):
        """
        Analyze metrics to diagnose which component is collapsing.
        
        Args:
            all_metrics: dict of all computed metrics
            
        Returns:
            dict with diagnosis
        """
        diagnosis = {
            'skill_generator_healthy': True,
            'encoder_healthy': True,
            'policy_healthy': True,
            'learned_skills_healthy': True,
            'issues': []
        }
        
        # Check skill generator
        if 'skill_generator/entropy' in all_metrics:
            entropy = all_metrics['skill_generator/entropy']
            max_entropy = all_metrics['skill_generator/max_entropy']
            if entropy < 0.5 * max_entropy:
                diagnosis['skill_generator_healthy'] = False
                diagnosis['issues'].append(
                    f"Skill generator has low entropy ({entropy:.3f}/{max_entropy:.3f}). "
                    "It's not exploring diverse skills."
                )
        
        if 'skill_generator/rank_ratio' in all_metrics:
            rank_ratio = all_metrics['skill_generator/rank_ratio']
            if rank_ratio < 0.5:
                diagnosis['skill_generator_healthy'] = False
                diagnosis['issues'].append(
                    f"Skill generator has low effective rank ratio ({rank_ratio:.3f}). "
                    "Skills are collapsing to low-dimensional manifold."
                )
        
        # Check encoder
        if 'encoder/rank_ratio' in all_metrics:
            rank_ratio = all_metrics['encoder/rank_ratio']
            if rank_ratio < 0.3:
                diagnosis['encoder_healthy'] = False
                diagnosis['issues'].append(
                    f"Encoder has very low effective rank ({rank_ratio:.3f}). "
                    "Encoder is collapsing - not producing diverse embeddings."
                )
        
        if 'encoder/std_mean' in all_metrics:
            std_mean = all_metrics['encoder/std_mean']
            if std_mean < 0.01:
                diagnosis['encoder_healthy'] = False
                diagnosis['issues'].append(
                    f"Encoder outputs have very low variance ({std_mean:.6f}). "
                    "Encoder might be stuck."
                )
        
        # Check policy
        if 'policy_between_skills/pairwise_dist_mean' in all_metrics:
            dist_mean = all_metrics['policy_between_skills/pairwise_dist_mean']
            if dist_mean < 0.1:
                diagnosis['policy_healthy'] = False
                diagnosis['issues'].append(
                    f"Policy produces similar actions for different skills (distance={dist_mean:.3f}). "
                    "Policy is not differentiating between skills."
                )
        
        # Check learned skills
        if 'learned_skills_between/pairwise_dist_mean' in all_metrics:
            dist_mean = all_metrics['learned_skills_between/pairwise_dist_mean']
            if dist_mean < 0.5:
                diagnosis['learned_skills_healthy'] = False
                diagnosis['issues'].append(
                    f"Learned skills produce very similar trajectories (distance={dist_mean:.3f}). "
                    "Skills are not diverse."
                )
        
        # Generate summary
        if len(diagnosis['issues']) == 0:
            diagnosis['summary'] = "✓ All components appear healthy!"
        else:
            diagnosis['summary'] = f"⚠ Found {len(diagnosis['issues'])} issue(s):\n"
            for i, issue in enumerate(diagnosis['issues'], 1):
                diagnosis['summary'] += f"  {i}. {issue}\n"
        
        return diagnosis


def add_debugging_to_rsd():
    """
    Returns code snippet to add to RSD._train_components for debugging.
    """
    code = '''
# Add this at the end of RSD._train_components method (around line 573)

if wandb.run is not None and (self.NumSampleTimes == 1 or (runner.step_itr % 50 == 0)):
    from debug_collapse import CollapseDebugger
    
    debugger = CollapseDebugger(log_to_wandb=True, save_dir=wandb.run.dir + '/debug')
    
    all_metrics = {}
    
    # 1. Check skill generator
    print("\\n[DEBUG] Checking skill generator...")
    sg_metrics = debugger.check_skill_generator(
        self.SampleZPolicy, 
        self.input_token, 
        discrete=self.discrete, 
        dim_option=self.dim_option
    )
    all_metrics.update(sg_metrics)
    
    # 2. Check encoder
    print("[DEBUG] Checking encoder...")
    sample_data = self._sample_replay_buffer()
    enc_metrics = debugger.check_encoder(
        self.traj_encoder,
        sample_data['obs'][:100],
        sample_data['options'][:100]
    )
    all_metrics.update(enc_metrics)
    
    # 3. Check policy
    print("[DEBUG] Checking policy...")
    pol_metrics = debugger.check_policy(
        self.option_policy,
        sample_data['obs'][:50],
        sample_data['options'][:50]
    )
    all_metrics.update(pol_metrics)
    
    # 4. Diagnose
    print("[DEBUG] Diagnosing collapse...")
    diagnosis = debugger.diagnose_collapse(all_metrics)
    print(diagnosis['summary'])
    
    # Log diagnosis
    wandb.log({
        'debug/skill_generator_healthy': diagnosis['skill_generator_healthy'],
        'debug/encoder_healthy': diagnosis['encoder_healthy'],
        'debug/policy_healthy': diagnosis['policy_healthy'],
        'debug/num_issues': len(diagnosis['issues']),
        'epoch': runner.step_itr
    })
'''
    return code


if __name__ == '__main__':
    print("=" * 80)
    print("RSD Collapse Debugger")
    print("=" * 80)
    print("\nThis module provides tools to diagnose model collapse in RSD training.")
    print("\nTo use this debugger:")
    print("1. Import it in your RSD.py file")
    print("2. Add debugging calls in _train_components method")
    print("\nRecommended integration point:")
    print(add_debugging_to_rsd())
