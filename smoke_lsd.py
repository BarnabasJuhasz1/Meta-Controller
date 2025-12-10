from oesd.algorithms.lsd import LSDConfig, LSDTrainer
cfg = LSDConfig(use_baseline_minigrid=True, obs_unit_scale=True, num_skills=8, size=8)
trainer = LSDTrainer(cfg)
print("obs_dim:", trainer.obs_dim)
obs = trainer._reset_env()
vec = trainer._obs_to_vec(obs)
print("sample obs (first 10):", vec.flatten()[:10])

def main():
	cfg = LSDConfig(use_baseline_minigrid=True, obs_unit_scale=True, num_skills=8, size=8)
	trainer = LSDTrainer(cfg)
	print("obs_dim:", trainer.obs_dim)
	obs = trainer._reset_env()
	vec = trainer._obs_to_vec(obs)
	print("sample obs (first 10):", vec.flatten()[:10])

if __name__ == '__main__':
	main()
