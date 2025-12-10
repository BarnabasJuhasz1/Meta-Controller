from oesd.algorithms.lsd import LSDConfig, LSDTrainer

cfg=LSDConfig()
# use baseline DoorKey env and unit-scaling like teammates
cfg.use_baseline_minigrid = True
cfg.obs_unit_scale = True
cfg.num_skills = 8
cfg.skill_dim = 8
# shortened retrain for diagnostics
cfg.num_episodes = 3000
cfg.autosave_interval = 1000
# lower entropy, more phi updates
cfg.entropy_coef = 0.01
cfg.phi_updates_per_episode = 128
cfg.seed = 42

trainer = LSDTrainer(cfg)
trainer.train()
trainer.save()
print('Retrain script completed.')
