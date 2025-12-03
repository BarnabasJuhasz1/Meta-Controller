# Scripts/run.py

import argparse
import numpy as np

from Interface.loader import load_policy
from Env.registry import make_env
from Utils.Vis.visualizer import visualize_checkpoint

from Algos.LSD.trainer import LSDTrainer, LSDConfig



# ----------------------------------------------------------------------
# Trainer lookup table
# ----------------------------------------------------------------------
def get_trainer_class(algo):
    if algo == "LSD":
        return LSDTrainer, LSDConfig


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval", "vis"])

    parser.add_argument("--algo", type=str, help="LSD, DIAYN, RSD")
    parser.add_argument("--env", type=str, help="Environment name")
    parser.add_argument("--steps", type=int, default=100_000)

    parser.add_argument("--model", type=str)
    parser.add_argument("--out", type=str, default="runs")

    args = parser.parse_args()

    # ==========================================================
    # TRAIN
    # ==========================================================
    if args.mode == "train":
        TrainerClass, ConfigClass = get_trainer_class(args.algo)

        # LSD creates its own env inside the trainer, so ConfigClass=True needed
        if ConfigClass is LSDConfig:
            cfg = LSDConfig()
            cfg.num_episodes = args.steps  # match "steps" with episode count
            trainer = TrainerClass(cfg)
        else:
            # for baselines that expect env directly
            env = make_env(args.env)
            trainer = TrainerClass(env)

        trainer.train()
        trainer.save(f"{args.out}/{args.algo}_{args.steps}.pth")
        return

    # ==========================================================
    # EVAL
    # ==========================================================
    if args.mode == "eval":
        if args.model is None:
            raise ValueError("--model required in eval mode")

        env = make_env(args.env)
        policy = load_policy(args.model)

        obs = env.reset()
        done = False

        # Default skill = zero vector → DIAYN/RSD-friendly
        # LSD will override this with user-selected skill externally
        skill = np.zeros(policy.skill_dim(), dtype=np.float32)

        while not done:
            action = policy.act(obs, skill)
            obs, r, done, info = env.step(action)
            env.render()

        return

    # ==========================================================
    # VISUALIZE
    # ==========================================================
    if args.mode == "vis":
        if args.model is None:
            raise ValueError("--model required in vis mode")

        visualize_checkpoint(args.model, args.env, out_dir=args.out)
        return


if __name__ == "__main__":
    main()
