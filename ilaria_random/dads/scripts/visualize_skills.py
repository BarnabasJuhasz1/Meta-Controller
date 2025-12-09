import torch
import gymnasium as gym

# 🔹 Adjust these imports if your file is named differently
from dads import PolicyNet, extract_state  # make sure dads.py is in PYTHONPATH

CHECKPOINT_PATH = "dads_doorkey.pt"
ENV_ID = "MiniGrid-DoorKey-8x8-v0"
DEVICE = "cpu"   # or "cuda" if available


def infer_dims_from_checkpoint(ckpt):
    """Infer state_dim, action_dim, skill_dim, num_skills from checkpoint weights."""
    skills = ckpt["skills"]                 # shape: [num_skills, skill_dim]
    num_skills, skill_dim = skills.shape

    sd = ckpt["policy_state_dict"]

    first_linear_w = None
    last_linear_w = None

    # Assume MLP: first and last 2D weight tensors are input & output layers
    for name, tensor in sd.items():
        if tensor.dim() == 2:  # weight matrix [out_features, in_features]
            if first_linear_w is None:
                first_linear_w = tensor
            last_linear_w = tensor

    if first_linear_w is None or last_linear_w is None:
        raise RuntimeError("Could not infer dims: no 2D weights found in policy_state_dict.")

    input_dim = first_linear_w.shape[1]   # in_features of first layer
    action_dim = last_linear_w.shape[0]   # out_features of last layer

    state_dim = input_dim - skill_dim     # because input is [state, z]

    return state_dim, action_dim, skill_dim, num_skills


def load_policy_and_skills():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    state_dim, action_dim, skill_dim, num_skills = infer_dims_from_checkpoint(ckpt)
    print(f"Inferred dims: state_dim={state_dim}, skill_dim={skill_dim}, action_dim={action_dim}, num_skills={num_skills}")

    policy = PolicyNet(
        state_dim=state_dim,
        skill_dim=skill_dim,
        action_dim=action_dim,
    ).to(DEVICE)

    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    skills = ckpt["skills"].to(DEVICE).float()

    return policy, skills


def visualize_skills():
    policy, skills = load_policy_and_skills()
    num_skills, skill_dim = skills.shape

    env = gym.make(ENV_ID, render_mode="human")

    for k in range(num_skills):
        print("\n==============================")
        print(f"   Visualizing skill {k}")
        print("==============================\n")

        z = skills[k]  # [skill_dim]

        obs, info = env.reset()
        done = False

        while not done:
            # 🔹 MUST match the preprocessing used in training (same extract_state)
            state_vec = extract_state(env, obs)  # numpy or torch

            if not isinstance(state_vec, torch.Tensor):
                state_vec = torch.from_numpy(state_vec).float()
            state_vec = state_vec.to(DEVICE)

            with torch.no_grad():
                # assume PolicyNet.forward(state, z) -> logits over actions
                logits = policy(state_vec.unsqueeze(0), z.unsqueeze(0))
                action = logits.argmax(dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            env.render()

    env.close()


if __name__ == "__main__":
    visualize_skills()
