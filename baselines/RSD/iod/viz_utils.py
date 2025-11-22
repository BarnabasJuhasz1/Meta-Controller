import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio

from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze, plot_trajectories, plot_value
import d4rl
import torch
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from tqdm import trange, tqdm
import copy

from iod.utils import get_torch_concat_obs
import torch.distributions as dist

def plot_trajectories(env, trajectories, fig, ax, color_list=None):
    if color_list is None:
        from itertools import cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_list = cycle(color_cycle)

    count = 0
    for color, trajectory in zip(color_list, trajectories):
        obs = np.array(trajectory['observation'])

        # convert back to xy?
        if 'ant' or 'minigrid'in env.env_name:
            all_x = []
            all_y = []
            for info in trajectory['info']:
                all_x.append(info['x'])
                all_y.append(info['y'])
            all_x = np.array(all_x)
            all_y = np.array(all_y)
        elif 'maze2d' in env.env_name:
            all_x = obs[:, 1] * 4 - 3.2
            all_y = obs[:, 0] * 4 - 3.2

        ax.scatter(all_x, all_y, s=5, c=color, alpha=0.2)
        ax.scatter(all_x[-1], all_y[-1], s=50, c=color, marker='*', alpha=1, edgecolors='black', label='traj.'+str(count))
        count += 1

    env.draw(ax)
    return ax

# --- Helper to handle trajectory plotting locally if AntMazeEnv is missing ---
# def plot_trajectories_local(env, trajectories, fig, ax):
#     """
#     Local implementation to plot trajectories. 
#     Handles scaling for MiniGrid if tile_size > 1.
#     """
#     if 'minigrid' in env.env_name.lower():
#         tile_size = 32 # Default minigrid tile size
#     else:
#         print("WARNING: NOT IN MINIGRID!")
#         tile_size = 1

#     colors = cm.rainbow(np.linspace(0, 1, len(trajectories)))
#     for i, traj in enumerate(trajectories):
#         # Extract coordinates from the tracking list
#         # Structure: traj['env_infos']['coordinates'] -> list of arrays
#         coords = np.array(traj['env_infos']['coordinates'])
        
#         # Scale coordinates if using MiniGrid pixel render
#         coords = coords * tile_size
        
#         ax.plot(coords[:, 0], coords[:, 1], color=colors[i], linewidth=0.8, alpha=0.7)
#         # Plot start/end
#         # ax.scatter(coords[0, 0], coords[0, 1], color=colors[i], s=5)
#         ax.scatter(coords[-1, 0], coords[-1, 1], color=colors[i], marker=".", s=20)
    
#     return ax

# def plot_trajectories_local(env, trajectories, fig, ax):
#     """
#     Local version of trajectory plotting that mimics plot_trajectories.
#     Uses trajectory['info'][i]['x'] / ['y'] exactly like plot_trajectories.
#     """

#     # Pick colors
#     colors = cm.rainbow(np.linspace(0, 1, len(trajectories)))

#     count = 0
#     for color, traj in zip(colors, trajectories):
#         infos = traj.get("info", None)
#         if infos is None:
#             print("Trajectory missing 'info' field.")
#             continue

#         # Extract x/y positions exactly like plot_trajectories
#         xs = []
#         ys = []
#         for info in infos:
#             xs.append(info["x"])
#             ys.append(info["y"])

#         xs = np.array(xs)
#         ys = np.array(ys)

#         # Path
#         ax.scatter(xs, ys, s=5, c=[color], alpha=0.2)

#         # Endpoint
#         ax.scatter(
#             xs[-1], ys[-1],
#             s=50, c=[color], marker='*', alpha=1,
#             edgecolors='black', label=f"traj.{count}"
#         )

#         count += 1

#     # Draw env map if provided
#     if hasattr(env, "draw"):
#         env.draw(ax)

#     return ax

# def plot_trajectories_local(env, trajectories, fig, ax, tile_size=32):
#     """
#     Local version of trajectory plotting that mimics plot_trajectories.
#     Uses trajectory['info'][i]['x'] / ['y'] exactly like plot_trajectories.
#     """

#     # Pick colors
#     colors = cm.rainbow(np.linspace(0, 1, len(trajectories)))

#     count = 0
#     for color, traj in zip(colors, trajectories):
#         infos = traj.get("info", None)
#         if infos is None:
#             print("Trajectory missing 'info' field.")
#             continue

#         # Extract x/y positions exactly like plot_trajectories
#         xs = []
#         ys = []
#         for info in infos:
#             xs.append(info["x"])
#             ys.append(info["y"])

#         # --- FIX START: Apply Tile Size Scaling ---
#         xs = np.array(xs) * tile_size
#         ys = np.array(ys) * tile_size
#         # --- FIX END ---

#         # Path
#         ax.scatter(xs, ys, s=5, c=[color], alpha=0.2)

#         # Endpoint
#         ax.scatter(
#             xs[-1], ys[-1],
#             s=50, c=[color], marker='*', alpha=1,
#             edgecolors='black', label=f"traj.{count}"
#         )

#         count += 1

#     # Draw env map if provided
#     # Note: We comment this out because PlotMazeTraj usually handles the background via imshow
#     # if hasattr(env, "draw"):
#     #     env.draw(ax)

#     return ax


def plot_trajectories_local(env, trajectories, fig, ax, tile_size=32):
    """
    Local version of trajectory plotting.
    Handles scaling, centering, and jittering for grid worlds.
    """

    # Pick colors
    colors = cm.rainbow(np.linspace(0, 1, len(trajectories)))

    count = 0
    for color, traj in zip(colors, trajectories):
        infos = traj.get("info", None)
        if infos is None:
            print("Trajectory missing 'info' field.")
            continue

        # Extract x/y positions
        xs = [] 
        ys = []
        for info in infos:
            xs.append(info["x"])
            ys.append(info["y"])

        xs = np.array(xs)
        ys = np.array(ys)

        # --- FIX START: Scaling, Centering, and Jittering ---
        if tile_size > 1:
            # 1. Scale to pixels
            xs = xs * tile_size
            ys = ys * tile_size
            
            # 2. Center in the tile (shift right/down by half a tile)
            half_tile = tile_size / 2.0
            xs += half_tile
            ys += half_tile
            
            # 3. Add random jitter
            # We generate ONE offset per trajectory so the line stays connected/smooth
            # Range: +/- 25% of the tile size (e.g., +/- 8 pixels for a 32px tile)
            # This keeps the dot inside the tile but separates it from others
            jitter_range = tile_size * 0.25 
            dx = np.random.uniform(-jitter_range, jitter_range)
            dy = np.random.uniform(-jitter_range, jitter_range)
            
            xs += dx
            ys += dy
        # --- FIX END ---

        # Path
        # ax.scatter(xs, ys, s=5, c=[color], alpha=0.2)
        ax.plot(xs, ys, color=color, alpha=0.2, linewidth=1.5)

        # Endpoint
        ax.scatter(
            xs[-1], ys[-1],
            s=50, c=[color], marker='*', alpha=1,
            edgecolors='black', label=f"traj.{count}"
        )

        count += 1

    # Draw env map if provided (usually handled by imshow in iod.py now)
    if hasattr(env, "draw"):
        # env.draw(ax)
        pass

    return ax

def Psi_baseline(x, *args, **kwargs):
    return x

def _vec_norm(vec):
    return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)

def calc_eval_metrics(trajectories, is_option_trajectories, coord_dims=[0,1]):
    eval_metrics = {}
    coords = []
    for traj in trajectories:
        traj1 = traj['env_infos']['coordinates'][:, coord_dims]
        traj2 = traj['env_infos']['next_coordinates'][-1:, coord_dims]
        coords.append(traj1)
        coords.append(traj2)
    coords = np.concatenate(coords, axis=0)
    uniq_coords = np.unique(np.floor(coords), axis=0)
    eval_metrics.update({
        'MjNumUniqueCoords': len(uniq_coords),
    })
    return eval_metrics

# save the traj. as fig
def PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=100, is_PCA=False, is_goal=True, ax=None):
    Repr_obs_array = np.array(All_Repr_obs_list[0])
    if is_goal:
        All_Goal_obs_array = np.array(All_Goal_obs_list[0])
    for i in range(1,len(All_Repr_obs_list)):
        Repr_obs_array = np.concatenate((Repr_obs_array, np.array(All_Repr_obs_list[i])), axis=0)
        if is_goal:
            All_Goal_obs_array = np.concatenate((All_Goal_obs_array, np.array(All_Goal_obs_list[i])), axis=0)
    # 创建 PCA 对象，指定降到2维
    if is_PCA:
        pca = PCA(n_components=2)
        # 对数据进行 PCA
        Repr_obs_2d = pca.fit_transform(Repr_obs_array)
    else:# # # Window Dist：

        Repr_obs_2d = Repr_obs_array
        if is_goal:
            All_Goal_obs_2d = All_Goal_obs_array
    # 绘制 PCA 降维后的数据
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0, 1, len(All_Repr_obs_list)))
    for i in range(0,len(All_Repr_obs_list)):
        color = colors[i]
        start_index = i * path_len
        end_index = (i+1) * path_len
        ax.scatter(Repr_obs_2d[start_index:end_index, 0], Repr_obs_2d[start_index:end_index, 1], color=color, s=5)
        if is_goal:
            ax.scatter(All_Goal_obs_2d[start_index:end_index, 0], All_Goal_obs_2d[start_index:end_index, 1], color=color, s=100, marker='*', edgecolors='black')
    path_file_traj = path + "-traj.png"
    ax.set_xlabel('z[0]')
    ax.set_ylabel('z[1]')
    ax.set_title('Repr of Traj. in Z Space')
    # plt.legend()
    if ax is None:
        plt.savefig(path_file_traj)
        plt.close()
        return
    else:
        return ax

def vec_norm(vec):
    return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)


def _get_concat_obs(obs, option):
    return get_torch_concat_obs(obs, option)

def _Psi(phi_x, phi_x0=None):
    if phi_x0 is None:
        return torch.tanh(1/150 * phi_x)
    else:
        return torch.tanh(1/150 * (phi_x-phi_x0))
        # return torch.tanh((phi_x-phi_x0))

def EstimateValue(policy, alpha, qf1, qf2, option, state, num_samples=1):
    batch = option.shape[0]
    # [s0, z]
    processed_cat_obs = _get_concat_obs(policy.process_observations(state), option.float())     # [b,dim_s+dim_z]
    
    # dist of pi(a|[s0, z])
    dist, info = policy(processed_cat_obs)    # [b, dim]
    actions = dist.sample((num_samples,))          # [n, b, dim]
    log_probs = dist.log_prob(actions).squeeze(-1)  # [n, b]
    
    processed_cat_obs_flatten = processed_cat_obs.repeat(num_samples, 1, 1).view(batch * num_samples, -1)      # [n*b, dim_s+z]
    actions_flatten = actions.view(batch * num_samples, -1)     # [n*b, dim_a]
    q_values = torch.min(qf1(processed_cat_obs_flatten, actions_flatten), qf2(processed_cat_obs_flatten, actions_flatten))      # [n*b, dim_1]
    
    alpha = alpha.param.exp()
        
    values = q_values - alpha * log_probs.view(batch*num_samples, -1)      # [n*b, 1]
    values = values.view(num_samples, batch, -1)        # [n, b, 1]
    E_V = values.mean(dim=0)        # [b, 1]
    
    weight = 1 - 0.99**(torch.clamp(torch.norm(option, p=2, dim=-1, keepdim=True)*300, min=75))
    E_V = E_V/weight 
    
    return E_V.squeeze(-1)

@torch.no_grad()
def viz_Value_in_Psi(policy, alpha, qf1, qf2, state, num_samples=10, device='cpu', path='./', fig=None):
    density = 200
    x = np.linspace(-1, 1, density)
    y = np.linspace(-1, 1, density)
    X, Y = np.meshgrid(x,y)
    if fig is None:
        fig = plt.figure(figsize=(18, 12), facecolor='w')
        ax3d = fig.add_subplot(111, projection='3d')
    else:
        ax3d = fig.add_subplot([0.52, 0.55, 0.4, 0.3], projection='3d')
    
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    pos = torch.tensor(pos).to(device)
    pos_flatten = pos.view(-1,2)
    option = pos_flatten
    state_batch = state.unsqueeze(0).repeat(option.shape[0], 1)

    V_flatten = EstimateValue(policy, alpha, qf1, qf2, option, state_batch, num_samples=10)
    V = V_flatten.view(pos.shape[0],pos.shape[1])
    print(V.max(), V.min())

    ax3d.plot_surface(X, Y, V.cpu().numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax3d.view_init(60, 270+20)
    ax3d.set_xlabel('X', fontsize=8, labelpad=-2)
    ax3d.set_ylabel('Y', fontsize=8, labelpad=-2)
    ax3d.set_zlabel('Value', fontsize=8, labelpad=-2)

    ax3d.tick_params(axis='x', labelsize=5, pad=-2)
    ax3d.tick_params(axis='y', labelsize=5, pad=-2)
    ax3d.tick_params(axis='z', labelsize=5, pad=-2)
        
    if fig is None:
        plt.savefig(path + '-Value' + '.png')
        print('save at: ' + path + '-Value' + '.png')
        plt.close()
        return 
    else: 
        return fig

@torch.no_grad()
def viz_Regert_in_Psi(base1, base2, state, num_samples=10, device='cpu', path='./'):
    def get_fuctions(base):
        return base['qf1'], base['qf2'], base['alpha'], base['policy'] 
    
    density = 200
    x = np.linspace(-1, 1, density)
    y = np.linspace(-1, 1, density)
    X, Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(18, 12), facecolor='w')
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    pos = torch.tensor(pos).to(device)
    pos_flatten = pos.view(-1,2)
    option = pos_flatten
    state_batch = state.unsqueeze(0).repeat(option.shape[0], 1)
    
    # value 1:
    qf1, qf2, alpha, policy = get_fuctions(base1)
    V1 = EstimateValue(policy, alpha, qf1, qf2, option, state_batch, num_samples=10)
    V1 = V1.view(pos.shape[0],pos.shape[1])
    
    # value 2:
    qf1, qf2, alpha, policy = get_fuctions(base2)
    V2 = EstimateValue(policy, alpha, qf1, qf2, option, state_batch, num_samples=10)
    V2 = V2.view(pos.shape[0],pos.shape[1])
    
    
    # Regret:
    Regret = V2 - V1
    print(Regret.max(), Regret.min())
    
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.plot_surface(X, Y, Regret.cpu().numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.view_init(60, 270+20)
    ax.set_xlabel('X')          
    ax.set_ylabel('Y')
    ax.set_zlabel('Regret')
    plt.savefig(path + '-Regret' + '.png')
    print('save at: ' + path + '-Regret' + '.png')
    plt.close()
    
@torch.no_grad()
def eval_cover_rate(env, agent_traj_encoder, agent_policy, dim_option, device, ax=None, max_path_length=300, Psi=_Psi, option_type=None):
    
    All_Repr_obs_list = []
    All_Goal_obs_list = []
    All_trajs_list = []
    FinallDistanceList = []
    ArriveList = []
    All_Cover_list = []
    eval_num = 100

    if 'minigrid' in env.env_name.lower():
        try:
            grid_w = env._env.width - 2 # exclude walls
            grid_h = env._env.height - 2
            # Standard MiniGrid tile_size is 32
            if hasattr(env._env, 'tile_size'):
                tile_size = env._env.tile_size
            else:
                tile_size = 32 
        except:
            grid_w, grid_h = 8, 8
            tile_size = 32

    if option_type != 'random':
        if 'minigrid' in env.env_name.lower():
            # Sample integer goals [1, width-2]
            np_random = np.random.default_rng(seed=0)
            GoalList = np_random.integers(1, min(grid_w, grid_h)+1, size=(eval_num, 2))
        # to do: fix the map from the ant-maze
        elif 'antmaze-medium' in env.env_name:
            np_random = np.random.default_rng(seed=0) 
            GoalList = env.goal_sampler(np_random=np_random)
            print(GoalList)
        else:
            # GoalList = np.load('path')
            GoalList = np.random.uniform(-1, 1, (eval_num, 2))

        eval_num = len(GoalList)
    else:
        # provide a fake Goal List
        GoalList = np.random.uniform(-1, 1, (eval_num, 2))
        eval_num = len(GoalList)

    options = np.random.uniform(-1,1, (eval_num, dim_option))
    
    for j in trange(eval_num):
        goal = GoalList[j]
        dist_theld = -1
        # ax.scatter(goal[0], goal[1], s=25, marker='o', alpha=1, edgecolors='black')
        # MiniGrid goals are integers, plot them slightly cleaner
        
        # If minigrid, scale goal coords to pixels for plotting over imshow
        plot_goal_x = goal[0] * tile_size if 'minigrid' in env.env_name.lower() else goal[0]
        plot_goal_y = goal[1] * tile_size if 'minigrid' in env.env_name.lower() else goal[1]

        if 'minigrid' in env.env_name.lower():
        #      dist_theld = -1.5 # Relaxed threshold for grid
        #      # Invert Y for plotting if using imshow (top-left origin) usually handled by scatter
        #      ax.scatter(goal[0], goal[1], s=50, marker='x', color='red', alpha=1)
        # else:
        #      ax.scatter(goal[0], goal[1], s=25, marker='o', alpha=1, edgecolors='black')
            dist_theld = -1.5 # Manhattan-ish distance < 1.5 implies arrival
            ax.scatter(plot_goal_x, plot_goal_y, s=60, marker='x', color='red', alpha=1, linewidth=2)
        else:
            ax.scatter(plot_goal_x, plot_goal_y, s=25, marker='o', alpha=1, edgecolors='black')

        # if 'maze2d' in env.env_name:
        #     goal_tmp = (goal + 3.2) / 4
        #     goal[0] = goal_tmp[1]
        #     goal[1] = goal_tmp[0]
        #     dist_theld = -1/4

        tensor_goal = torch.tensor(goal).to(device)
        # s0
        obs_0 = env.reset()
        obs_0 = torch.tensor(obs_0).unsqueeze(0).to(device).float()
        obs = copy.deepcopy(obs_0)
        phi_obs_ = agent_traj_encoder(obs).mean
        phi_obs0 = copy.deepcopy(phi_obs_)
        # goal
        if option_type == 'random':
            # option = vec_norm(torch.tensor(options[j]).unsqueeze(0).to(device).float())
            option = torch.tensor(options[j]).unsqueeze(0).to(device).float()
        else: 
            if hasattr(env, 'get_target_obs'):
                print("ENV HAS ATTRIBUTE get_target_obs")
                target_obs = env.get_target_obs(obs_0, tensor_goal)
                phi_target_obs = agent_traj_encoder(target_obs).mean
                if option_type == 'baseline':
                    option = _vec_norm(phi_target_obs - phi_obs0)
                elif 'Projection' in option_type:
                    option = Psi(phi_target_obs, phi_obs0)
                    # option = _vec_norm(option)
                elif 'uniform' in option_type:
                    option = torch.tensor(options[j]).unsqueeze(0).to(device).float()
            else:
                print("ENV DID NOT IMPLEMENT ATTRIBUTE get_target_obs")
                # Fallback for MiniGrid if get_target_obs missing
                option = torch.tensor(options[j]).unsqueeze(0).to(device).float()

        Repr_obs_list = []
        Repr_goal_list = []
        gt_return_list = []
        traj_list = {}
        traj_list["observation"] = []
        traj_list["info"] = []
        Cover_list = {}
        arrive = 0
        for t in range(max_path_length):
            phi_obs_ = agent_traj_encoder(obs).mean
            
            obs_option = torch.cat((obs, option), -1).float()
            # for viz
            # import pdb; pdb.set_trace()
            Repr_obs_list.append(Psi(phi_obs_, phi_obs0).cpu().numpy()[0])
            Repr_goal_list.append(option.cpu().numpy()[0])
            # get actions from policy
            action, agent_info = agent_policy.get_action(obs_option)
            # interact with the env
            obs, reward, dones, info = env.step(action)
            
            if 'minigrid' in env.env_name.lower():
                # print("MINIGRIDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDd")
                # MiniGrid: Extract integer coordinates from unwrapped env
                # agent_pos = env.unwrapped.agent_pos
                # info['x'], info['y'] = agent_pos[0], agent_pos[1]
                # # Use Euclidean or Manhattan distance for grid
                # gt_dist = np.linalg.norm(goal - np.array(agent_pos))
                # current_coords = np.array(agent_pos)

                # current_coords = info['coordinates']
                # # Check arrival against Logic Goal (not scaled)
                # gt_dist = np.linalg.norm(goal - current_coords)
                current_coords = obs[:2]

                gt_dist = np.linalg.norm(goal - current_coords)
                if hasattr(env.env, 'get_xy'):
                    info['x'], info['y'] = env.env.get_xy()
                else:
                    info['x'], info['y'] = obs[0], obs[1]
            else:
                # Standard Maze2d Logic
                current_coords = obs[:2]

                gt_dist = np.linalg.norm(goal - current_coords)
                if hasattr(env.env, 'get_xy'):
                    info['x'], info['y'] = env.env.get_xy()
                else:
                    info['x'], info['y'] = obs[0], obs[1]

            # gt_dist = np.linalg.norm(goal - obs[:2])
            
            # if hasattr(env.env, 'get_xy'):
            #     info['x'], info['y'] = env.env.get_xy()
            # else:
            #     info['x'], info['y'] = obs[0], obs[1]
            
            traj_list["observation"].append(obs)
            traj_list["info"].append(info)
            if 'env_infos' not in Cover_list:
                Cover_list['env_infos'] = {}
                Cover_list['env_infos']['coordinates'] = []
                Cover_list['env_infos']['next_coordinates'] = []
            # Cover_list['env_infos']['coordinates'].append(obs[:2])
            # Cover_list['env_infos']['next_coordinates'].append(obs[:2])
            Cover_list['env_infos']['coordinates'].append(current_coords)
            Cover_list['env_infos']['next_coordinates'].append(current_coords)
            
            obs = torch.tensor(obs).unsqueeze(0).to(device).float()
            gt_reward = - gt_dist / (30 * max_path_length)
            gt_return_list.append(gt_reward)
            if -gt_dist > dist_theld:
                arrive = 1
                # ax.scatter(goal[0], goal[1], s=100, marker='o', alpha=1, edgecolors='black')
                # ax.scatter(goal[0], goal[1], s=100, marker='*', alpha=1, edgecolors='white', color='gold')
                # Plot Arrival Star (Scaled)
                # p_x = current_coords[0] * tile_size if 'minigrid' in env.env_name.lower() else current_coords[0]
                # p_y = current_coords[1] * tile_size if 'minigrid' in env.env_name.lower() else current_coords[1]
                print('arrive', (plot_goal_x, plot_goal_y))
                ax.scatter(plot_goal_x, plot_goal_y, s=120, marker='*', alpha=1, edgecolors='white', color='gold')
                if option_type != 'random':
                    break
        
        if arrive == 1:
            ArriveList.append(1)
        else:
            ArriveList.append(0)
            
        All_Repr_obs_list.append(Repr_obs_list)
        All_Goal_obs_list.append(Repr_goal_list)
        All_trajs_list.append(traj_list)
        FinallDistanceList.append(-gt_dist)
        Cover_list['env_infos']['coordinates'] = np.array(Cover_list['env_infos']['coordinates'])
        Cover_list['env_infos']['next_coordinates'] = np.array(Cover_list['env_infos']['next_coordinates'])
        All_Cover_list.append(Cover_list)

    return ax, FinallDistanceList, All_Repr_obs_list, All_Goal_obs_list, All_trajs_list, FinallDistanceList, ArriveList, All_Cover_list       
      
def viz_SZN_dist_circle(SZN, input_token, path, psi_z=None, ax=None):
    dist = SZN(input_token)
    auto_save = 0
    from matplotlib.patches import Ellipse
    if ax is None:
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        auto_save = 1
    for i in range(1):
        mu_x = dist.mean[i][0].detach().cpu().numpy()
        sigma_x = dist.stddev[i][0].detach().cpu().numpy()
        mu_y = dist.mean[i][1].detach().cpu().numpy()
        sigma_y = dist.stddev[i][1].detach().cpu().numpy()
        e = Ellipse(xy = (mu_x,mu_y), width = sigma_x * 2, height = sigma_y * 2, angle=0)
        ax.add_artist(e)
        
    if psi_z is not None:
        ax.scatter(psi_z[:, 0], psi_z[:, 1], marker='*', alpha=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title('Dist. of SZN in Z Space')
    if auto_save:
        plt.savefig(path + '-c' + '.png')
        print("save at:", path + '-c' + '.png')
        plt.close()
        return 
    else:
        return ax

def viz_dist_circle(window, path=None, psi_z=None, ax=None):
    from matplotlib.patches import Ellipse
    auto_save = 0
    if ax is None:
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        auto_save = 1
    for i in range(len(window)):
        dist = window[i]
        for i in range(1):
            mu_x = dist.mean[i][0].detach().cpu().numpy()
            sigma_x = dist.stddev[i][0].detach().cpu().numpy()
            mu_y = dist.mean[i][1].detach().cpu().numpy()
            sigma_y = dist.stddev[i][1].detach().cpu().numpy()
            e = Ellipse(xy = (mu_x,mu_y), width = sigma_x * 2, height = sigma_y * 2, angle=0)
            e.set_edgecolor("blue")      
            e.set_linewidth(1.5)          
            e.set_facecolor((1, 1, 1, 0)) 
            ax.add_artist(e)
        
    if psi_z is not None:
        ax.scatter(psi_z[:, 0], psi_z[:, 1], marker='*', alpha=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title('Dist. of SZN in Z Space')
    if auto_save:
        plt.savefig(path + '-c' + '.png')
        print("save at:", path + '-c' + '.png')
        plt.close()
        return 
    else:
        return ax


def viz_GMM_circle(GMM, path='./', psi_z=None, ax=None):
    from matplotlib.patches import Ellipse
    means_from_component = GMM.component_distribution.base_dist.loc
    stddevs_from_component = GMM.component_distribution.base_dist.scale

    auto_save = 0
    if ax is None:
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        auto_save = 1
    for i in range(len(means_from_component)):
        mu_x = means_from_component[i][0].detach().cpu().numpy()
        sigma_x = stddevs_from_component[i][0].detach().cpu().numpy()
        mu_y = means_from_component[i][1].detach().cpu().numpy()
        sigma_y = stddevs_from_component[i][1].detach().cpu().numpy()
        e = Ellipse(xy = (mu_x,mu_y), width = sigma_x * 2, height = sigma_y * 2, angle=0)
        e.set_edgecolor("blue")      
        e.set_linewidth(1.5)          
        e.set_facecolor((1, 1, 1, 0)) 
        ax.add_artist(e)
        
    if psi_z is not None:
        ax.scatter(psi_z[:, 0], psi_z[:, 1], marker='*', alpha=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title('Dist. of SZN in Z Space')
    if auto_save:
        plt.savefig(path + '-c' + '.png')
        print("save at:", path + '-c' + '.png')
        plt.close()
        return 
    else:
        return ax



@torch.no_grad()
def PlotMazeTrajDist(env, SZN, input_token, agent_traj_encoder, qf1, qf2, alpha, policy, device, Psi, dim_option=2, max_path_length=300, path='./'):    
    obs0 = env.reset()
    s0 = torch.tensor(obs0).to(device).float()
    fig, ax = plt.subplots(2,2)
    fig.subplots_adjust(wspace=0.4, hspace=0.4) 
    np_random = np.random.default_rng(seed=0) 
    env.draw(ax[0,0])
    ax[0,0].set_title('State of Traj. in Maze')
    ax[0,1].set_axis_off()
    ax[0,1].set_title('Estimate Value in Z Space')
    fig = viz_Value_in_Psi(policy, alpha, qf1, qf2, state=s0, num_samples=10, device=device, path=path, fig=fig)
    ax[0,0], FinallDistanceList, All_Repr_obs_list, All_Goal_obs_list, All_trajs_list, FinallDistanceList, ArriveList = eval_cover_rate(env, agent_traj_encoder, policy, dim_option, device, ax=ax[0,0], max_path_length=max_path_length, Psi=Psi)
    # calculate metrics
    FD = np.array(FinallDistanceList).mean()
    AR = np.array(ArriveList).mean()
    print("FD:", FD, '\n', "AR:", AR)
    # ax[0,0] = plot_trajectories(env, All_trajs_list, fig, ax[0,0])
    ax[0,0] = plot_trajectories_local(env, All_trajs_list, fig, ax[0,0], tile_size)
    ax[1,0] = PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length, is_goal=True, ax=ax[1,0])
    ax[1,1] = viz_SZN_dist_circle(SZN, input_token, path, psi_z=None, ax=ax[1,1])

    filepath = path + "-Maze_traj.png"
    plt.savefig(filepath) 
    print(filepath)



@torch.no_grad()
def PlotMazeTrajWindowDist(env, window, agent_traj_encoder, qf1, qf2, alpha, policy, device, Psi, dim_option=2, max_path_length=300, path='./', option_type=None): 
    obs0 = env.reset()
    s0 = torch.tensor(obs0).to(device).float()
    fig, ax = plt.subplots(2,2)
    fig.subplots_adjust(wspace=0.4, hspace=0.4) 
    env.draw(ax[0,0])
    ax[0,0].set_title('State of Traj. in Maze')
    ax[0,1].set_axis_off()
    ax[0,1].set_title('Estimate Value in Z Space')
    if dim_option == 2:
        fig = viz_Value_in_Psi(policy, alpha, qf1, qf2, state=s0, num_samples=10, device=device, path=path, fig=fig)
        
    if Psi is None:
        Psi = Psi_baseline
 
    ax[0,0], FinallDistanceList, All_Repr_obs_list, All_Goal_obs_list, All_trajs_list, FinallDistanceList, ArriveList, All_Cover_list = eval_cover_rate(env, agent_traj_encoder, policy, dim_option, device, ax=ax[0,0], max_path_length=max_path_length, Psi=Psi, option_type=option_type)
    # calculate metrics
    FD = np.array(FinallDistanceList).mean()
    AR = np.array(ArriveList).mean()
    print("FD:", FD, '\n', "AR:", AR)
    # ax[0,0] = plot_trajectories(env, All_trajs_list, fig, ax[0,0])
    ax[0,0] = plot_trajectories_local(env, All_trajs_list, fig, ax[0,0], tile_size)
    ax[1,0] = PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length, is_goal=True, ax=ax[1,0])
    ax[1,1] = viz_dist_circle(window, path, psi_z=None, ax=ax[1,1])
    
    filepath = path + "-Maze_traj.png"
    plt.savefig(filepath) 
    print(filepath)
    
    eval_metrics = calc_eval_metrics(All_Cover_list, is_option_trajectories=True)
    print('[eval_metrics]:', eval_metrics)
    
    return FD, AR, eval_metrics
    



# @torch.no_grad()
# def PlotMazeTraj(env, agent_traj_encoder, policy, device, Psi, dim_option=2, max_path_length=300, path='./', option_type=None): 
#     obs0 = env.reset()
#     fig, ax = plt.subplots(1,2, figsize=(16,8))
#     env.draw(ax[0])
#     ax[0].set_title('State of Traj. in Maze')
        
#     if Psi is None:
#         Psi = Psi_baseline
 
#     ax[0], FinallDistanceList, All_Repr_obs_list, All_Goal_obs_list, All_trajs_list, FinallDistanceList, ArriveList, All_Cover_list = eval_cover_rate(env, agent_traj_encoder, policy, dim_option, device, ax=ax[0], max_path_length=max_path_length, Psi=Psi, option_type=option_type)
#     # calculate metrics
#     FD = np.array(FinallDistanceList).mean()
#     AR = np.array(ArriveList).mean()
#     print("FD:", FD, '\n', "AR:", AR)
#     #ax[0] = plot_trajectories(env, All_trajs_list, fig, ax[0], cover_list=All_Cover_list)
#     #ax[0] = plot_trajectories(env, All_Cover_list, fig, ax[0])
#     ax[0] = plot_trajectories(env, All_trajs_list, fig, ax[0])
#     ax[1] = PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length, is_goal=False, ax=ax[1])
    
#     filepath = path + "-Maze_traj.png"
#     plt.savefig(filepath) 
#     print(filepath)
    
#     eval_metrics = calc_eval_metrics(All_Cover_list, is_option_trajectories=True)
#     print('[eval_metrics]:', eval_metrics)
    
#     return FD, AR, eval_metrics
    
@torch.no_grad()
def PlotMazeTraj(env, agent_traj_encoder, policy, device, Psi, dim_option=2, max_path_length=300, path='./', option_type=None): 
    obs0 = env.reset()
    fig, ax = plt.subplots(1,2, figsize=(16,8))
    
    # --- MODIFICATION START ---
    # if 'minigrid' in env.env_name.lower():
    #     # Render RGB image from MiniGrid
    #     # Depending on your wrapper, you might need env.render() or env.unwrapped.render()
    #     img = env.render(mode='rgb_array') 
    #     if img is None: # fallback if render returns nothing
    #         img = env.unwrapped.render(mode='rgb_array', highlight=False)
            
    #     # Display image (origin='upper' is default for images, matches Minigrid coords)
    #     ax[0].imshow(img)

    #     # Set limits based on grid size
    #     # width = env.unwrapped.width
    #     # height = env.unwrapped.height
        
    #     # ax[0].set_xlim(0, width)
    #     # ax[0].set_ylim(height, 0) # Invert Y to match Minigrid (0,0 is top-left)
    #     # ax[0].set_aspect('equal')
    #     # ax[0].grid(True)

    #     # Ensure axis limits match grid size if you want to overlay scatter plots correctly
    #     # You might need to scale scatter points if img size != grid size
    #     # Usually simpler to just use imshow and let scatter plot interact with pixel coords 
    #     # OR, render tile_size=1 to map 1-to-1
    # else:
    # env.draw(ax[0])
    # --- MODIFICATION END ---
    # --- DRAW ENVIRONMENT ---
    # env.draw(ax[0]) # This calls ax.imshow() for MiniGrid
    ax[0].set_title('State of Traj. in Maze')
    
    if Psi is None:
        Psi = Psi_baseline
 
    # Call the modified eval_cover_rate
    ax[0], FinallDistanceList, All_Repr_obs_list, All_Goal_obs_list, All_trajs_list, FinallDistanceList, ArriveList, All_Cover_list = eval_cover_rate(env, agent_traj_encoder, policy, dim_option, device, ax=ax[0], max_path_length=max_path_length, Psi=Psi, option_type=option_type)
    
    # calculate metrics
    FD = np.array(FinallDistanceList).mean()
    AR = np.array(ArriveList).mean()
    print("FD:", FD, '\n', "AR:", AR)

    # Important: If using Minigrid with imshow, you might need to adjust the trajectory plotter
    # to ensure the points align with the image.
    # Assuming eval_cover_rate returns integer coords (x,y), and you used standard render:
    # Minigrid render usually outputs a high-res image (e.g. 32px per tile).
    # You might need to multiply coords by tile_size.
    
    # FIX for Scatter alignment on Image:
    # ax[0] = plot_trajectories(env, All_trajs_list, fig, ax[0])
    ax[0] = plot_trajectories_local(env, All_trajs_list, fig, ax[0])
    ax[1] = PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length, is_goal=False, ax=ax[1])
    
    filepath = path + "-Maze_traj.png"
    plt.savefig(filepath) 
    print(filepath)
    
    eval_metrics = calc_eval_metrics(All_Cover_list, is_option_trajectories=True)
    print('[eval_metrics]:', eval_metrics)
    
    return FD, AR, eval_metrics
    
    
def UpdateGMM(dists, GMM=None, mix_dist_prob=None, device='cuda'):
    if GMM is None:
        component_distribution = dist.Independent(
            dist.Normal(
                loc=torch.stack([g.mean[0] for g in dists]),
                scale=torch.stack([g.stddev[0] for g in dists])
            ),
            reinterpreted_batch_ndims=1
        )

        if mix_dist_prob is None:
            # 创建均匀的 mixture_distribution
            mixture_distribution = dist.Categorical(
                probs=(torch.ones(len(dists)) / len(dists)).to(device)
            )
        else: 
            mixture_distribution = dist.Categorical(
                probs=mix_dist_prob
            )

        # 组合成一个 MixtureSameFamily 分布
        window_dist = dist.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution
        )

        return window_dist
    
    else:
        component_distribution = GMM.component_distribution
        mixture_distribution = mixture_distribution

        window_dist = dist.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution
        )

        return window_dist


def viz_SZN_dist(SZN, input_token, path):
    dist = SZN(input_token)
    # Data
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x,y)
    from scipy.stats import multivariate_normal
    num = dist.mean.shape[0]
    fig = plt.figure(figsize=(18, 12), facecolor='w')
    for i in range(dist.mean.shape[0]):
        # Multivariate Normal
        mu_x = dist.mean[i][0].detach().cpu().numpy()
        sigma_x = dist.stddev[i][0].detach().cpu().numpy()
        mu_y = dist.mean[i][1].detach().cpu().numpy()
        sigma_y = dist.stddev[i][1].detach().cpu().numpy()
        rv = multivariate_normal([mu_x, mu_y], [[sigma_x, 0], [0, sigma_y]])
        # Probability Density
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        pd = rv.pdf(pos)
        # Plot
        ax = fig.add_subplot(2, num//2, i+1, projection='3d')
        ax.plot_surface(X, Y, pd, cmap='viridis', linewidth=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability Density')
        ax.set_title(label = str(mu_x)[:3] + '-' + str(sigma_x)[:3] + '\n' + str(mu_y)[:3] + '-' + str(sigma_y)[:3])
    plt.savefig(path + '-all' + '.png')
    plt.close()
    

# viz the Regert Map
def viz_Regert_in_Psi(self, state, device='cpu', path='./', ax=None):
    if self.dim_option > 2:
        return
    density = 100
    x = np.linspace(-1, 1, density)
    y = np.linspace(-1, 1, density)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    pos = torch.tensor(pos).to(device)
    pos_flatten = pos.view(-1,2)
    option = pos_flatten
    state_batch = state.repeat(option.shape[0], 1)
    Regret = self.cal_regeret(option, state_batch)[0].view(pos.shape[0], pos.shape[1])
    if ax is None:
        fig = plt.figure(figsize=(18, 12), facecolor='w')
        ax = fig.add_subplot(111, projection='3d')
        
    ax.plot_surface(X, Y, Regret.cpu().numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.view_init(60, 270+20)
    ax.set_xlabel('X')          
    ax.set_ylabel('Y')
    ax.set_zlabel('Regret')
    if ax is None:
        plt.savefig(path + '-Regret' + '.png')
        print('save at: ' + path + '-Regret' + '.png')
        plt.close()
    
    

def PlotGMM(window_dist, psi_z, fig, ax, device, dim=4):
    x_grid = np.linspace(-1, 1, 100)
    y_grid = np.linspace(-1, 1, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    grid_points_tensor = torch.tensor(grid_points).to(device)
    zeros_tensor = torch.zeros(grid_points_tensor.shape[0], dim-2).to(device)
    grid_points_tensor_expanded = torch.cat((grid_points_tensor, zeros_tensor), dim=1)
    log_prob = window_dist.log_prob(grid_points_tensor_expanded).cpu().numpy()
    prob_density = np.exp(log_prob).reshape(X_grid.shape)
    contour = ax.contourf(X_grid, Y_grid, prob_density, levels=20, cmap='viridis')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_ticks([])
    if psi_z is not None:
        ax.scatter(psi_z[:, 0], psi_z[:, 1], alpha=0.5, color='gray', edgecolor='none', marker='o', s=5)
    ax.set_title('GMM Probability Density')
    ax.set_xlabel('Z[0]')
    ax.set_ylabel('Z[1]')

