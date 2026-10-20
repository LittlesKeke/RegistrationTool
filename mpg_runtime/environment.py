import torch
import numpy as np
import transformations as tra
from config import DEVICE


STEPSIZES = [0.0003, 0.003, 0.01, 0.03, 0.09, 0.27]
ALL_STEPS = torch.FloatTensor(STEPSIZES[::-1] + [0] + STEPSIZES).to(DEVICE)
POS_STEPS = torch.FloatTensor([0] + STEPSIZES).to(DEVICE)
NUM_STEPS = len(STEPSIZES)

"""
封装注册环境行为，
即初始化、
更新给定动作的状态、
计算当前状态下动作的奖励
"""


def init(data):
    """
    Get the initial observation, the ground-truth pose for the expert and initialize the agent's accumulator (identity).
    """
    # observation
    pcd_source, pcd_target = data['points_src'][..., :3].to(DEVICE), data['points_ref'][..., :3].to(DEVICE)
    B = pcd_source.shape[0]

    # GT (for expert)
    pose_target = torch.eye(4, device=DEVICE).repeat(B, 1, 1)
    if 'transform_gt' in data:
        pose_target[:, :3, :] = data['transform_gt'][:, :3, :]

    # initial estimates (identity, for student)
    pose_source = torch.eye(4, device=DEVICE).repeat(B, 1, 1)

    return pcd_source, pcd_target, pose_source, pose_target


def _action_to_step(axis_actions):
    """
    将动作ID转换为符号和步长
    用于确定动作是向前还是向后移动，以及移动的幅度。
    """
    step = ALL_STEPS[axis_actions]
    sign = ((axis_actions - NUM_STEPS >= 0).float() - 0.5) * 2
    return sign, step


def step(source, actions, pose_source, disentangled=True):
    """
    根据给定的动作更新状态
    """
    actions_t, actions_r = actions[:, 0], actions[:, 1]
    indices = torch.arange(source.shape[0]).unsqueeze(0)

    # 动作转化为变换
    steps_t = torch.zeros((actions.shape[0], 3), device=DEVICE)
    steps_r = torch.zeros((actions.shape[0], 3), device=DEVICE)
    for i in range(3):
        sign, step = _action_to_step(actions_t[:, i])
        steps_t[indices, i] = step * sign  # 平移

        sign, step = _action_to_step(actions_r[:, i])
        steps_r[indices, i] = step * sign # 旋转

    # 变换累加器更新
    if disentangled:  # eq. 7 in paper
        pose_source[:, :3, :3] = tra.euler_angles_to_matrix(steps_r, 'XYZ') @ pose_source[:, :3, :3]
        pose_source[:, :3, 3] += steps_t
    else:  # concatenate 4x4 matrices (eq. 5 in paper)
        pose_update = torch.eye(4, device=DEVICE).repeat(pose_source.shape[0], 1, 1)
        pose_update[:, :3, :3] = tra.euler_angles_to_matrix(steps_r, 'XYZ')
        pose_update[:, :3, 3] = steps_t

        pose_source = pose_update @ pose_source

    # update source with the accumulated transformation
    new_source = tra.apply_trafo(source, pose_source, disentangled)

    return new_source, pose_source


def expert(pose_source, targets, mode='steady'):
    """
    获得当前状态下的专家动作
    """
    # compute delta, eq. 10 in paper
    delta_t = targets[:, :3, 3] - pose_source[:, :3, 3]
    delta_R = targets[:, :3, :3] @ pose_source[:, :3, :3].transpose(2, 1)  # global accumulator
    delta_r = tra.matrix_to_euler_angles(delta_R, 'XYZ')

    def _get_axis_action(axis_delta, mode='steady'):
        lower_idx = (torch.bucketize(torch.abs(axis_delta), POS_STEPS) - 1).clamp(0, NUM_STEPS)
        if mode == 'steady':
            nearest_idx = lower_idx
        elif mode == 'greedy':
            upper_idx = (lower_idx + 1).clamp(0, NUM_STEPS)
            lower_dist = torch.abs(torch.abs(axis_delta) - POS_STEPS[lower_idx])
            upper_dist = torch.abs(POS_STEPS[upper_idx] - torch.abs(axis_delta))
            nearest_idx = torch.where(lower_dist < upper_dist, lower_idx, upper_idx)
        else:
            raise ValueError
        # -- step idx to action
        axis_action = nearest_idx  # [0, num_steps] -- 0 = NOP
        axis_action[axis_delta < 0] *= -1  # [-num_steps, num_steps + 1] -- 0 = NOP
        axis_action += NUM_STEPS  # [0, 2 * num_steps + 1 -- num_steps = NOP

        return axis_action[:, None, None]

    # find bounds per axis s.t. b- <= d <= b+
    action_t = torch.cat([_get_axis_action(delta_t[:, i], mode) for i in range(3)], dim=2)
    action_r = torch.cat([_get_axis_action(delta_r[:, i], mode) for i in range(3)], dim=2)
    action = torch.cat([action_t, action_r], dim=1)

    return action


# def reward_step(current_pcd_source, pcd_target, prev_chamfer_dist=None, alpha=0.3):
#     """
#     只基于CD距离计算密集步骤奖励
#     """
#     # 目标点云和当前点云之间的最小距离
#     dist = torch.min(tra.square_distance(current_pcd_source, pcd_target), dim=-1)[0]
#     chamfer_dist = torch.mean(dist, dim=1).view(-1, 1, 1)

#     goal_reward = torch.exp(- chamfer_dist ** alpha)

#     # 如果是第一次计算，没有上一次的 Chamfer 距离
#     if prev_chamfer_dist is None:
#         return torch.zeros_like(chamfer_dist), chamfer_dist

#     better = (chamfer_dist < prev_chamfer_dist).float() * 0.5
#     same = (chamfer_dist == prev_chamfer_dist).float() * 0.1
#     worse = (chamfer_dist > prev_chamfer_dist).float() * 0.6

#     reward = better - worse - same
#     return reward + goal_reward, chamfer_dist


def reward_step(current_pcd_source, pcd_target, pose_source, pose_target, prev_chamfer_dist=None):
    """
    稠密奖励+稀疏目标奖励
    """
    # 1. 计算当前CD和姿态误差
    dist = torch.min(tra.square_distance(current_pcd_source, pcd_target), dim=-1)[0]
    chamfer_dist = torch.mean(dist, dim=1).view(-1, 1, 1)

    #t_error = torch.norm(pose_source[:, :3, 3] - pose_target[:, :3, 3], dim=1, keepdim=True).unsqueeze(-1)

    delta_R = pose_target[:, :3, :3] @ pose_source[:, :3, :3].transpose(2, 1)
    trace = torch.einsum('bii->b', delta_R)
    cos_theta = (trace - 1) / 2
    r_error_rad = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1.e-6))
    r_error = r_error_rad.unsqueeze(-1).unsqueeze(-1)

    # --- 奖励计算 ---
    reward = torch.zeros_like(chamfer_dist)

    # 2. 主要奖励：归一化的CD改进
    if prev_chamfer_dist is not None:
        improvement = prev_chamfer_dist - chamfer_dist # 正值表示改进
        # 对改进给予与改进幅度相关的奖励，对退步给予更大惩罚
        delta_reward = torch.tanh(improvement * 50.0) # 使用tanh将奖励/惩罚缩放到[-1, 1]
        reward += 1.0 * delta_reward # 增量奖励的权重

    # 3. 辅助奖励：对最终误差的惩罚
    # 使用log来塑造奖励，使得误差小时惩罚梯度更大，鼓励精细调整。
    # 权重应该比改进奖励小，作为一种“塑形”而非主导。
    POSE_WEIGHT = 1.0
    CD_WEIGHT = 0.5

    # 使用负对数作为惩罚项，误差越小，惩罚越小（奖励越大）
    # log(x+eps)在x->0时，梯度很大，适合精调
    cd_penalty = -torch.log(chamfer_dist + 1e-6)
    pose_penalty = -torch.log(r_error + 1e-6)

    reward += CD_WEIGHT * cd_penalty + POSE_WEIGHT * pose_penalty

    return reward, chamfer_dist
