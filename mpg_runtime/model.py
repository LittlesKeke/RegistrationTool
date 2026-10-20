import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy
from config import *
import numpy as np


class Agent(nn.Module):
    """强化学习智能体类，负责状态嵌入和行动-价值计算"""
    def __init__(self):
        super().__init__()
        self.state_emb = StateEmbedMultiLevelDGCNN()
        self.actor_critic = ActorCriticHead()

    def forward(self, src, tgt):
        """
        前向传播函数。
        src: 输入源数据 tgt: 目标数据
        return:
            state: 状态表示
            action: 动作概率分布输出
            value: 价值输出
            emb_tgt: 目标嵌入表示
        """
        state, emb_tgt = self.state_emb(src, tgt)
        action, value = self.actor_critic(state)

        # Reshape action to B x axis x [step, sign]
        action = (
            action[0].view(-1, 3, 2 * NUM_STEPSIZES + 1),
            action[1].view(-1, 3, 2 * NUM_STEPSIZES + 1)
        )

        value = value.view(-1, 1, 1)

        return state, action, value, emb_tgt

    def prepare_target(self, tgt, detach=False):
        features = self.state_emb.prepare_target(tgt)
        return features.detach() if detach else features

    def forward_with_target(self, src, target_features):
        state, emb_tgt = self.state_emb.forward_with_target(src, target_features)
        action, value = self.actor_critic(state)
        action = (
            action[0].view(-1, 3, 2 * NUM_STEPSIZES + 1),
            action[1].view(-1, 3, 2 * NUM_STEPSIZES + 1),
        )
        return state, action, value.view(-1, 1, 1), emb_tgt


class StateEmbedPointNet(nn.Module):
# PointNet点云数据转化为可供后续处理的tensor
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(IN_CHANNELS, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, FEAT_DIM, 1)

    def forward(self, src, tgt):
        B, N, D = src.shape

        # O=(src,tgt) -> S=[Phi(src), Phi(tgt)]
        emb_src,feature_src = self.embed(src.transpose(2, 1))
        emb_tgt,feature_tgt = self.embed(tgt.transpose(2, 1)) if not (BENCHMARK and len(tgt.shape)!=3) else tgt
        state = torch.cat((emb_src, emb_tgt), dim=-1)
        state = state.view(B, -1)

        return state, feature_tgt

    def embed(self, x):
        B, D, N = x.shape
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.conv3(x2)

        # pooling
        x_pooled = torch.max(x3, 2, keepdim=True)[0]
        return x_pooled.view(B, -1), x1  # B x F

class StateEmbed(nn.Module):
# 在原有PointNet基础上引入DGCNN处理点云局部特征
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(IN_CHANNELS, 64, 1)
        self.dgcnn = DGCNNFeatureExtractor()
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(64 + 256, 512, 1),  # 64 (PointNet局部) + 256 (DGCNN)
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, FEAT_DIM, 1)

        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, src, tgt):
        # 统一处理输入维度 (B, N, 3) -> (B, 3, N)
        if src.dim() == 3:
            src = src.transpose(2, 1)  # (B, 3, N)
        if tgt.dim() == 3 and tgt.shape[1] != 3:
            tgt = tgt.transpose(2, 1) if tgt.shape[1] != 3 else tgt

        B = src.size(0)

        # 提取源点云特征
        emb_src, dgcnn_feat_src = self.process_pointcloud(src)

        # 提取目标点云特征
        if isinstance(tgt, torch.Tensor):
            emb_tgt, dgcnn_feat_tgt = self.process_pointcloud(tgt)
        else:
            # 处理特殊情况下的tgt输入
            emb_tgt = tgt
            dgcnn_feat_tgt = None

        # 构建状态表示
        state = torch.cat((emb_src, emb_tgt), dim=1)
        state = state.view(B, -1)

        # 返回状态和局部特征(用于可视化)
        return state, dgcnn_feat_src

    def process_pointcloud(self, x):
        """处理单个点云，返回全局嵌入和DGCNN特征"""
        # 第一层PointNet特征 (64维局部特征)
        pn_local = F.relu(self.conv1(x))  # (B, 64, N)

        # DGCNN局部特征提取 (256维)
        dgcnn_feat = self.dgcnn(x)  # (B, 256, N)

        # 特征融合
        fused_feat = torch.cat([pn_local, dgcnn_feat], dim=1)  # (B, 64+256, N)
        fused_feat = F.relu(self.fusion_conv(fused_feat))  # (B, 512, N)

        # 继续PointNet处理
        x = F.relu(self.conv2(fused_feat))
        x = self.conv3(x)  # (B, feat_dim, N)

        # 全局特征
        global_feat = self.global_pool(x).view(-1, FEAT_DIM)  # (B, feat_dim)

        return global_feat, fused_feat  # 返回全局特征和融合特征

class StateEmbedWithInteraction(nn.Module):
# 点交互模块双向注意力
    def __init__(self):
        super().__init__()
        self.initial_feature = nn.Sequential(
            nn.Conv1d(IN_CHANNELS, 64, 1),
            nn.ReLU()
        )
        self.dgcnn = DGCNNFeatureExtractor()
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(64 + 256, 512, 1),  # 128 (PointNet局部) + 256 (DGCNN)
            nn.ReLU()
        )
        self.cross_attention = AttentionalPropagation(feature_dim=512,num_heads=4)# 交叉注意力
        self.final_mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, FEAT_DIM, 1) # FEAT_DIM 通常是 1024
        )
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def process_pointcloud(self, x):
        """统一处理单个点云，提取用于交互的初始逐点特征"""
        # PointNet 基础特征
        pn_feat = self.initial_feature(x)
        # DGCNN 局部特征
        dgcnn_feat = self.dgcnn(x)
        # 融合特征
        initial_feat = self.fusion_conv(torch.cat([pn_feat, dgcnn_feat], dim=1))
        return initial_feat

    def forward(self, src, tgt):
        # 统一输入维度: (B, N, 3) -> (B, 3, N)
        src = src.transpose(2, 1) if src.dim() == 3 and src.shape[2] == 3 else src
        tgt = tgt.transpose(2, 1) if tgt.dim() == 3 and tgt.shape[2] == 3 else tgt

        # 1. 提取源和目标的初始逐点特征
        feat_src = self.process_pointcloud(src)  # (B, D, N) D=INTERACTION_FEATURE_DIM
        feat_tgt = self.process_pointcloud(tgt)  # (B, D, M)

        # 2. 进行双向交叉注意力
        # 源点云从目标点云获取信息
        src_enhanced_by_tgt = self.cross_attention(feat_src, feat_tgt, True)
        # 目标点云从源点云获取信息
        tgt_enhanced_by_src = self.cross_attention(feat_tgt, feat_src, True)

        # 3. 组合特征用于最终的全局状态提取
        # 将点云的“原始”特征与其被另一个点云“增强”后的特征进行拼接
        final_feat_src = torch.cat([feat_src, src_enhanced_by_tgt], dim=1)
        final_feat_tgt = torch.cat([feat_tgt, tgt_enhanced_by_src], dim=1)

        # 4. 提取全局特征
        global_feat_src = self.global_pool(self.final_mlp(final_feat_src)).squeeze(-1)
        global_feat_tgt = self.global_pool(self.final_mlp(final_feat_tgt)).squeeze(-1)

        # 5. 构建最终状态表示
        state = torch.cat((global_feat_src, global_feat_tgt), dim=1)
        return state, src_enhanced_by_tgt # 返回状态和增强后的特征以供分析

class StateEmbedPyramidDGCNN(nn.Module):
# 金字塔形DGCNN特征提取器
    def __init__(self, n_pts_per_level=32, neighborhood_factor=1/64):
        super().__init__()
        self.n_pts_per_level = n_pts_per_level
        self.neighborhood_factor = neighborhood_factor
        self.feat_dim = 128
        self.fusion_conv_feat_dim = 512
        self.final_feat_dim = 1024
        # 1. PointNet 基础特征提取
        self.initial_feature = nn.Sequential(
            nn.Conv1d(IN_CHANNELS, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, self.feat_dim, 1)
        )
        # 2. 多级特征提取器
        self.feature_extractor = PyramidFeatureExtractor(
            n_pts_per_level=n_pts_per_level,
            feat_dim=self.feat_dim
        )
        # 3. 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.fusion_conv_feat_dim, 1),
            nn.ReLU()
        )
        # 4. 顶层特征交互模块
        self.cross_attention = AttentionalPropagation(feature_dim=self.fusion_conv_feat_dim, num_heads=4)
        # 5. 最终状态生成模块
        self.final_mlp = nn.Sequential(
            nn.Conv1d(self.fusion_conv_feat_dim * 2, 1024, 1), # original + enhanced
            nn.ReLU(),
            nn.Conv1d(1024, self.final_feat_dim, 1)
        )
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def _create_pyramid_gpu(self, points):
        """
        在GPU上动态创建多级金字塔。
        Input:
            points: pointcloud data, [B, N, 3]
        Return:
            pyramid_levels: a list of tensors for each pyramid level
        """
        B = points.shape[0]
        pyramid_levels = []

        # 初始状态：要处理的patch就是完整的点云
        current_patches = [points]
        num_seeds_last_level = 1

        while True:
            # 将当前层级的patch列表合并为一个大的batch
            # [ (B, N1, 3), (B, N2, 3), ... ] -> (B * num_patches, N_avg, 3)
            patches_tensor = torch.cat(current_patches, dim=0)

            current_patch_size = patches_tensor.shape[1]

            # --- Base Case: 如果patch太小，直接采样并终止 ---
            if current_patch_size <= self.n_pts_per_level:
                idx = farthest_point_sample_gpu(patches_tensor, self.n_pts_per_level)
                final_samples = index_points(patches_tensor, idx)
                pyramid_levels.append(final_samples)
                break

            # --- Iterative Step ---
            # 1. 从当前所有patch中采样种子点
            seed_idx = farthest_point_sample_gpu(patches_tensor, self.n_pts_per_level)
            seeds = index_points(patches_tensor, seed_idx)
            pyramid_levels.append(seeds)

            # 2. 为下一层准备新的、更小的patches
            neighborhood_size = int(current_patch_size * self.neighborhood_factor)
            neighborhood_size = max(neighborhood_size, self.n_pts_per_level) # 确保邻域比采样点多

            # 围绕每个种子点，从其所在的patch中找到一个局部邻域
            knn_idx = knn(patches_tensor.transpose(2, 1), k=neighborhood_size)
            # Gather KNN for seed points only
            neighborhoods = index_points(patches_tensor, torch.gather(knn_idx, 1, seed_idx.unsqueeze(-1).expand(-1, -1, neighborhood_size)))

            # 将下一层的patches准备好，注意要打平batch维度
            current_patches = [neighborhoods.view(-1, neighborhood_size, 3)]
            num_seeds_last_level *= self.n_pts_per_level

        # 调整Pyramid中Tensor的形状以匹配HierarchicalFeatureExtractor的期望
        # (B, 32, 3), (B*32, 32, 3), (B*32*32, 32, 3) ...
        reshaped_pyramid = []
        num_parents = B
        for level in pyramid_levels:
            reshaped_pyramid.append(level.view(num_parents, self.n_pts_per_level, 3))
            num_parents *= self.n_pts_per_level

        return reshaped_pyramid

    def process_pointcloud(self, pcd):
        """为单个点云创建金字塔并提取顶层特征"""
        # 1. 初始特征提取
        initial_features = self.initial_feature(pcd.transpose(2, 1))

        # 2. 动态构建金字塔 提取层次化特征
        pyramid = self._create_pyramid_gpu(pcd)
        top_level_features = self.feature_extractor(pyramid)

        features = self.fusion_conv(torch.cat([initial_features, top_level_features], dim=2))
        return features

    def forward(self, src, tgt):
        # 确保输入维度为 (B, N, 3)
        src = src if src.dim() == 3 and src.shape[2] == 3 else src.transpose(1, 2)
        tgt = tgt if tgt.dim() == 3 and tgt.shape[2] == 3 else tgt.transpose(1, 2)

        # 1. 为源和目标点云提取顶层多尺度特征
        # 输出 shape: (B, D, 32)
        feat_src = self.process_pointcloud(src)
        feat_tgt = self.process_pointcloud(tgt)

        # 2. 在顶层特征上进行双向交叉注意力
        src_enhanced_by_tgt = self.cross_attention(feat_src, feat_tgt, True)
        tgt_enhanced_by_src = self.cross_attention(feat_tgt, feat_src, True)

        # 3. 组合特征
        final_feat_src = torch.cat([feat_src, src_enhanced_by_tgt], dim=1)
        final_feat_tgt = torch.cat([feat_tgt, tgt_enhanced_by_src], dim=1)

        # 4. 提取全局特征
        global_feat_src = self.global_pool(self.final_mlp(final_feat_src)).squeeze(-1)
        global_feat_tgt = self.global_pool(self.final_mlp(final_feat_tgt)).squeeze(-1)

        # 5. 构建最终状态表示
        state = torch.cat((global_feat_src, global_feat_tgt), dim=1)

        # 返回状态和目标点云的顶层特征（可用于后续分析或辅助任务）
        return state, final_feat_tgt

class StateEmbedMultiLevelDGCNN(nn.Module):
# 多级DGCNN特征提取器
    def __init__(self, n_pts_per_level=32, neighborhood_factor=1/2):
        super().__init__()
        self.n_pts_per_level = n_pts_per_level
        self.neighborhood_factor = neighborhood_factor
        self.feat_dim = 128
        self.fusion_conv_feat_dim = 512
        self.final_feat_dim = 1024
        # 1. PointNet 基础特征提取
        self.initial_feature = nn.Sequential(
            nn.Conv1d(IN_CHANNELS, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, self.feat_dim, 1)
        )
        # 2. 多级特征提取器
        self.feature_extractor = HierarchicalFeatureExtractor(
            n_pts_per_level=n_pts_per_level,
            feat_dim=self.feat_dim
        )
        # 3. 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.fusion_conv_feat_dim, 1),
            nn.ReLU()
        )
        # 4. 顶层特征交互模块
        self.cross_attention = AttentionalPropagation(feature_dim=self.fusion_conv_feat_dim, num_heads=4)
        # 5. 最终状态生成模块
        self.final_mlp = nn.Sequential(
            nn.Conv1d(self.fusion_conv_feat_dim * 2, 1024, 1), # original + enhanced
            nn.ReLU(),
            nn.Conv1d(1024, self.final_feat_dim, 1)
        )
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def _create_pyramid_gpu(self, points, seed_idx):
        """
        在GPU上动态创建多级金字塔。
        Input:
            points: pointcloud data, [B, N, 3]
            seed_idx: 种子点索引, [B, 32]
        Return:
            pyramid_levels: 多级尺度的点云金字塔 多个[B*32, 32, 3]
        """
        B, N, _ = points.shape
        pyramid_levels = []

        # 1. 一次性采样出所有层级将使用的种子点
        seeds = index_points(points, seed_idx)  # Shape: (B, 32, 3)

        # 计算所有种子点的k近邻
        neighborhood_size = N
        knn_idx = knn_points(seeds, points, k=neighborhood_size)  # Shape: (B, 32, N)

        while True:
            # --- Base Case: 如果patch太小，直接采样并终止 ---
            if neighborhood_size < self.n_pts_per_level:
                break

            # --- Iterative Step ---
            # 获取当前种子点的邻域
            neighborhoods = index_points(points, knn_idx)  # 获取所有种子点的邻域，Shape: (B, 32, neighborhood_size, 3)

            # Reshape for batch FPS: (B*32, neighborhood_size, 3)
            patches_flat = neighborhoods.view(B * self.n_pts_per_level, neighborhood_size, 3)

            # 从每个邻域中再次采样，得到统一大小的patch
            patch_idx = farthest_point_sample_gpu(patches_flat, self.n_pts_per_level)
            sampled_patches = index_points(patches_flat, patch_idx)
            pyramid_levels.append(sampled_patches)

            # 缩小下一层的邻域范围
            neighborhood_size = int(neighborhood_size * self.neighborhood_factor)

            # 更新knn_idx以切片获取新的邻域
            knn_idx = knn_idx[:, :, :neighborhood_size]  # 切片获取新的邻域范围

        return pyramid_levels

    def process_pointcloud(self, pcd):
        """为单个点云创建金字塔并提取顶层特征"""
        # 1. 初始特征提取
        seed_idx = farthest_point_sample_gpu(pcd, self.n_pts_per_level)
        pcd_features = self.initial_feature(pcd.transpose(2, 1))
        #seed_features = self.initial_feature(index_points(pcd, seed_idx).transpose(2, 1))

        # 2. 动态构建金字塔 提取层次化特征
        pyramid = self._create_pyramid_gpu(pcd, seed_idx)
        top_level_features = self.feature_extractor(pyramid)

        features = self.fusion_conv(torch.cat([pcd_features, top_level_features], dim=2))
        return features

    def prepare_target(self, tgt):
        tgt = tgt if tgt.dim() == 3 and tgt.shape[2] == 3 else tgt.transpose(1, 2)
        return self.process_pointcloud(tgt)

    def forward_with_target(self, src, feat_tgt):
        # 确保输入维度为 (B, N, 3)
        src = src if src.dim() == 3 and src.shape[2] == 3 else src.transpose(1, 2)

        # 1. 为源和目标点云提取顶层多尺度特征
        # 输出 shape: (B, D, 32)
        feat_src = self.process_pointcloud(src)

        # 2. 在顶层特征上进行双向交叉注意力
        src_enhanced_by_tgt = self.cross_attention(feat_src, feat_tgt, True)
        tgt_enhanced_by_src = self.cross_attention(feat_tgt, feat_src, True)

        # 3. 组合特征
        final_feat_src = torch.cat([feat_src, src_enhanced_by_tgt], dim=1)
        final_feat_tgt = torch.cat([feat_tgt, tgt_enhanced_by_src], dim=1)

        # 4. 提取全局特征
        global_feat_src = self.global_pool(self.final_mlp(final_feat_src)).squeeze(-1)
        global_feat_tgt = self.global_pool(self.final_mlp(final_feat_tgt)).squeeze(-1)

        # 5. 构建最终状态表示
        state = torch.cat((global_feat_src, global_feat_tgt), dim=1)

        # 返回状态和目标点云的顶层特征（可用于后续分析或辅助任务）
        return state, final_feat_tgt

    def forward(self, src, tgt):
        return self.forward_with_target(src, self.prepare_target(tgt))

class PyramidFeatureExtractor(nn.Module):
    """从金字塔中由粗到精提取特征"""
    def __init__(self, n_pts_per_level, feat_dim):
        super().__init__()
        self.n_pts_per_level = n_pts_per_level

        # DGCNN模块可以共享，也可以为不同层级创建不同的实例
        self.dgcnn = DGCNNFeatureExtractor(k=n_pts_per_level, feat_dim=feat_dim)

        # MLP用于融合父节点特征和聚合后的子节点特征
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(feat_dim * 2, feat_dim, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dim, feat_dim, 1)
        )

    def forward(self, pyramid):
        B = pyramid[0].shape[0] # 原始Batch Size
        child_features = None

        # 逆序遍历金字塔 (from fine to coarse)
        for level_idx in range(len(pyramid) - 1, -1, -1):
            level_points = pyramid[level_idx] # Shape: (B * 32^idx, 32, 3)
            num_patches = level_points.shape[0]

            # 1. 提取当前层的基础特征 (B*..., 3, 32) -> (B*..., D, 32)
            current_features = self.dgcnn(level_points.transpose(1, 2))

            # 2. 如果不是最底层，则融合来自子层的特征
            if child_features is not None:
                # 聚合子特征 (Max Pooling)
                # (B*...*32, D, 32) -> (B*...*32, D)
                agg_child_features = child_features.max(dim=-1)[0]

                # Reshape以匹配当前层的patch数量
                # (B*...*32, D) -> (B*..., 32, D) -> (B*..., D, 32)
                agg_child_features = agg_child_features.view(
                    num_patches, self.n_pts_per_level, -1
                ).transpose(1, 2)

                # 拼接并用MLP融合
                fused_input = torch.cat([current_features, agg_child_features], dim=1)
                current_features = self.fusion_mlp(fused_input)

            # 保存当前结果，作为下一轮（上一层）的子特征
            child_features = current_features

        # 返回金字塔最顶层(最粗糙层)的特征
        # Shape: (B, D, 32)
        return child_features

class HierarchicalFeatureExtractor(nn.Module):
    """从多级特征中由粗到精提取特征"""
    def __init__(self, n_pts_per_level, feat_dim):
        super().__init__()
        self.n_pts_per_level = n_pts_per_level
        self.feat_dim = feat_dim

        # DGCNN模块可以共享，也可以为不同层级创建不同的实例
        self.dgcnn = DGCNNFeatureExtractor(k=n_pts_per_level, feat_dim=feat_dim)

        # MLP用于融合父节点特征和聚合后的子节点特征
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(feat_dim * 2, feat_dim, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dim, feat_dim, 1)
        )

    def forward(self, pyramid):
        B, N, _ = pyramid[0].shape # e.g., B * 32
        fused_features = None

        # 逆序遍历金字塔 (from fine scale to coarse scale)
        for level_idx in range(len(pyramid) - 1, -1, -1):
            level_points = pyramid[level_idx] # Shape: (B * 32, 32, 3)

            # 1. 提取当前尺度的基础DGCNN特征
            # (B*32, 3, 32) -> (B*32, D, 32)
            current_scale_features = self.dgcnn(level_points.transpose(1, 2))

            # 2. 如果不是最精细的尺度，则融合来自更精细尺度的信息
            if fused_features is not None:
                # 聚合来自更精细尺度的特征 (Max Pooling)
                # (B*32, D, 32) -> (B*32, D)
                agg_finer_scale_features = fused_features.max(dim=-1)[0]

                # 将聚合后的特征扩展，以便与当前尺度的逐点特征拼接
                # (B*32, D) -> (B*32, D, 1) -> (B*32, D, 32)
                agg_finer_scale_features_expanded = agg_finer_scale_features.unsqueeze(-1).expand_as(current_scale_features)

                # 拼接并用MLP融合
                # Input shape to fusion_mlp: (B*32, 2*D, 32)
                fusion_input = torch.cat([current_scale_features, agg_finer_scale_features_expanded], dim=1)
                current_scale_features = self.fusion_mlp(fusion_input)

            # 保存当前融合结果，作为下一轮（更粗糙尺度）的输入
            fused_features = current_scale_features

        # 最终返回的是融合了所有尺度信息后的特征
        # 此时 fused_features 是在最粗糙尺度(最大邻域)上计算，并融合了所有更精细尺度信息的特征
        # Shape: (B, D, )
        return fused_features.reshape(int(B/N), self.feat_dim, -1)

class DGCNNFeatureExtractor(nn.Module):
    """DGCNN特征提取器(简化版)，仅输出局部特征"""
    def __init__(self, k=32, feat_dim=256):
        super().__init__()
        self.k = k

        # 图特征构建层
        self.graph_conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1),
            nn.ReLU()
        )

        self.graph_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU()
        )

        self.graph_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU()
        )

        # 特征聚合层
        self.conv_out = nn.Sequential(
            nn.Conv1d(64 + 128 + 256, feat_dim, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        """x: (B, C, N)"""
        B, C, N = x.shape

        # 构建多级图特征
        feat1 = self.build_graph_feature(x, self.k)
        feat1 = self.graph_conv1(feat1)
        feat1 = feat1.max(dim=-1)[0]  # (B, 64, N)

        feat2 = self.build_graph_feature(feat1, self.k)
        feat2 = self.graph_conv2(feat2)
        feat2 = feat2.max(dim=-1)[0]  # (B, 128, N)

        feat3 = self.build_graph_feature(feat2, self.k)
        feat3 = self.graph_conv3(feat3)
        feat3 = feat3.max(dim=-1)[0]  # (B, 256, N)

        # 聚合多级特征
        out = torch.cat([feat1, feat2, feat3], dim=1)  # (B, 64+128+256, N)
        out = self.conv_out(out)  # (B, feat_dim, N)

        return out

    def build_graph_feature(self, x, k):
        """构建图特征: (B, C, N) -> (B, 2*C, N, k) -> (B, 6, N, k)?"""
        B, C, N = x.shape

        # 寻找k最近邻
        idx = knn(x, k)

        # 中心点特征扩展 (B, C, N) -> (B, C, N, K)
        central = x.unsqueeze(3).repeat(1, 1, 1, k)

        # 邻居特征收集 (B, C, N, K)
        knn_idx = idx.unsqueeze(1).expand(-1, C, -1, -1)
        knn_feat = torch.gather(x.unsqueeze(3).expand(-1, -1, -1, k), 2, knn_idx)

        # 构建边特征：拼接[中心点, 邻居-中心点]
        edge_feat = torch.cat([central, knn_feat - central], dim=1)

        return edge_feat

def knn(x, k):
    """
    计算每个点的k个最近邻。
    x: 输入点云 (B, 3, N)
    k: 每个点的最近邻数
    return: 每个点的k个最近邻索引 (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx

def knn_points(query, context, k):
    """
    Find k-nearest neighbors in context for each point in query.
    Input:
        query: query points, [B, M, C]
        context: context points, [B, N, C]
        k: int
    Return:
        idx: k-nearest neighbor index in context, [B, M, k]
    """
    dist = torch.cdist(query, context) # [B, M, N]
    _, idx = dist.topk(k=k, dim=-1, largest=False)
    return idx

def farthest_point_sample_gpu(xyz, npoint):
    """
    GPU-friendly Farthest Point Sampling.
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Index points using indices.
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron with 1D Convolutions """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                # Using BatchNorm instead of InstanceNorm for potential batch-level stats
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

# --- Core Attention Mechanisms ---
def attention(query, key, value):
    """ Scaled Dot-Product Attention """
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
       # if isself:
           # origin = origin.view(batch_dim, self.dim, self.num_heads, -1)

        x, _ = attention(query, key, value)

        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp_oringin = MLP([feature_dim*2, feature_dim*2, feature_dim])
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)
    def forward(self, x, source, cross):
        if cross:
            message = self.attn(x, source, source)
            out = self.mlp_oringin(torch.cat([x, message], dim=1))

        else:
            out = self.attn(x, source, source)
        return out

class Propagate(nn.Module):
    def __init__(self, in_channel, emb_dims):
        super(Propagate, self).__init__()
        self.conv2d = Conv2DBlock((in_channel, emb_dims, emb_dims), 1)
        self.conv1d = Conv1DBlock((emb_dims, emb_dims), 1)

    def forward(self, x, idx):
        batch_idx = np.arange(x.size(0)).reshape(x.size(0), 1, 1)
        nn_feat = x[batch_idx, :, idx].permute(0, 3, 1, 2)
        x = nn_feat - x.unsqueeze(-1)
        x = self.conv2d(x)
        x = x.max(-1)[0]
        x = self.conv1d(x)
        return x

class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv1DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i + 1], ksize))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x

class Conv2DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv2DBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv2DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.conv.append(Conv2DBNReLU(channels[i], channels[i + 1], ksize))
        self.conv.append(nn.Conv2d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class ActorCriticHead(nn.Module):
    '''演员-评论家结构：根据状态预测动作'''
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()

        # 处理旋转的emb_r
        self.emb_r = nn.Sequential(
            nn.Linear(STATE_DIM, HEAD_DIM*2),
            self.activation,
            #nn.Dropout(0.5),
            nn.Linear(HEAD_DIM*2, HEAD_DIM),
            self.activation
        )
        # 用于旋转的动作概率分布 共6种旋转动作*6种步长+无操作
        self.action_r = nn.Linear(HEAD_DIM, NUM_ACTIONS*NUM_STEPSIZES+NUM_NOPS)

        # 处理平移的emb_t
        self.emb_t = nn.Sequential(
            nn.Linear(STATE_DIM, HEAD_DIM*2),
            self.activation,
            #nn.Dropout(0.5),
            nn.Linear(HEAD_DIM*2, HEAD_DIM),
            self.activation
        )
        # 用于平移的动作概率分布 共6种平移动作*6种步长+无操作
        self.action_t = nn.Linear(HEAD_DIM, NUM_ACTIONS*NUM_STEPSIZES+NUM_NOPS)

        # 预测价值输出
        self.emb_v = nn.Sequential(
            nn.Linear(HEAD_DIM * 2, HEAD_DIM),
            self.activation,
            #nn.Dropout(0.5),
        )
        self.value = nn.Linear(HEAD_DIM, 1)

    def forward(self, state):
        emb_t = self.emb_t(state)
        emb_r = self.emb_r(state)

        # 输出平移和旋转动作
        action_logits_t = self.action_t(emb_t)
        action_logits_r = self.action_r(emb_r)

        # 输出值
        state_action = torch.cat([emb_t, emb_r], dim=1)
        emb_v = self.emb_v(state_action)
        value = self.value(emb_v)

        return [action_logits_t, action_logits_r], value



# 动作辅助函数
def action_from_logits(logits, deterministic=True):
    # 将原始的动作logits转换为动作分布（使用 Categorical），
    # 然后从这些分布中采样动作（可以是确定性或随机的）
    distributions = _get_distributions(*logits)
    actions = _get_actions(*(distributions + (deterministic,)))

    return torch.stack(actions).transpose(1, 0)


def action_stats(logits, action):
    # 计算所选动作的对数概率（log-probabilities）和熵（entropies）
    distributions = _get_distributions(*logits)
    logprobs, entropies = _get_logprob_entropy(*(distributions + (action[:, 0], action[:, 1])))

    return torch.stack(logprobs).transpose(1, 0), torch.stack(entropies).transpose(1, 0)


def _get_distributions(action_logits_t, action_logits_r):
    # 基于提供的logits为旋转和平移动作创建 Categorical 分布
    distribution_t = Categorical(logits=action_logits_t)
    distribution_r = Categorical(logits=action_logits_r)

    return distribution_t, distribution_r


def _get_actions(distribution_t, distribution_r, deterministic=True):
    # 从分布中采样动作，或选择具有最大概率的动作
    if deterministic:
        action_t = torch.argmax(distribution_t.probs, dim=-1)
        action_r = torch.argmax(distribution_r.probs, dim=-1)
    else:
        action_t = distribution_t.sample()
        action_r = distribution_r.sample()
    return action_t, action_r


def _get_logprob_entropy(distribution_t, distribution_r, action_t, action_r):
    # 计算所选动作的对数概率和熵
    logprob_t = distribution_t.log_prob(action_t)
    logprob_r = distribution_r.log_prob(action_r)

    entropy_t = distribution_t.entropy()
    entropy_r = distribution_r.entropy()

    return [logprob_t, logprob_r], [entropy_t, entropy_r]


# 模型辅助函数
def load(model, path):
    #infos = torch.load(path)
    infos = torch.load(path, map_location=torch.device(DEVICE), weights_only=True)
    model.load_state_dict(infos['model_state_dict'])
    return infos


def save(model, path, infos={}):
    infos['model_state_dict'] = model.state_dict()
    torch.save(infos, path)

if __name__ == '__main__':
    # 调试
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr
    # 测试模型
    src = torch.randn(2, 1024, 3)  # 随机生成源点云
    tgt = torch.randn(2, 2048, 3)  # 随机生成目标点云
    agent = Agent()  # 创建智能体实例
    # 测试前向传播，同时记录时间
    start_time = time.time()
    state, action, value, emb_tgt = agent(src, tgt)  # 前向传播
    end_time = time.time()
