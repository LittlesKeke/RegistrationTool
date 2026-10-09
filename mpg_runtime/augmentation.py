import math
from typing import Dict, List
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from scipy.spatial import KDTree
import torch
import open3d as o3d
import transforms3d as t3d
from pointEmbed import knn_point
from config import TRANSLATE, ROTATE, NUM_POINTS_IN_PATCH
# Adapted from RPM-Net (Yew et al., 2020): https://github.com/yewzijian/RPMNet


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


class SplitSourceRef:
    """Clones the point cloud into separate source and reference point clouds"""
    def __call__(self, sample: Dict):
        if 'points' in sample:
            sample['points_raw'] = sample.pop('points')
        else:
            assert 'points_raw' in sample
        if isinstance(sample['points_raw'], torch.Tensor):
            sample['points_src'] = sample['points_raw'].detach()
            sample['points_ref'] = sample['points_raw'].detach()
        else:  # is numpy
            sample['points_src'] = sample['points_raw'].copy()
            sample['points_ref'] = sample['points_raw'].copy()

        return sample


class LocalPointCloudSampler:
    """用于生成部分-全局对应数据
    从源点云中随机选择一个局部点云，且该点云占源点云的百分之k，局部性优先"""
    def __init__(self, k: float):
        """
        Args:
            k (float): 需要保留的点云占比，范围是(0, 1]
        """
        assert 0 < k <= 1, "k 应为 (0, 1] 之间的数值"
        self.k = k

    def __call__(self, sample: Dict) -> Dict:
        """
        1. 当k=1时跳过
        2. 始终保证最终点云由离中心最近的k%个点组成
        """
        if 'points_src' not in sample:
            raise KeyError("输入样本需包含 'points_src' 键")

        if self.k == 1:
            return sample
        points_src = sample['points_src']
        num_points = points_src.shape[0]
        num_local_points = int(self.k * num_points)

        # 随机选择中心点
        center_idx = np.random.randint(0, num_points)
        center_point = points_src[center_idx]

        # 计算所有点到中心的距离
        distances = np.linalg.norm(points_src[:, :3] - center_point[:3], axis=1)

        # 按距离排序获取最近的k%个点（保证局部性）
        sorted_indices = np.argsort(distances)
        local_indices = sorted_indices[:num_local_points]

        # 更新采样结果
        sample['points_src'] = points_src[local_indices]

        return sample


class RandomCrop:
    """
    用于生成部分-部分对应数据
    随机裁剪源、目标点云，并根据需要保证重叠率

    """
    def __init__(self, p_keep: List = None, SameDirection: bool = False, target_overlap_ratio:float = 0.732):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)
        self.SameDirection = SameDirection
        self.target_overlap_ratio = target_overlap_ratio

    @staticmethod
    def crop(points, p_keep, rand_xyz=None):
        if rand_xyz is None:
            rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def calculate_overlap(self, points_src, points_ref, transform_gt, threshold=0.05):
        """计算两个点云之间的重叠率"""
        # 将源点云通过GT变换对齐到目标点云
        if transform_gt is not None:
            points_src_transformed = points_src[:, :3] @ transform_gt[:3, :3].T + transform_gt[:3, 3]
        else:
            points_src_transformed = points_src[:, :3]

        # 使用KDTree进行最近邻搜索
        tree_ref = KDTree(points_ref[:, :3])
        distances, _ = tree_ref.query(points_src_transformed, k=1)

        # 统计重叠点（距离小于阈值）
        overlap_mask = distances < threshold
        overlap_ratio = np.sum(overlap_mask) / len(points_src)

        return overlap_ratio, overlap_mask

    def calculate_overlap(self, points_src, points_ref, transform_gt, threshold=0.0001):
        """计算两个点云之间的重叠率"""
        # 将源点云通过GT变换对齐到目标点云
        if transform_gt is not None:
            points_src_transformed = points_src[:, :3] @ transform_gt[:3, :3].T + transform_gt[:3, 3]
        else:
            points_src_transformed = points_src[:, :3]

        # 使用KDTree进行最近邻搜索
        tree_ref = KDTree(points_ref[:, :3])
        distances, _ = tree_ref.query(points_src_transformed, k=1)

        # 统计重叠点（距离小于阈值）
        overlap_mask = distances < threshold
        overlap_ratio = np.sum(overlap_mask) / len(points_src)

        return overlap_ratio, overlap_mask

    def adjust_overlap(self, points_src_cropped, points_ref_cropped, points_ref_original, transform_gt, target_ratio):
        """调整点云重叠率，确保符合目标值"""
        # 计算当前重叠情况
        overlap_ratio, overlap_mask = self.calculate_overlap(points_src_cropped, points_ref_cropped, transform_gt)

        # 如果已经符合要求，直接返回
        if abs(overlap_ratio - target_ratio) < 0.01:
            return points_src_cropped, points_ref_cropped, overlap_ratio

        # 计算需要调整的点数
        current_overlap_points = np.sum(overlap_mask)
        target_overlap_points = int(target_ratio * len(points_src_cropped))
        points_to_adjust = abs(current_overlap_points - target_overlap_points)

        if overlap_ratio > target_ratio:
            # 重叠率过高，需要减少重叠点
            # 找出重叠的点在目标点云中的索引
            overlap_indices = np.where(overlap_mask)[0]

            if len(overlap_indices) > points_to_adjust:
                # 随机选择部分重叠点替换为非重叠点
                indices_to_remove = np.random.choice(
                    overlap_indices,
                    size=points_to_adjust,
                    replace=False
                )

                # 找出目标点云中未被选中的点（包括原始点云中未被裁剪的点）
                all_ref_indices = set(range(len(points_ref_original)))
                current_ref_indices = set(range(len(points_ref_cropped)))
                available_indices = list(all_ref_indices - current_ref_indices)

                if len(available_indices) > 0:
                    # 随机选择新的点替换重叠点
                    new_indices = np.random.choice(
                        available_indices,
                        size=min(points_to_adjust, len(available_indices)),
                        replace=False
                    )

                    # 创建新的目标点云
                    new_ref_points = points_ref_cropped.copy()

                    # 替换点：移除重叠点，添加新点
                    keep_mask = np.ones(len(new_ref_points), dtype=bool)
                    keep_mask[indices_to_remove] = False
                    new_ref_points = new_ref_points[keep_mask]

                    # 添加新点
                    if len(new_indices) > 0:
                        new_points = points_ref_original[new_indices]
                        new_ref_points = np.vstack([new_ref_points, new_points])

                    # 重新计算重叠率
                    final_overlap_ratio, _ = self.calculate_overlap(
                        points_src_cropped, new_ref_points, transform_gt
                    )

                    return points_src_cropped, new_ref_points, final_overlap_ratio

        else:
            # 重叠率过低，需要增加重叠点
            # 找出非重叠的点在源点云中的索引
            non_overlap_indices = np.where(~overlap_mask)[0]

            if len(non_overlap_indices) > 0:
                # 找出目标点云中可能产生重叠的点
                # 将源点云变换到目标坐标系
                points_src_transformed = points_src_cropped[:, :3] @ transform_gt[:3, :3].T + transform_gt[:3, 3]

                # 在原始目标点云中寻找与源点云非重叠点最近的点
                tree_original_ref = KDTree(points_ref_original[:, :3])

                # 为每个非重叠点找到最近邻
                distances, nearest_indices = tree_original_ref.query(
                    points_src_transformed[non_overlap_indices],
                    k=1
                )

                # 筛选出距离较近的点（可能产生重叠）
                close_mask = distances < 0.1  # 稍微放宽阈值
                close_non_overlap_indices = non_overlap_indices[close_mask]
                close_ref_indices = nearest_indices[close_mask]

                if len(close_ref_indices) > points_to_adjust:
                    # 随机选择要添加的点
                    selected_indices = np.random.choice(
                        range(len(close_ref_indices)),
                        size=points_to_adjust,
                        replace=False
                    )

                    # 获取要添加的目标点云索引
                    ref_indices_to_add = close_ref_indices[selected_indices]

                    # 确保这些点不在当前目标点云中
                    current_ref_points_set = set([tuple(point) for point in points_ref_cropped[:, :3]])
                    new_points = []

                    for idx in ref_indices_to_add:
                        point = points_ref_original[idx]
                        if tuple(point[:3]) not in current_ref_points_set:
                            new_points.append(point)

                    if new_points:
                        # 添加新点
                        new_ref_points = np.vstack([points_ref_cropped, np.array(new_points)])

                        # 重新计算重叠率
                        final_overlap_ratio, _ = self.calculate_overlap(
                            points_src_cropped, new_ref_points, transform_gt
                        )

                        return points_src_cropped, new_ref_points, final_overlap_ratio

        # 如果调整失败，返回原始点云和实际重叠率
        final_overlap_ratio, _ = self.calculate_overlap(
            points_src_cropped, points_ref_cropped, transform_gt
        )
        return points_src_cropped, points_ref_cropped, final_overlap_ratio

    def __call__(self, sample):
        # 要求共享方向时在call中生成randxyz，否则在crop中生成
        rand_xyz = None
        if self.SameDirection:
            rand_xyz = uniform_2_sphere()

        if np.all(self.p_keep == 1.0):
            return sample  # No need crop

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if len(self.p_keep) == 1:
            # 单独裁剪时不需考虑重叠率
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0], rand_xyz)
        else:
            # 双端裁剪时考虑重叠率
            points_src_cropped = self.crop(sample['points_src'], self.p_keep[0], rand_xyz)
            points_ref_cropped = self.crop(sample['points_ref'], self.p_keep[1],
                                         rand_xyz if self.SameDirection else None)

            # 调整重叠率确保符合目标值
            points_src_adjusted, points_ref_adjusted, final_overlap = self.adjust_overlap(
                points_src_cropped, points_ref_cropped, sample['points_ref'], sample['transform_gt'], self.target_overlap_ratio
            )

            sample['points_src'] = points_src_adjusted
            sample['points_ref'] = points_ref_adjusted
            sample['overlap_ratio'] = final_overlap

        return sample


class Resampler:
    def __init__(self, num: int):
        """随机重采样
        Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'] = self._resample(sample['points'], self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            sample['points_src'] = self._resample(sample['points_src'], src_size)
            sample['points_ref'] = self._resample(sample['points_ref'], ref_size)

        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]


class FixedResampler(Resampler):
    """Fixed resampling to always choose the first N points.
    Always deterministic regardless of whether the deterministic flag has been set
    """
    @staticmethod
    def _resample(points, k):
        multiple = k // points.shape[0]
        remainder = k % points.shape[0]

        resampled = np.concatenate((np.tile(points, (multiple, 1)), points[:remainder, :]), axis=0)
        return resampled


class FPSResampler:
    def __init__(self, num: int):
        """最远点采样
        Args:  num (int): Number of points to resample to, i.e. M
        """
        self.num = num

    def __call__(self, sample):
        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'crop_proportion' not in sample:
            src_size, ref_size = self.num, self.num
        elif len(sample['crop_proportion']) == 1:
            src_size = math.ceil(sample['crop_proportion'][0] * self.num)
            ref_size = self.num
        elif len(sample['crop_proportion']) == 2:
            src_size = math.ceil(sample['crop_proportion'][0] * self.num)
            ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
        else:
            raise ValueError('Crop proportion must have 1 or 2 elements')

        #sample['points_raw'] = sample['points_ref']
        sample['points_src'] = self._farthest_point_sample(sample['points_src'], src_size)
        sample['points_ref'] = self._farthest_point_sample(sample['points_ref'], ref_size)

        return sample

    def _farthest_point_sample(self, points, k):
        """
        Applies the Farthest Point Sampling (FPS) algorithm to select 'k' points from 'points'.

        Args:
            points (np.array): Point cloud data of shape (N, 3) where N is the number of points.
            k (int): Number of points to sample.

        Returns:
            np.array: The resampled points of shape (k, 3).
        """
        # Assuming points is a numpy array
        N = points.shape[0]
        if k >= N:
            return points  # If k >= N, return the points as is

        # Convert points to tensor for FPS
        points_tensor = torch.tensor(points).float().unsqueeze(0)  # Add batch dimension
        fps_idx = self._farthest_point_sample_torch(points_tensor, k)  # Apply FPS sampling
        return points[fps_idx[0].numpy()]  # Convert indices back to numpy and return sampled points

    def _farthest_point_sample_torch(self, xyz, npoint):
        """
        Performs Farthest Point Sampling (FPS) on the given point cloud.

        Args:
            xyz (torch.Tensor): Point cloud data of shape (B, N, 3).
            npoint (int): Number of points to sample.

        Returns:
            torch.Tensor: Sampled indices of shape (B, npoint).
        """
        device = xyz.device
        xyz = xyz[..., :3]
        B, N, C = xyz.shape  # B: Batch size, N: Number of points, C: Channels
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10  # Initializing distance matrix
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # Randomly initialize farthest points
        batch_indices = torch.arange(B, dtype=torch.long).to(device)

        for i in range(npoint):
            centroids[:, i] = farthest  # Store the index of the farthest point
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # Get coordinates of the farthest point
            dist = torch.sum((xyz - centroid) ** 2, -1)  # Calculate squared Euclidean distances
            mask = dist < distance
            distance[mask] = dist[mask]  # Update distance matrix
            farthest = torch.max(distance, -1)[1]  # Select the next farthest point

        return centroids


class VoxelResampler:
    """体素下采样在测试数据使用用于平衡不同来源的点云数据"""
    def __init__(self, voxel_size: float):
        """体素下采样
        Args:
            voxel_size (float): Size of the voxel
        """
        self.voxel_size = voxel_size
    def __call__(self, sample):
        sample['points_src'] = self._voxel_downsample(sample['points_src'], self.voxel_size)
        sample['points_ref'] = self._voxel_downsample(sample['points_ref'], self.voxel_size)
        return sample

    def _voxel_downsample(self, points, voxel_size):
        """调用pcl的voxel_downsample实现体素下采样"""
        pcd = o3d.geometry.PointCloud()
        points = np.ascontiguousarray(points, dtype=np.float64)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd.points, dtype=np.float32)


class RandomJitter:
    """ generate perturbations """
    def __init__(self, scale=0.01, clip=0.05, only_ref=False):
        self.scale = scale
        self.clip = clip
        self.only_ref = only_ref

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 3)),
                        a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):

        if 'points' in sample:
            sample['points'] = self.jitter(sample['points'])
        else:
            if not self.only_ref:
                sample['points_src'] = self.jitter(sample['points_src'])
            sample['points_ref'] = self.jitter(sample['points_ref'])

        return sample


class TransformSE3:
    def __init__(self):
        """Applies a random rigid transformation to the source point cloud"""

    def apply_transform(self, p0, transform_mat):
        p1 = (transform_mat[:3, :3] @ p0[:, :3].T).T + transform_mat[:3, 3]
        if p0.shape[1] >= 6:  # Need to rotate normals too
            n1 = (transform_mat[:3, :3] @ p0[:, 3:6].T).T
            p1 = np.concatenate((p1, n1), axis=-1)
        if p0.shape[1] == 4:  # label (pose estimation task)
            p1 = np.concatenate((p1, p0[:, 3][:, None]), axis=-1)
        if p0.shape[1] > 6:  # additional channels after normals
            p1 = np.concatenate((p1, p0[:, 6:]), axis=-1)

        igt = transform_mat
        # invert to get gt
        gt = igt.copy()
        gt[:3, :3] = gt[:3, :3].T
        gt[:3, 3] = -gt[:3, :3] @ gt[:3, 3]

        return p1, gt, igt

    def __call__(self, sample):
        raise NotImplementedError("Subclasses implement transformation (random, given, etc).")


class RandomTransformSE3(TransformSE3):
    def __init__(self, rot_mag: float = ROTATE, trans_mag: float = TRANSLATE, random_mag: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        super().__init__()
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        if self._random_mag:
            rot_mag, trans_mag = np.random.uniform() * self._rot_mag, np.random.uniform() * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_dcm(rand_rot))
        axis_angle /= np.linalg.norm(axis_angle)
        axis_angle *= np.deg2rad(rot_mag)
        rand_rot = Rotation.from_rotvec(axis_angle).as_dcm()

        # Generate translation
        rand_trans = uniform_2_sphere()
        rand_trans *= np.random.uniform(high=trans_mag)
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1).astype(np.float32)

        return rand_SE3

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, sample):
        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'transform_gt' in sample:
            # 提取gt变换，对目标点云应用随机变换
            transform_gt = sample['transform_gt']
            points_ref, random_transform, i_random_transform = self.transform(sample['points_ref'])

            # 组合原始gt和新随机变换
            combined_transform = np.dot(np.concatenate((i_random_transform, np.array([[0, 0, 0, 1]])), axis=0), transform_gt)

            # 更新GT变换为组合变换
            sample['points_ref'] = points_ref
            sample['transform_gt'] = combined_transform

            # # 再对源点云叠加一次随机变换使之偏离原点，同时会带来更多的误差，可注释掉
            # points_src, random_transform, i_random_transform = self.transform(sample['points_src'])
            # # 组合变换
            # combined_transform = np.dot(combined_transform, np.concatenate((random_transform, np.array([[0, 0, 0, 1]])), axis=0))
            # # 更新源点云
            # sample['points_src'] = points_src
            # # 更新GT变换为组合变换
            # sample['transform_gt'] = combined_transform

        else:
            # 若无GT变换直接进行随机变换
            if 'points' in sample:
                sample['points'], _, _ = self.transform(sample['points'])
            else:
                transformed, transform_r_s, transform_s_r = self.transform(sample['points_src'])
                sample['transform_gt'] = transform_r_s  # Apply to source to get reference
                sample['points_src'] = transformed

        return sample


# noinspection PyPep8Naming
class RandomTransformSE3_euler(RandomTransformSE3):
    """Same as RandomTransformSE3, but rotates using euler angle rotations

    This transformation is consistent to Deep Closest Point but does not
    generate uniform rotations

    """
    def generate_transform(self):

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3


class ShufflePoints:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        if 'points' in sample:
            sample['points'] = np.random.permutation(sample['points'])
        else:
            sample['points_ref'] = np.random.permutation(sample['points_ref'])
            sample['points_src'] = np.random.permutation(sample['points_src'])
        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def __call__(self, sample):
        sample['deterministic'] = True
        return sample


# Additional augmentations proposed in ReAgent
class Scale:
    """Scales source and target by a random scaling factor."""
    def __init__(self, scale=0.1, clip=0.5):
        self.scale = scale
        self.clip = clip

    def scale_points(self, points, scale):
        points_mean = points[:, :3].mean(axis=0)
        points[:, :3] = (points - points_mean) * scale + points_mean  # centroid location stays the same
        return points

    def __call__(self, sample: Dict):
        sample['scale'] = np.random.normal(1, self.scale, 3).clip(1 - self.clip, 1 + self.clip)

        sample['points_src'][:, :3] = self.scale_points(sample['points_src'][:, :3], sample['scale'])
        sample['points_ref'][:, :3] = self.scale_points(sample['points_ref'][:, :3], sample['scale'])
        #sample['points_raw'][:, :3] = self.scale_points(sample['points_raw'][:, :3], sample['scale'])
        return sample


class Shear:
    """Shears source and target by a random angle and shear plane."""
    def __init__(self, scale=5, clip=15):
        self.scale = scale
        self.clip = clip

    def __call__(self, sample: Dict):
        angle = np.deg2rad(np.clip(np.random.normal(0, self.scale), -self.clip, self.clip))
        direction = uniform_2_sphere()
        normal = np.cross(direction, uniform_2_sphere())
        S = t3d.shears.sadn2mat(angle, direction, normal)

        sample['points_src'][:, :3] = (S @ sample['points_src'][:, :3].T).T
        sample['points_ref'][:, :3] = (S @ sample['points_ref'][:, :3].T).T
        sample['points_raw'][:, :3] = (S @ sample['points_raw'][:, :3].T).T

        return sample


class Mirror:
    """Mirrors source and target through a random plane of reflection."""
    def __call__(self, sample: Dict):
        normal = uniform_2_sphere()
        M = t3d.reflections.rfnorm2mat(normal)[:3, :3]

        sample['points_src'][:, :3] = (M @ sample['points_src'][:, :3].T).T
        sample['points_ref'][:, :3] = (M @ sample['points_ref'][:, :3].T).T
        #sample['points_raw'][:, :3] = (M @ sample['points_raw'][:, :3].T).T

        return sample


class Normalize:
    """Normalizes source and target to be mean-centered and scales s.t. farthest point is of distance 1."""
    def __init__(self, using_target=True):
        self.using_target = using_target  # normalize wrt target

    def __call__(self, sample):
        # 以points_ref的中心为原点，缩放至其最大距离为1
        t = sample['points_ref'][:, :3].mean(axis=0)  # center offset
        centered = sample['points_ref'][:, :3] - t
        dists = np.linalg.norm(centered, axis=1)
        s = dists.max()  # scale

        # apply to source and target
        sample['points_ref'][:, :3] = centered / s
        sample['points_src'][:, :3] = (sample['points_src'][:, :3] - t) / s

        # for test set with given estimate in unnormalized scale
        if 'transform_gt' in sample:
            sample['transform_gt'][:3, 3] /= s

        # keep track (to undo if needed)
        sample['normalization'] = np.eye(4, dtype=np.float32)
        sample['normalization'][np.diag_indices(3)] = s
        sample['normalization'][:3, 3] = t.squeeze()

        return sample
