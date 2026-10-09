import torch
import functools

"""
实现解耦的转换
解耦转换分离了平移和旋转操作 使绕轴旋转不被平移干涉
提供来自pytorch3d的旋转转换
"""


def compute_inverse_transformation(trafo):
    """
    计算刚体变换矩阵的逆矩阵
    trafo: 形状为 [B, 4, 4] 的变换矩阵（支持批量）
    """
    R = trafo[:, :3, :3]  # 旋转部分
    t = trafo[:, :3, 3]   # 平移部分

    # 计算逆矩阵
    R_inv = R.transpose(-1, -2)           # R^T
    t_inv = -torch.bmm(R_inv, t.unsqueeze(-1)).squeeze(-1)  # -R^T * t

    # 构建逆矩阵
    inv_trafo = torch.eye(4, device=trafo.device).repeat(trafo.shape[0], 1, 1)
    inv_trafo[:, :3, :3] = R_inv
    inv_trafo[:, :3, 3] = t_inv

    return inv_trafo


def apply_trafo(pcd, trafo, disentangled=True):
    """
    将一个变换应用到一个点云上
    """
    ret = pcd.clone()

    if disentangled:  # 若解耦，回到原点进行旋转
        ret_mean = ret[..., :3].mean(dim=1)[:, None, :]
        ret[..., :3] -= ret_mean
    ret[..., :3] = (trafo[:, :3, :3] @ ret[..., :3].transpose(-1, -2)).transpose(-1, -2)  # rotate
    if disentangled:  # 再平移回原本位置
        ret[..., :3] += ret_mean
    ret[..., :3] += trafo[:, :3, 3][..., None, :]  # translate

    return ret


def to_disentangled(poses, pcd):
    """
    将点云的全局变换转换为解耦形式
    """
    poses[:, :3, 3] = poses[:, :3, 3] - pcd[..., :3].mean(dim=1) \
                      + (poses[:, :3, :3] @ pcd[..., :3].mean(dim=1)[:, :, None]).view(-1, 3)
    return poses


def to_global(poses, pcd):
    """
    与to_disentangled相反
    将解耦的变换转换回全局变换
    """
    poses[:, :3, 3] = poses[:, :3, 3] + pcd[..., :3].mean(dim=1) \
                      - (poses[:, :3, :3] @ pcd[..., :3].mean(dim=1)[:, :, None]).view(-1, 3)
    return poses


def square_distance(pcd1, pcd2):
    """
    计算两个点云之间的平方距离
    """
    return torch.sum((pcd1[:, :, None, :].contiguous() - pcd2[:, None, :, :].contiguous()) ** 2, dim=-1)


# from pytorch3d
def _axis_angle_rotation(axis: str, angle):
    """
    根据给定的轴和欧拉角角度生成旋转矩阵

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    将以欧拉角（弧度）表示的旋转转换为旋转矩阵。

    Args:
        euler_angles: 以弧度为单位的欧拉角作为形状张量 (..., 3).
        convention: {"X", "Y", and "Z"}.

    Returns:
        张量形式的旋转矩阵 (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    从一个旋转矩阵中提取第一个或第三个欧拉角
    使用反正切函数计算角度

    Args:
        axis: 正在寻找的角度“X” "Y" “Z”
        other_axis: 中轴“X” "Y" "Z"
        data: 作为形状张量的旋转矩阵 (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def matrix_to_euler_angles(matrix, convention: str):
    """
    将旋转矩阵转换回欧拉角

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)
