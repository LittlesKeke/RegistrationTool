
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 选点窗口
def pick_points(pcd, window_title="请选择点"):
    """
    弹出一个阻塞式窗口让用户选择点。
    Args:
        pcd (open3d.geometry.PointCloud): 要在其中选择点的点云。
        window_title (str): 窗口标题。

    Returns:
        list[int]: 用户选择的点的索引列表。
    """
    # 使用 draw_geometries_with_editing，这是专门用于此任务的阻塞式函数
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_title)
    vis.add_geometry(pcd)
    vis.run()  # 这里会阻塞程序，直到用户关闭窗口
    vis.destroy_window()

    return vis.get_picked_points()

# 手动+ICP配准逻辑
def manual_registration(ui_main_window):
    """统一的手动配准函数"""
    # GICP参数
    icp_convergence_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=100,
        relative_fitness=1e-6,
        relative_rmse=1e-6
    )

    # 读取点云
    source_pcd = ui_main_window.o3d_widget_src.get_origin_pcd()  # 获取源点云
    target_pcd = ui_main_window.o3d_widget_tgt.get_point_cloud()  # 获取目标点云
    #align_centroids(source_pcd, target_pcd)

    # 若已经进行过配准（增量配准模式）
    if ui_main_window.o3d_widget_result.source_pcd is not None:
        combined_matrix = np.eye(4)  # 初始化一个单位矩阵
        for matrix in ui_main_window.registration_matrices:
            combined_matrix = np.dot(matrix, combined_matrix)  # 逐个矩阵进行乘法合并
        ui_main_window.registration_matrices.clear()# 清除历史
        icp_p2p = o3d.pipelines.registration.registration_generalized_icp(
            source_pcd, target_pcd,
            15,
            init=combined_matrix,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria=icp_convergence_criteria
        )

    else:
        # 初次配准：使用对应点配准
        # 读取选择点（从widget的selected_points属性获取）
        picked_source = getattr(ui_main_window.o3d_widget_src, 'selected_points', [])
        picked_target = getattr(ui_main_window.o3d_widget_tgt, 'selected_points', [])

        # 检查选点的数量
        if len(picked_target) < 3 or len(picked_source) < 3:
            raise ValueError("至少需要选择3个点!")
        if len(picked_target) != len(picked_source):
            raise ValueError("两个点云的选点数量必须相等!")

        # 生成对应点对
        corr = np.zeros((len(picked_source), 2))
        corr[:, 0] = picked_source
        corr[:, 1] = picked_target

        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source_pcd, target_pcd, o3d.utility.Vector2iVector(corr))
        icp_p2p = o3d.pipelines.registration.registration_generalized_icp(
            source_pcd, target_pcd, 16, trans_init,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            icp_convergence_criteria
        )

    # 保存变换矩阵到历史记录中
    ui_main_window.registration_matrices.append(icp_p2p.transformation)

    # 清除临时保存的选择点（可选）
    if hasattr(ui_main_window.o3d_widget_src, 'selected_points'):
        delattr(ui_main_window.o3d_widget_src, 'selected_points')
    if hasattr(ui_main_window.o3d_widget_tgt, 'selected_points'):
        delattr(ui_main_window.o3d_widget_tgt, 'selected_points')

    return icp_p2p.transformation

def translate_point_cloud(ui_main_window, axis, direction):
    # 平移源点云（根据按钮传入的轴和方向）
    translation_vector = [0.0, 0.0, 0.0]

    if axis == 'x':
        if direction == 'plus':
            translation_vector = [5.0, 0.0, 0.0]  # x轴正向平移
        elif direction == 'minus':
            translation_vector = [-5.0, 0.0, 0.0]  # x轴负向平移
    elif axis == 'y':
        if direction == 'plus':
            translation_vector = [0.0, 5.0, 0.0]  # y轴正向平移
        elif direction == 'minus':
            translation_vector = [0.0, -5.0, 0.0]  # y轴负向平移
    elif axis == 'z':
        if direction == 'plus':
            translation_vector = [0.0, 0.0, 5.0]  # z轴正向平移
        elif direction == 'minus':
            translation_vector = [0.0, 0.0, -5.0]  # z轴负向平移

    ui_main_window.o3d_widget_result.translate_point_cloud(translation_vector)
    # 返回该步骤的变换矩阵
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation_vector
    ui_main_window.registration_matrices.append(translation_matrix)  # 保存变换矩阵到列表中
    return

# 计算点云的质心并将两个点云的质心对齐
def align_centroids(pcd1, pcd2):
    # 计算质心
    centroid1 = np.mean(np.asarray(pcd1.points), axis=0)
    centroid2 = np.mean(np.asarray(pcd2.points), axis=0)

    # 计算平移量，将两个点云的质心对齐
    translation1 = np.array(-centroid1, dtype=np.float64)
    translation2 = np.array(-centroid2, dtype=np.float64)

    # 平移点云
    pcd1.points = o3d.utility.Vector3dVector(np.asarray(pcd1.points) + translation1)
    pcd2.points = o3d.utility.Vector3dVector(np.asarray(pcd2.points) + translation2)

    return


# 计算每个源点云点与目标点云的最近点距离
def calculate_normalized_distances(source_pcd, target_pcd):
    distances = source_pcd.compute_point_cloud_distance(target_pcd)

    # 将误差归一化
    distances = np.asarray(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    normalized_distances = (distances - min_distance) / (max_distance - min_distance) # 将距离归一化到[0, 1]

    return normalized_distances, max_distance

# 颜色映射
def map_colors_based_on_distance(normalized_distances):
    colors = np.zeros((len(normalized_distances), 3))
    cmap = plt.get_cmap('jet')

    for i, normalized_distance in enumerate(normalized_distances):
        r, g, b, _ = cmap(normalized_distance)
        colors[i, 0] = r  # R
        colors[i, 1] = g  # G
        colors[i, 2] = b  # B

    return colors
