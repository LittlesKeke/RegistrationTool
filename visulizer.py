## visulizer.py
import copy
import numpy as np
import open3d as o3d
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib.pyplot as plt
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

"""
基于 Open3D 计算结果
基于 VTK 渲染
"""

def _points_to_vtk_polydata(points_np, colors_np=None) -> vtk.vtkPolyData:
    points_np = np.asarray(points_np)
    poly = vtk.vtkPolyData()
    if points_np.size == 0:
        return poly

    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(points_np.shape[0])
    for i, p in enumerate(points_np):
        vtk_points.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))

    verts = vtk.vtkCellArray()
    for i in range(points_np.shape[0]):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)

    poly.SetPoints(vtk_points)
    poly.SetVerts(verts)

    if colors_np is not None:
        colors_np = np.asarray(colors_np)
        if colors_np.shape[0] == points_np.shape[0]:
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Colors")
            for c in (colors_np[:, :3] * 255.0).clip(0, 255).astype(np.uint8):
                colors.InsertNextTuple3(int(c[0]), int(c[1]), int(c[2]))
            poly.GetPointData().SetScalars(colors)

    return poly


def _pcd_to_vtk_polydata(pcd: o3d.geometry.PointCloud) -> vtk.vtkPolyData:
    """将 Open3D 点云转换为 VTK PolyData（仅用于显示，不改变原逻辑）"""
    points_np = np.asarray(pcd.points)
    colors_np = np.asarray(pcd.colors) if pcd.has_colors() else None
    return _points_to_vtk_polydata(points_np, colors_np)


class BaseWidget(QtWidgets.QWidget):
    """
    基类
    - 保留 origin_point_cloud / point_cloud 等 Open3D 数据结构用于计算
    - 显示部分使用 VTK
    """

    def __init__(self, visualizer=None, parent=None):
        # visualizer 参数仅为兼容旧代码，不再使用
        super(BaseWidget, self).__init__(parent)

        self.setMinimumSize(260, 180)
        self.source_pcd = None
        self.target_pcd = None
        self.origin_point_cloud = None
        self.point_cloud = None  # 当前显示的点云
        self._is_closed = False   # 标记窗口是否已关闭，避免关闭后仍然渲染

        # --- VTK 渲染管线 ---
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        # 统一使用 TrackballCamera 交互风格，让所有几何体随相机一起运动
        camera_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(camera_style)
        self.vtk_widget.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.vtk_widget.setMouseTracking(True)
        QtCore.QTimer.singleShot(0, self._initialize_vtk_interactor)

        self.renderer.SetBackground(0.055, 0.075, 0.09)

        # 用于单点云显示的 actor / mapper
        self._single_mapper = vtk.vtkPolyDataMapper()
        self._single_actor = vtk.vtkActor()
        self._single_actor.SetMapper(self._single_mapper)

        # 点大小（类似 Open3D render_option.point_size）
        self._single_actor.GetProperty().SetPointSize(2.0)
        self.renderer.AddActor(self._single_actor)

        # 初始化渲染，确保窗口显示为黑色而不是白色
        if self.vtk_widget is not None:
            rw = self.vtk_widget.GetRenderWindow()
            if rw is not None:
                rw.Render()

    def _initialize_vtk_interactor(self):
        if self._is_closed:
            return
        try:
            self.vtk_widget.Initialize()
        except Exception:
            pass
        try:
            if self.interactor is not None:
                self.interactor.Enable()
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._render)

    # ===================== 公共逻辑 =====================
    def _apply_z_based_coloring(self, pcd):
        """应用基于 Z 值的着色 - 浅灰到黑色（保持原逻辑）"""
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            return

        z = pts[:, 2]
        z_min, z_max = np.min(z), np.max(z)
        if z_max == z_min:
            z_max += 1e-8  # 避免除以零
        t = (z - z_min) / (z_max - z_min)

        # 使用 matplotlib 的灰度颜色映射，随后反转得到浅灰到黑色
        colors = plt.get_cmap('gray')(t)[:, :3]
        colors = 1.0 - colors
        pcd.colors = o3d.utility.Vector3dVector(colors)

    def _render(self):
        """统一触发一次渲染（在窗口还存活时才执行）"""
        if self._is_closed:
            return
        rw = self.vtk_widget.GetRenderWindow() if self.vtk_widget is not None else None
        if rw is None:
            return
        if hasattr(self, "_before_render"):
            self._before_render()
        rw.Render()

    # ===================== 对外接口（单点云） =====================
    def load_point_cloud(self, file_name, color_uniform=False):
        """加载点云并显示（对外接口保持不变）"""
        new_pcd = o3d.io.read_point_cloud(file_name)
        if not new_pcd.has_points():
            raise ValueError(f"Could not load a non-empty point cloud: {file_name}")

        self.origin_point_cloud = copy.deepcopy(new_pcd)
        self.point_cloud = new_pcd

        if not color_uniform:
            self._apply_z_based_coloring(self.point_cloud)

        poly = _pcd_to_vtk_polydata(self.point_cloud)
        self._single_mapper.SetInputData(poly)
        self.renderer.ResetCamera()
        self._render()

    def get_origin_pcd(self):
        """获取原始点云"""
        return self.origin_point_cloud

    def get_point_cloud(self):
        """获取当前点云"""
        return self.point_cloud

    def close(self):
        """手动调用关闭方法，释放VTK资源"""
        if not self._is_closed:
            self._is_closed = True
            try:
                if self.interactor is not None:
                    self.interactor.Disable()
            except Exception:
                pass
            try:
                rw = self.vtk_widget.GetRenderWindow() if self.vtk_widget is not None else None
                if rw is not None:
                    rw.Finalize()
            except Exception:
                pass

    def closeEvent(self, event):
        """在关闭 Qt 控件时，优雅地释放 VTK 资源，避免 wglMakeCurrent 报错"""
        self.close()
        super().closeEvent(event)


class InputVisualizerWidget(BaseWidget):
    """输入点云窗口(仅输入点云)"""

    point_selected = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(visualizer=None, parent=parent)

    # 这里保留接口以避免外部调用报错，目前未在主流程中使用
    def get_selected_points(self):
        return []


class SlicePanInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Interactor style for 2D slice views: left-drag pans instead of rotates."""

    def OnLeftButtonDown(self):
        self.OnMiddleButtonDown()

    def OnLeftButtonUp(self):
        self.OnMiddleButtonUp()


class OutputVisualizerWidget(BaseWidget):
    """配准结果可视化窗口（包含源点云、目标点云、坐标轴和三个正交切面）"""

    def __init__(self, parent=None):
        super().__init__(visualizer=None, parent=parent)
        self.target_pcd = None
        self.source_pcd = None
        # 三个切面的位置
        self.cut_plane_x = 0.0  # yoz平面（x轴切面）
        self.cut_plane_y = 0.0  # xoz平面（y轴切面）
        self.cut_plane_z = 0.0  # xoy平面（z轴切面）
        self.cut_planes_locked = False  # 是否锁定当前切面
        self.cut_plane_widget = None # 通信Frame4可视化组件

        # 结果窗口需要两个独立的 actor：源点云和目标点云
        self._src_mapper = vtk.vtkPolyDataMapper()
        self._src_actor = vtk.vtkActor()
        self._src_actor.SetMapper(self._src_mapper)
        self._src_actor.GetProperty().SetPointSize(2.0)

        self._tgt_mapper = vtk.vtkPolyDataMapper()
        self._tgt_actor = vtk.vtkActor()
        self._tgt_actor.SetMapper(self._tgt_mapper)
        self._tgt_actor.GetProperty().SetPointSize(2.0)

        self.renderer.RemoveActor(self._single_actor)
        self.renderer.AddActor(self._src_actor)
        self.renderer.AddActor(self._tgt_actor)

        # 添加坐标轴
        self.add_coordinate()

        # 切面可视化相关组件
        self._init_cut_planes()

        # 交互样式
        self.style_camera = vtk.vtkInteractorStyleTrackballCamera()
        self.style_actor = vtk.vtkInteractorStyleTrackballActor()

    def add_coordinate(self):
        """添加坐标轴参考系（用 VTK 实现）"""
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(200.0, 200.0, 200.0)
        axes.SetShaftTypeToCylinder()
        self.renderer.AddActor(axes)
        self._render()

    def _init_cut_planes(self):
        """初始化三个切面可视化组件（XOY, XOZ, YOZ）"""
        # 获取点云边界用于设置切面大小
        self._plane_size = 1000.0  # 默认大小，将在加载点云后更新

        # XOY平面（z轴切面）
        self._plane_xoy = vtk.vtkPlaneSource()
        self._plane_xoy.SetCenter(0, 0, self.cut_plane_z)
        self._plane_xoy.SetNormal(0, 0, 1)  # 法向量为z轴
        self._plane_xoy.SetXResolution(20)
        self._plane_xoy.SetYResolution(20)

        self._plane_xoy_mapper = vtk.vtkPolyDataMapper()
        self._plane_xoy_mapper.SetInputConnection(self._plane_xoy.GetOutputPort())
        self._plane_xoy_actor = vtk.vtkActor()
        self._plane_xoy_actor.SetMapper(self._plane_xoy_mapper)
        # 浅黑色突出显示（医学图像常用）
        self._plane_xoy_actor.GetProperty().SetColor(0.2, 0.2, 0.2)  # 浅黑色
        self._plane_xoy_actor.GetProperty().SetOpacity(0.4)

        # XOZ平面（y轴切面）
        self._plane_xoz = vtk.vtkPlaneSource()
        self._plane_xoz.SetCenter(0, self.cut_plane_y, 0)
        self._plane_xoz.SetNormal(0, 1, 0)  # 法向量为y轴
        self._plane_xoz.SetXResolution(20)

        self._plane_xoz_mapper = vtk.vtkPolyDataMapper()
        self._plane_xoz_mapper.SetInputConnection(self._plane_xoz.GetOutputPort())
        self._plane_xoz_actor = vtk.vtkActor()
        self._plane_xoz_actor.SetMapper(self._plane_xoz_mapper)
        self._plane_xoz_actor.GetProperty().SetColor(0.2, 0.2, 0.2)  # 浅黑色
        self._plane_xoz_actor.GetProperty().SetOpacity(0.4)

        # YOZ平面（x轴切面）
        self._plane_yoz = vtk.vtkPlaneSource()
        self._plane_yoz.SetCenter(self.cut_plane_x, 0, 0)
        self._plane_yoz.SetNormal(1, 0, 0)  # 法向量为x轴
        self._plane_yoz.SetYResolution(20)

        self._plane_yoz_mapper = vtk.vtkPolyDataMapper()
        self._plane_yoz_mapper.SetInputConnection(self._plane_yoz.GetOutputPort())
        self._plane_yoz_actor = vtk.vtkActor()
        self._plane_yoz_actor.SetMapper(self._plane_yoz_mapper)
        self._plane_yoz_actor.GetProperty().SetColor(0.2, 0.2, 0.2)  # 浅黑色
        self._plane_yoz_actor.GetProperty().SetOpacity(0.4)

        # 切面上的点云（每个切面都有源点和目标点）
        # XOY平面
        self._cut_xoy_src_mapper = vtk.vtkPolyDataMapper()
        self._cut_xoy_src_actor = vtk.vtkActor()
        self._cut_xoy_src_actor.SetMapper(self._cut_xoy_src_mapper)
        self._cut_xoy_src_actor.GetProperty().SetPointSize(3.0)
        self._cut_xoy_src_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 红色点

        self._cut_xoy_tgt_mapper = vtk.vtkPolyDataMapper()
        self._cut_xoy_tgt_actor = vtk.vtkActor()
        self._cut_xoy_tgt_actor.SetMapper(self._cut_xoy_tgt_mapper)
        self._cut_xoy_tgt_actor.GetProperty().SetPointSize(3.0)
        self._cut_xoy_tgt_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # 蓝色点

        # XOZ平面
        self._cut_xoz_src_mapper = vtk.vtkPolyDataMapper()
        self._cut_xoz_src_actor = vtk.vtkActor()
        self._cut_xoz_src_actor.SetMapper(self._cut_xoz_src_mapper)
        self._cut_xoz_src_actor.GetProperty().SetPointSize(3.0)
        self._cut_xoz_src_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 红色点

        self._cut_xoz_tgt_mapper = vtk.vtkPolyDataMapper()
        self._cut_xoz_tgt_actor = vtk.vtkActor()
        self._cut_xoz_tgt_actor.SetMapper(self._cut_xoz_tgt_mapper)
        self._cut_xoz_tgt_actor.GetProperty().SetPointSize(3.0)
        self._cut_xoz_tgt_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # 蓝色点

        # YOZ平面
        self._cut_yoz_src_mapper = vtk.vtkPolyDataMapper()
        self._cut_yoz_src_actor = vtk.vtkActor()
        self._cut_yoz_src_actor.SetMapper(self._cut_yoz_src_mapper)
        self._cut_yoz_src_actor.GetProperty().SetPointSize(3.0)
        self._cut_yoz_src_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 红色点

        self._cut_yoz_tgt_mapper = vtk.vtkPolyDataMapper()
        self._cut_yoz_tgt_actor = vtk.vtkActor()
        self._cut_yoz_tgt_actor.SetMapper(self._cut_yoz_tgt_mapper)
        self._cut_yoz_tgt_actor.GetProperty().SetPointSize(3.0)
        self._cut_yoz_tgt_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # 蓝色点

        # 添加所有切面和切面点到渲染器
        self.renderer.AddActor(self._plane_xoy_actor)
        self.renderer.AddActor(self._plane_xoz_actor)
        self.renderer.AddActor(self._plane_yoz_actor)
        self.renderer.AddActor(self._cut_xoy_src_actor)
        self.renderer.AddActor(self._cut_xoy_tgt_actor)
        self.renderer.AddActor(self._cut_xoz_src_actor)
        self.renderer.AddActor(self._cut_xoz_tgt_actor)
        self.renderer.AddActor(self._cut_yoz_src_actor)
        self.renderer.AddActor(self._cut_yoz_tgt_actor)

    def set_cut_plane_widget(self, widget):
        """设置切面可视化窗口的引用，用于联动更新"""
        self.cut_plane_widget = widget

    def _extract_cut_points(self, pcd, plane_type, position, epsilon=1.0):
        """从点云中提取位于指定切平面附近的点

        Args:
            pcd: 点云对象
            plane_type: 'xoy', 'xoz', 'yoz'
            position: 切面位置（x, y, 或 z 坐标）
            epsilon: 提取范围
        """
        if pcd is None:
            return vtk.vtkPolyData(), np.array([])

        points_np = np.asarray(pcd.points)
        if points_np.size == 0:
            return vtk.vtkPolyData(), np.array([])

        # 根据切面类型提取点
        if plane_type == 'xoy':  # z轴切面
            mask = np.abs(points_np[:, 2] - position) < epsilon
        elif plane_type == 'xoz':  # y轴切面
            mask = np.abs(points_np[:, 1] - position) < epsilon
        elif plane_type == 'yoz':  # x轴切面
            mask = np.abs(points_np[:, 0] - position) < epsilon
        else:
            return vtk.vtkPolyData(), np.array([])

        cut_points_np = points_np[mask]

        if cut_points_np.size == 0:
            return vtk.vtkPolyData(), np.array([])

        colors_np = np.asarray(pcd.colors)[mask] if pcd.has_colors() else None
        return _points_to_vtk_polydata(cut_points_np, colors_np), cut_points_np

    def _update_plane_size(self):
        """根据点云边界更新切面大小"""
        bounds = self.get_point_cloud_bounds()
        if bounds is None:
            return

        # 计算点云的尺寸
        x_size = bounds['x'][1] - bounds['x'][0]
        y_size = bounds['y'][1] - bounds['y'][0]
        z_size = bounds['z'][1] - bounds['z'][0]

        # 设置切面大小（略大于点云范围）
        margin = 50.0
        max_size = max(x_size, y_size, z_size) + margin * 2
        half_size = max_size / 2.0

        # 更新XOY平面（z轴切面）的尺寸
        self._plane_xoy.SetPoint1(half_size, -half_size, self.cut_plane_z)
        self._plane_xoy.SetPoint2(-half_size, half_size, self.cut_plane_z)
        self._plane_xoy.SetOrigin(-half_size, -half_size, self.cut_plane_z)

        # 更新XOZ平面（y轴切面）的尺寸
        self._plane_xoz.SetPoint1(half_size, self.cut_plane_y, -half_size)
        self._plane_xoz.SetPoint2(-half_size, self.cut_plane_y, half_size)
        self._plane_xoz.SetOrigin(-half_size, self.cut_plane_y, -half_size)

        # 更新YOZ平面（x轴切面）的尺寸
        self._plane_yoz.SetPoint1(self.cut_plane_x, half_size, -half_size)
        self._plane_yoz.SetPoint2(self.cut_plane_x, -half_size, half_size)
        self._plane_yoz.SetOrigin(self.cut_plane_x, -half_size, -half_size)

    def _update_cut_points_xoy(self):
        """更新XOY平面的切面点"""
        epsilon = 1.0
        if self.source_pcd is not None:
            cut_src_poly, _ = self._extract_cut_points(self.source_pcd, 'xoy', self.cut_plane_z, epsilon)
            self._cut_xoy_src_mapper.SetInputData(cut_src_poly)
        if self.target_pcd is not None:
            cut_tgt_poly, _ = self._extract_cut_points(self.target_pcd, 'xoy', self.cut_plane_z, epsilon)
            self._cut_xoy_tgt_mapper.SetInputData(cut_tgt_poly)

    def _update_cut_points_xoz(self):
        """更新XOZ平面的切面点"""
        epsilon = 1.0
        if self.source_pcd is not None:
            cut_src_poly, _ = self._extract_cut_points(self.source_pcd, 'xoz', self.cut_plane_y, epsilon)
            self._cut_xoz_src_mapper.SetInputData(cut_src_poly)
        if self.target_pcd is not None:
            cut_tgt_poly, _ = self._extract_cut_points(self.target_pcd, 'xoz', self.cut_plane_y, epsilon)
            self._cut_xoz_tgt_mapper.SetInputData(cut_tgt_poly)

    def _update_cut_points_yoz(self):
        """更新YOZ平面的切面点"""
        epsilon = 1.0
        if self.source_pcd is not None:
            cut_src_poly, _ = self._extract_cut_points(self.source_pcd, 'yoz', self.cut_plane_x, epsilon)
            self._cut_yoz_src_mapper.SetInputData(cut_src_poly)
        if self.target_pcd is not None:
            cut_tgt_poly, _ = self._extract_cut_points(self.target_pcd, 'yoz', self.cut_plane_x, epsilon)
            self._cut_yoz_tgt_mapper.SetInputData(cut_tgt_poly)

    def set_cut_plane_x(self, x_value):
        """设置YOZ平面（x轴切面）的位置并更新显示"""
        if self.cut_planes_locked:
            return
        self.cut_plane_x = x_value
        self._plane_yoz.SetCenter(self.cut_plane_x, 0, 0)
        self._update_cut_points_yoz()
        self._render()

    def set_cut_plane_y(self, y_value):
        """设置XOZ平面（y轴切面）的位置并更新显示"""
        if self.cut_planes_locked:
            return
        self.cut_plane_y = y_value
        self._plane_xoz.SetCenter(0, self.cut_plane_y, 0)
        self._update_cut_points_xoz()
        self._render()

    def set_cut_plane_z(self, z_value):
        """设置XOY平面（z轴切面）的位置并更新显示"""
        if self.cut_planes_locked:
            return
        self.cut_plane_z = z_value
        self._plane_xoy.SetCenter(0, 0, self.cut_plane_z)
        self._update_cut_points_xoy()
        self._render()

    def get_point_cloud_bounds(self):
        """获取点云的边界范围"""
        all_points = []
        if self.source_pcd is not None:
            src_points = np.asarray(self.source_pcd.points)
            if src_points.size > 0:
                all_points.append(src_points)
        if self.target_pcd is not None:
            tgt_points = np.asarray(self.target_pcd.points)
            if tgt_points.size > 0:
                all_points.append(tgt_points)

        if len(all_points) == 0:
            return None

        combined_points = np.vstack(all_points)
        return {
            'x': (np.min(combined_points[:, 0]), np.max(combined_points[:, 0])),
            'y': (np.min(combined_points[:, 1]), np.max(combined_points[:, 1])),
            'z': (np.min(combined_points[:, 2]), np.max(combined_points[:, 2]))
        }

    def load_point_clouds(self, source_pcd, target_pcd):
        """一次性加载源和目标点云"""
        # 每次加载新数据时解除切面锁定
        self.cut_planes_locked = False

        self.source_pcd = copy.deepcopy(source_pcd)  # 结构光源点云
        self.target_pcd = copy.deepcopy(target_pcd)  # MRI 目标点云

        # 仅对目标点云做 Z 着色，保持原逻辑
        self._apply_z_based_coloring(self.target_pcd)

        src_poly = _pcd_to_vtk_polydata(self.source_pcd)
        tgt_poly = _pcd_to_vtk_polydata(self.target_pcd)

        self._src_mapper.SetInputData(src_poly)
        self._tgt_mapper.SetInputData(tgt_poly)

        # 设置切面位置为点云的中心
        bounds = self.get_point_cloud_bounds()
        if bounds is not None:
            self.cut_plane_x = (bounds['x'][0] + bounds['x'][1]) / 2.0
            self.cut_plane_y = (bounds['y'][0] + bounds['y'][1]) / 2.0
            self.cut_plane_z = (bounds['z'][0] + bounds['z'][1]) / 2.0

            # 更新切面位置和大小
            self._update_plane_size()
            self._plane_xoy.SetCenter(0, 0, self.cut_plane_z)
            self._plane_xoz.SetCenter(0, self.cut_plane_y, 0)
            self._plane_yoz.SetCenter(self.cut_plane_x, 0, 0)

        # 更新所有切面的点云
        self._update_cut_points_xoy()
        self._update_cut_points_xoz()
        self._update_cut_points_yoz()

        self.renderer.ResetCamera()
        self._render()

    def _get_transformed_points_polydata(self, pcd, actor, plane_type, position, epsilon=1.0):
        """
        获取经过Actor变换后的点的PolyData，并进行切片提取
        返回: (vtkPolyData, numpy_points)
        """
        if pcd is None:
            return vtk.vtkPolyData(), np.array([])

        points = np.asarray(pcd.points)
        if points.size == 0:
            return vtk.vtkPolyData(), np.array([])

        # 获取Actor矩阵
        matrix = actor.GetUserMatrix()
        if matrix is None:
            # 如果没有矩阵，使用原始点
            transformed_points = points
        else:
            # 转换矩阵并计算
            transform = np.eye(4)
            for i in range(4):
                for j in range(4):
                    transform[i, j] = matrix.GetElement(i, j)

            # 齐次坐标变换
            points_h = np.hstack((points, np.ones((points.shape[0], 1))))
            transformed_points_h = points_h @ transform.T
            transformed_points = transformed_points_h[:, :3]

        # 提取切面点
        if plane_type == 'xoy':
            mask = np.abs(transformed_points[:, 2] - position) < epsilon
        elif plane_type == 'xoz':
            mask = np.abs(transformed_points[:, 1] - position) < epsilon
        elif plane_type == 'yoz':
            mask = np.abs(transformed_points[:, 0] - position) < epsilon
        else:
            return vtk.vtkPolyData(), np.array([])

        cut_points_np = transformed_points[mask]

        if cut_points_np.size == 0:
            return vtk.vtkPolyData(), np.array([])

        colors_np = np.asarray(pcd.colors)[mask] if pcd.has_colors() else None
        return _points_to_vtk_polydata(cut_points_np, colors_np), cut_points_np

    def update_source(self):
        """当 self.source_pcd 的颜色/位置更新后，刷新 VTK 显示"""
        if self.source_pcd is None:
            return
        src_poly = _pcd_to_vtk_polydata(self.source_pcd)
        self._src_mapper.SetInputData(src_poly)
        self._render()

    def set_cut_plane_lock(self, locked: bool):
        """锁定或解锁切面位置"""
        self.cut_planes_locked = locked

        if locked:
            # 1. 切换交互模式为拖动物体
            self.interactor.SetInteractorStyle(self.style_actor)

            # 2. 使切面 Actor 不可被拾取，防止误拖动
            self._plane_xoy_actor.SetPickable(False)
            self._plane_xoz_actor.SetPickable(False)
            self._plane_yoz_actor.SetPickable(False)

            # 3. 添加监听器，实时更新
            self._observer_id = self.interactor.AddObserver("InteractionEvent", self._on_actor_interaction)

        else:
            # 1. 移除监听器
            if hasattr(self, '_observer_id'):
                self.interactor.RemoveObserver(self._observer_id)

            # 2. 恢复切面可拾取（可选，如果不希望它们挡住鼠标射线可保持False）
            self._plane_xoy_actor.SetPickable(True)
            self._plane_xoz_actor.SetPickable(True)
            self._plane_yoz_actor.SetPickable(True)

            # 3. 将 Actor 的变换应用到 Open3D 数据中 ("Baking")
            self._apply_actor_transform(self._src_actor, self.source_pcd)
            self._apply_actor_transform(self._tgt_actor, self.target_pcd)

            # 4. 切换回相机控制模式
            self.interactor.SetInteractorStyle(self.style_camera)

            # 5. 刷新所有视图
            self._update_cut_points_all()
            if self.cut_plane_widget:
                self.cut_plane_widget.update_current_slice() # 让Frame4回退到读取pcd模式
            self._render()

    def _on_actor_interaction(self, obj, event):
        """
        当鼠标拖动发生时调用。
        核心逻辑：
        1. 找出被拖动的是哪个Actor（Src还是Tgt）。
        2. 将变换矩阵同步给另一个Actor（实现合并拖动）。
        3. 计算变换后的点云与固定切面的交点。
        4. 更新本窗口的红蓝点。
        5. 将切面数据发送给 Frame4。
        """
        # 1. 获取当前被拖动的对象
        # 修正：使用 GetInteractionProp() 而不是 GetInteractionActor()
        interaction_prop = None
        if hasattr(self.style_actor, 'GetInteractionProp'):
            interaction_prop = self.style_actor.GetInteractionProp()

        # 如果获取不到交互对象（某些极端情况），尝试检查谁的矩阵发生了变化
        if interaction_prop is None:
            # 备用方案：简单检查两个Actor是否有矩阵被激活
            # 注意：这只是一个fallback，通常 GetInteractionProp 应该能工作
            if self._src_actor.GetUserMatrix() is not None:
                interaction_prop = self._src_actor
            elif self._tgt_actor.GetUserMatrix() is not None:
                interaction_prop = self._tgt_actor
            else:
                return # 没有任何移动发生

        # 2. 联动逻辑：同步矩阵
        # 获取当前正在变动的矩阵
        current_matrix = interaction_prop.GetUserMatrix()

        if current_matrix:
            # 如果拖动的是源点云，强制目标点云跟随
            if interaction_prop == self._src_actor:
                self._tgt_actor.SetUserMatrix(current_matrix)
            # 如果拖动的是目标点云，强制源点云跟随
            elif interaction_prop == self._tgt_actor:
                self._src_actor.SetUserMatrix(current_matrix)

        # 准备数据用于更新 Frame 4
        f4_src_poly = None
        f4_tgt_poly = None
        current_frame4_plane = self.cut_plane_widget.current_plane if self.cut_plane_widget else 'xoy'

        # 3. 计算切点 (XOY平面 - Z轴切面)
        # 注意：这里传入的 actor 必须包含最新的 UserMatrix
        src_poly_xoy, src_pts_xoy = self._get_transformed_points_polydata(
            self.source_pcd, self._src_actor, 'xoy', self.cut_plane_z)
        tgt_poly_xoy, tgt_pts_xoy = self._get_transformed_points_polydata(
            self.target_pcd, self._tgt_actor, 'xoy', self.cut_plane_z)

        self._cut_xoy_src_mapper.SetInputData(src_poly_xoy)
        self._cut_xoy_tgt_mapper.SetInputData(tgt_poly_xoy)
        if current_frame4_plane == 'xoy':
            f4_src_poly, f4_tgt_poly = src_poly_xoy, tgt_poly_xoy

        # 3. 计算切点 (XOZ平面 - Y轴切面)
        src_poly_xoz, src_pts_xoz = self._get_transformed_points_polydata(
            self.source_pcd, self._src_actor, 'xoz', self.cut_plane_y)
        tgt_poly_xoz, tgt_pts_xoz = self._get_transformed_points_polydata(
            self.target_pcd, self._tgt_actor, 'xoz', self.cut_plane_y)

        self._cut_xoz_src_mapper.SetInputData(src_poly_xoz)
        self._cut_xoz_tgt_mapper.SetInputData(tgt_poly_xoz)
        if current_frame4_plane == 'xoz':
            f4_src_poly, f4_tgt_poly = src_poly_xoz, tgt_poly_xoz

        # 3. 计算切点 (YOZ平面 - X轴切面)
        src_poly_yoz, src_pts_yoz = self._get_transformed_points_polydata(
            self.source_pcd, self._src_actor, 'yoz', self.cut_plane_x)
        tgt_poly_yoz, tgt_pts_yoz = self._get_transformed_points_polydata(
            self.target_pcd, self._tgt_actor, 'yoz', self.cut_plane_x)

        self._cut_yoz_src_mapper.SetInputData(src_poly_yoz)
        self._cut_yoz_tgt_mapper.SetInputData(tgt_poly_yoz)
        if current_frame4_plane == 'yoz':
            f4_src_poly, f4_tgt_poly = src_poly_yoz, tgt_poly_yoz

        # 4. 渲染本窗口
        self._render()

        # 5. 实时更新 Frame 4
        if self.cut_plane_widget:
            self.cut_plane_widget.update_dynamic_slice(f4_src_poly, f4_tgt_poly)

    def _apply_actor_transform(self, actor, pcd):
        """将Actor变换应用到点云数据并重置Actor"""
        if pcd is None: return
        matrix = actor.GetUserMatrix()
        if matrix is None: return

        transform = np.eye(4)
        for i in range(4):
            for j in range(4):
                transform[i, j] = matrix.GetElement(i, j)

        pcd.transform(transform)
        actor.SetUserMatrix(None) # 重置矩阵，因为点坐标已经变了

    def _update_cut_points_all(self):
        """辅助函数：更新所有静态切面点"""
        self._update_cut_points_xoy()
        self._update_cut_points_xoz()
        self._update_cut_points_yoz()

class CutplaneVisualizerWidget(BaseWidget):
    """三正交切片可视化窗口 - 使用VTK显示xoy、xoz、yoz三个平面的2D切片"""
    def __init__(self, parent=None):
        super().__init__(visualizer=None, parent=parent)
        self.parent_frame = None
        self.result_widget = None
        self.ui_manager = None  # 外部UI管理器的引用
        self.current_plane = 'xoy'# 当前显示的切面类型：'xoy', 'xoz', 'yoz'
        self._camera_initialized_for_plane = False
        self.slice_interactor_style = SlicePanInteractorStyle()
        self.interactor.SetInteractorStyle(self.slice_interactor_style)
        QtCore.QTimer.singleShot(0, self._activate_slice_interactor_style)

        # 三个切面的位置
        self.cut_plane_x = 0.0  # yoz平面（x轴切面）
        self.cut_plane_y = 0.0  # xoz平面（y轴切面）
        self.cut_plane_z = 0.0  # xoy平面（z轴切面）

        # 初始化切面显示
        self._init_cut_plane_display()
        self._init_scale_bar()

    def _activate_slice_interactor_style(self):
        self.interactor.SetInteractorStyle(self.slice_interactor_style)

    def _init_cut_plane_display(self):
        """初始化切面显示（VTK）"""
        # 清理单点云actor，切面窗口只显示切面点云
        self.renderer.RemoveActor(self._single_actor)

        # 切面点云mapper和actor（用于显示当前切面的点）
        self._cut_points_src_mapper = vtk.vtkPolyDataMapper()
        self._cut_points_src_actor = vtk.vtkActor()
        self._cut_points_src_actor.SetMapper(self._cut_points_src_mapper)
        self._cut_points_src_actor.GetProperty().SetPointSize(3.0)
        self._cut_points_src_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 红色点

        self._cut_points_tgt_mapper = vtk.vtkPolyDataMapper()
        self._cut_points_tgt_actor = vtk.vtkActor()
        self._cut_points_tgt_actor.SetMapper(self._cut_points_tgt_mapper)
        self._cut_points_tgt_actor.GetProperty().SetPointSize(3.0)
        self._cut_points_tgt_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # 蓝色点

        self.renderer.AddActor(self._cut_points_src_actor)
        self.renderer.AddActor(self._cut_points_tgt_actor)

    def _init_scale_bar(self):
        """Add a 2D scale bar that follows camera zoom in the slice view."""
        self._scale_bar_target_px = 120
        self._scale_bar_margin = 28

        self._scale_bar_poly = vtk.vtkPolyData()
        self._scale_bar_points = vtk.vtkPoints()
        self._scale_bar_lines = vtk.vtkCellArray()
        self._scale_bar_poly.SetPoints(self._scale_bar_points)
        self._scale_bar_poly.SetLines(self._scale_bar_lines)

        display_coord = vtk.vtkCoordinate()
        display_coord.SetCoordinateSystemToDisplay()

        self._scale_bar_mapper = vtk.vtkPolyDataMapper2D()
        self._scale_bar_mapper.SetInputData(self._scale_bar_poly)
        self._scale_bar_mapper.SetTransformCoordinate(display_coord)

        self._scale_bar_actor = vtk.vtkActor2D()
        self._scale_bar_actor.SetMapper(self._scale_bar_mapper)
        self._scale_bar_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self._scale_bar_actor.GetProperty().SetLineWidth(3.0)

        self._scale_bar_text_mapper = vtk.vtkTextMapper()
        text_prop = self._scale_bar_text_mapper.GetTextProperty()
        text_prop.SetColor(1.0, 1.0, 1.0)
        text_prop.SetFontSize(14)
        text_prop.SetBold(True)
        text_prop.SetJustificationToCentered()
        text_prop.SetVerticalJustificationToBottom()

        self._scale_bar_text_actor = vtk.vtkActor2D()
        self._scale_bar_text_actor.SetMapper(self._scale_bar_text_mapper)

        self.renderer.AddActor2D(self._scale_bar_actor)
        self.renderer.AddActor2D(self._scale_bar_text_actor)

        camera = self.renderer.GetActiveCamera()
        camera.AddObserver("ModifiedEvent", lambda obj, event: self._update_scale_bar())
        self.interactor.AddObserver("MouseWheelForwardEvent", lambda obj, event: self._update_scale_bar())
        self.interactor.AddObserver("MouseWheelBackwardEvent", lambda obj, event: self._update_scale_bar())
        self.interactor.AddObserver("InteractionEvent", lambda obj, event: self._update_scale_bar())

    def _before_render(self):
        self._update_scale_bar()

    def _nice_scale_value(self, value):
        if value <= 0:
            return 0
        exponent = np.floor(np.log10(value))
        fraction = value / (10 ** exponent)
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 3.5:
            nice_fraction = 2
        elif fraction < 7.5:
            nice_fraction = 5
        else:
            nice_fraction = 10
        return nice_fraction * (10 ** exponent)

    def _format_scale_label(self, value):
        if value >= 10:
            return f"{value:.0f} mm"
        if value >= 1:
            return f"{value:.1f} mm"
        return f"{value:.2f} mm"

    def _world_units_per_pixel(self):
        rw = self.vtk_widget.GetRenderWindow() if self.vtk_widget is not None else None
        if rw is None:
            return None
        width, height = rw.GetSize()
        if height <= 0:
            return None

        camera = self.renderer.GetActiveCamera()
        if camera.GetParallelProjection():
            return (2.0 * camera.GetParallelScale()) / float(height)

        view_angle = np.deg2rad(camera.GetViewAngle())
        distance = camera.GetDistance()
        visible_height = 2.0 * distance * np.tan(view_angle / 2.0)
        return visible_height / float(height)

    def _update_scale_bar(self):
        world_per_pixel = self._world_units_per_pixel()
        if world_per_pixel is None or world_per_pixel <= 0:
            return

        rw = self.vtk_widget.GetRenderWindow() if self.vtk_widget is not None else None
        width, height = rw.GetSize()
        target_world = world_per_pixel * self._scale_bar_target_px
        scale_world = self._nice_scale_value(target_world)
        if scale_world <= 0:
            return

        scale_px = max(40, min(180, int(round(scale_world / world_per_pixel))))
        x1 = max(self._scale_bar_margin, width - self._scale_bar_margin - scale_px)
        x2 = x1 + scale_px
        y = self._scale_bar_margin
        tick = 8

        self._scale_bar_points.Reset()
        for point in ((x1, y, 0), (x2, y, 0), (x1, y - tick, 0), (x1, y + tick, 0),
                      (x2, y - tick, 0), (x2, y + tick, 0)):
            self._scale_bar_points.InsertNextPoint(*point)

        self._scale_bar_lines.Reset()
        for line in ((0, 1), (2, 3), (4, 5)):
            vtk_line = vtk.vtkLine()
            vtk_line.GetPointIds().SetId(0, line[0])
            vtk_line.GetPointIds().SetId(1, line[1])
            self._scale_bar_lines.InsertNextCell(vtk_line)

        self._scale_bar_poly.Modified()
        self._scale_bar_text_mapper.SetInput(self._format_scale_label(scale_world))
        self._scale_bar_text_actor.SetPosition((x1 + x2) / 2.0, y + 10)

    def switch_plane(self, plane_type):
        """切换显示的切面（由外部UI调用）"""
        self.current_plane = plane_type
        self._camera_initialized_for_plane = False

        # 更新显示
        self.update_current_slice()

    def get_point_cloud_bounds(self):
        """获取点云的边界范围（考虑源点云和目标点云）"""
        if self.result_widget is None:
            return None

        # 收集所有点云的点
        all_points = []

        if self.result_widget.source_pcd is not None:
            src_points = np.asarray(self.result_widget.source_pcd.points)
            if src_points.size > 0:
                all_points.append(src_points)

        if self.result_widget.target_pcd is not None:
            tgt_points = np.asarray(self.result_widget.target_pcd.points)
            if tgt_points.size > 0:
                all_points.append(tgt_points)

        if len(all_points) == 0:
            return None

        # 合并所有点
        combined_points = np.vstack(all_points)

        return {
            'x': (np.min(combined_points[:, 0]), np.max(combined_points[:, 0])),
            'y': (np.min(combined_points[:, 1]), np.max(combined_points[:, 1])),
            'z': (np.min(combined_points[:, 2]), np.max(combined_points[:, 2]))
        }

    def _extract_cut_points_vtk(self, pcd, plane_type, position, epsilon=1.0):
        """从点云中提取位于指定切平面附近的点，返回VTK PolyData"""
        if pcd is None:
            return vtk.vtkPolyData()
        points_np = np.asarray(pcd.points)
        if points_np.size == 0:
            return vtk.vtkPolyData()

        # 根据切面类型提取点
        if plane_type == 'xoy':  # z轴切面
            mask = np.abs(points_np[:, 2] - position) < epsilon
        elif plane_type == 'xoz':  # y轴切面
            mask = np.abs(points_np[:, 1] - position) < epsilon
        elif plane_type == 'yoz':  # x轴切面
            mask = np.abs(points_np[:, 0] - position) < epsilon
        else:
            return vtk.vtkPolyData()

        cut_points_np = points_np[mask]
        if cut_points_np.size == 0:
            return vtk.vtkPolyData()

        colors_np = np.asarray(pcd.colors)[mask] if pcd.has_colors() else None
        return _points_to_vtk_polydata(cut_points_np, colors_np)

    def update_current_slice(self):
        """更新当前切面的显示"""
        if self.result_widget is None:
            return

        # 如果在拖动锁定模式下，不要执行这个常规更新，以免覆盖动态更新
        if hasattr(self.result_widget, 'cut_planes_locked') and self.result_widget.cut_planes_locked:
            return

        source_pcd = self.result_widget.source_pcd
        target_pcd = self.result_widget.target_pcd

        if source_pcd is None and target_pcd is None:
            return

        epsilon = 1.0

        # 根据当前切面类型更新显示
        if self.current_plane == 'xoy':
            position = self.result_widget.cut_plane_z if self.result_widget else self.cut_plane_z
        elif self.current_plane == 'xoz':
            position = self.result_widget.cut_plane_y if self.result_widget else self.cut_plane_y
        elif self.current_plane == 'yoz':
            position = self.result_widget.cut_plane_x if self.result_widget else self.cut_plane_x
        else:
            return

        # 更新源点云切面点
        if source_pcd is not None:
            src_poly = self._extract_cut_points_vtk(source_pcd, self.current_plane, position, epsilon)
            self._cut_points_src_mapper.SetInputData(src_poly)

        # 更新目标点云切面点
        if target_pcd is not None:
            tgt_poly = self._extract_cut_points_vtk(target_pcd, self.current_plane, position, epsilon)
            self._cut_points_tgt_mapper.SetInputData(tgt_poly)

        # 设置相机视角以适应切面
        if not self._camera_initialized_for_plane:
            self._setup_camera_for_plane()
            self._camera_initialized_for_plane = True
        self._render()

    def update_dynamic_slice(self, source_poly, target_poly):
        """Update Frame 4 from locked-plane dragging without rebuilding from PCD data."""
        if source_poly is not None:
            self._cut_points_src_mapper.SetInputData(source_poly)
        if target_poly is not None:
            self._cut_points_tgt_mapper.SetInputData(target_poly)
        self._render()

    def _setup_camera_for_plane(self):
        """根据当前切面类型设置相机视角"""
        camera = self.renderer.GetActiveCamera()
        if self.current_plane == 'xoy':
            # XOY平面：从Z轴正方向看
            camera.SetPosition(0, 0, 1000)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 1, 0)
        elif self.current_plane == 'xoz':
            # XOZ平面：从Y轴正方向看
            camera.SetPosition(0, 1000, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
        elif self.current_plane == 'yoz':
            # YOZ平面：从X轴正方向看
            camera.SetPosition(1000, 0, 0)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
        camera.ParallelProjectionOn()
        self.renderer.ResetCamera()

    def on_cut_plane_slider_change(self, value):
        """滑动条值变化时更新切面位置（由外部UI调用）"""
        if self.result_widget is None:
            return
        if hasattr(self.result_widget, 'cut_planes_locked') and self.result_widget.cut_planes_locked:
            return

        bounds = self.get_point_cloud_bounds()
        if bounds is None:
            return

        # 根据当前切面类型更新对应的位置
        if self.current_plane == 'xoy':
            z_min, z_max = bounds['z']
            z_value = z_min + (z_max - z_min) * (value / 100.0)
            self.cut_plane_z = z_value
            if hasattr(self.result_widget, 'set_cut_plane_z'):
                self.result_widget.set_cut_plane_z(z_value)
        elif self.current_plane == 'xoz':
            y_min, y_max = bounds['y']
            y_value = y_min + (y_max - y_min) * (value / 100.0)
            self.cut_plane_y = y_value
            if hasattr(self.result_widget, 'set_cut_plane_y'):
                self.result_widget.set_cut_plane_y(y_value)
        elif self.current_plane == 'yoz':
            x_min, x_max = bounds['x']
            x_value = x_min + (x_max - x_min) * (value / 100.0)
            self.cut_plane_x = x_value
            if hasattr(self.result_widget, 'set_cut_plane_x'):
                self.result_widget.set_cut_plane_x(x_value)

        self.update_current_slice()

    def on_point_cloud_loaded(self):
        """当点云加载后调用，初始化切面显示"""
        if self.result_widget is None:
            return
        self._camera_initialized_for_plane = False

        # 从result_widget同步切面位置
        if hasattr(self.result_widget, 'cut_plane_x'):
            self.cut_plane_x = self.result_widget.cut_plane_x
        if hasattr(self.result_widget, 'cut_plane_y'):
            self.cut_plane_y = self.result_widget.cut_plane_y
        if hasattr(self.result_widget, 'cut_plane_z'):
            self.cut_plane_z = self.result_widget.cut_plane_z

        # 更新显示
        self.update_current_slice()

        # 通知外部UI组件更新
        if hasattr(self, 'ui_manager') and self.ui_manager is not None and hasattr(self.ui_manager, 'update_ui'):
            self.ui_manager.update_ui()

    def update_slices(self):
        """更新切面显示（兼容旧接口）"""
        self.update_current_slice()
