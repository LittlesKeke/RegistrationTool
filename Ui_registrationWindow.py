## Ui_registrationWindow.py
import os, copy, time, tempfile
import numpy as np
import open3d as o3d
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QAction
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from visulizer import InputVisualizerWidget,OutputVisualizerWidget,CutplaneVisualizerWidget
from manualRegistration import *
from local_registration import CHECKPOINT_PATH, register_point_clouds




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # 时间统计变量
        self.local_compute_time = 0.0
        # 全局变量
        self.registration_matrices = []  # 全局配准矩阵变量
        self.source_pcd_file = None      # 源PCD文件路径
        self.target_pcd_file = None      # 目标PCD文件路径
        self.registration_thread = None
        self.registration_worker = None
        self.original_source_pcd = None
        self.original_target_pcd = None
        self.pending_auto_pre_transform = None
        self.cut_plane_ui = None  # 切面可视化UI管理器
        #self.i = 0
        # 窗口
        self.setup_main_window(MainWindow)
        self.create_widgets(MainWindow)
        self.create_menu_bar(MainWindow)
        self.retranslateUi(MainWindow)

    def setup_main_window(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1680, 1040)
        MainWindow.setMinimumSize(1180, 760)
        MainWindow.setWindowTitle("MedReg Studio")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)
        self.centralwidget.setObjectName("workspaceRoot")
        self.centralwidget.setStyleSheet("""
            QWidget#workspaceRoot { background: #edf2f5; color: #263746; }
            QFrame#topBar, QFrame#toolPanel, QFrame#viewCard {
                background: #ffffff; border: 1px solid #d8e1e7; border-radius: 8px;
            }
            QLabel#appTitle { color: #172b3a; font-size: 20px; font-weight: 700; }
            QLabel#mutedText { color: #718391; font-size: 11px; }
            QLabel#viewBadge { color: #43616f; background: #eef4f6; border: 1px solid #dbe5e9;
                border-radius: 4px; padding: 3px 7px; font-size: 10px; font-weight: 700; }
            QPushButton { background: #f4f7f9; color: #2b414f; border: 1px solid #d4dfe5;
                border-radius: 5px; padding: 8px 10px; min-height: 20px; font-size: 12px; }
            QPushButton:hover { background: #e8f0f3; border-color: #a9c0ca; }
            QPushButton:pressed { background: #dce9ee; }
            QPushButton#primaryAction { background: #147d83; color: #ffffff; border-color: #147d83;
                font-weight: 700; min-height: 32px; }
            QPushButton#primaryAction:hover { background: #106d73; }
            QPushButton#secondaryAction { background: #e8f2f3; color: #17656a; border-color: #c7dcde; }
            QPushButton:disabled { color: #9aa9b2; background: #f2f4f5; border-color: #e0e5e8; }
            QCheckBox { color: #526875; font-size: 11px; spacing: 6px; }
            QCheckBox::indicator { width: 15px; height: 15px; }
            QSlider::groove:horizontal { background: #dce5e9; height: 5px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #16848a; border: 1px solid #ffffff;
                width: 15px; margin: -6px 0; border-radius: 7px; }
            QStatusBar { background: #ffffff; color: #526875; border-top: 1px solid #d8e1e7; }
            QMenuBar { background: #ffffff; color: #334b59; border-bottom: 1px solid #d8e1e7; }
            QMenuBar::item { padding: 7px 12px; }
            QMenuBar::item:selected, QMenu::item:selected { background: #e8f2f3; color: #17656a; }
            QMenu { background: #ffffff; color: #334b59; border: 1px solid #d8e1e7; }
            QMenu::item { padding: 7px 28px 7px 14px; }
        """)
        self.root_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.root_layout.setContentsMargins(12, 10, 12, 8)
        self.root_layout.setSpacing(10)
        self.header = self._create_header()
        self.root_layout.addWidget(self.header)
        self.content_layout = QtWidgets.QHBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)
        self.root_layout.addLayout(self.content_layout, 1)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        self.statusbar.setSizeGripEnabled(False)
        self.statusbar.setStyleSheet("QStatusBar{background:#ffffff;color:#526875;border-top:1px solid #d8e1e7;padding:4px 10px;}")
        self.statusbar.showMessage("就绪 · 请导入源点云和参考点云")

    def _create_header(self):
        bar = QtWidgets.QFrame(self.centralwidget)
        bar.setObjectName("topBar")
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)
        brand = QtWidgets.QLabel()
        brand.setAlignment(QtCore.Qt.AlignCenter)
        brand.setFixedSize(38, 38)
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo1.png")
        if os.path.exists(logo_path):
            brand.setPixmap(QtGui.QPixmap(logo_path).scaled(38, 38, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            brand.setText("MR")
            brand.setStyleSheet("background:#dceff0;color:#116f75;border-radius:7px;font-size:15px;font-weight:700;")
        title_col = QtWidgets.QVBoxLayout()
        title_col.setSpacing(1)
        title = QtWidgets.QLabel("MedReg Studio")
        title.setObjectName("appTitle")
        subtitle = QtWidgets.QLabel("医学影像配准工作站  ·  Point cloud registration")
        subtitle.setObjectName("mutedText")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        layout.addWidget(brand)
        layout.addLayout(title_col)
        layout.addStretch(1)
        self.header_status = QtWidgets.QLabel("●  待机")
        self.header_status.setObjectName("viewBadge")
        layout.addWidget(self.header_status)
        self.header_model = QtWidgets.QLabel("MPG · 10步推理 · GICP精修")
        self.header_model.setObjectName("mutedText")
        layout.addWidget(self.header_model)
        return bar

    def create_widgets(self, MainWindow):
        self.o3d_widget_src = InputVisualizerWidget(MainWindow)
        self.o3d_widget_tgt = InputVisualizerWidget(MainWindow)
        self.o3d_widget_result = OutputVisualizerWidget(MainWindow)
        self.o3d_widget_cut_plane = CutplaneVisualizerWidget(MainWindow)
        self.workspace = QtWidgets.QWidget(self.centralwidget)
        self.workspace_layout = QtWidgets.QGridLayout(self.workspace)
        self.workspace_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_layout.setSpacing(10)

        self.frame_src = self.create_view_card("源点云 · 结构光", "SOURCE", self.o3d_widget_src)
        self.frame_tgt = self.create_view_card("目标点云 · MRI参考", "TARGET", self.o3d_widget_tgt)
        self.frame_result = self.create_view_card("配准结果", "OVERLAY", self.o3d_widget_result)
        self.frame_cut = self.create_view_card("正交切面", "SLICE", self.o3d_widget_cut_plane)
        self.frame = self.frame_tgt
        self.frame_2 = self.frame_src
        self.frame_3 = self.frame_result
        self.frame_4 = self.frame_cut
        self.label_src = self.frame_src.findChild(QtWidgets.QLabel, "viewTitle")
        self.label_tgt = self.frame_tgt.findChild(QtWidgets.QLabel, "viewTitle")
        self.label_result = self.frame_result.findChild(QtWidgets.QLabel, "viewTitle")

        self.workspace_layout.addWidget(self.frame_src, 0, 0)
        self.workspace_layout.addWidget(self.frame_tgt, 0, 1)
        self.workspace_layout.addWidget(self.frame_result, 1, 0)
        self.workspace_layout.addWidget(self.frame_cut, 1, 1)
        self.workspace_layout.setRowStretch(0, 1)
        self.workspace_layout.setRowStretch(1, 1)
        self.workspace_layout.setColumnStretch(0, 1)
        self.workspace_layout.setColumnStretch(1, 1)

        self.tool_panel = QtWidgets.QFrame(self.centralwidget)
        self.tool_panel.setObjectName("toolPanel")
        self.tool_panel.setMinimumWidth(245)
        self.tool_panel.setMaximumWidth(300)
        self.tool_panel_layout = QtWidgets.QVBoxLayout(self.tool_panel)
        self.tool_panel_layout.setContentsMargins(14, 16, 14, 14)
        self.tool_panel_layout.setSpacing(10)
        self.content_layout.addWidget(self.tool_panel)
        self.content_layout.addWidget(self.workspace, 1)

        self.create_frame_labels()
        self.create_button()
        self.set_slice_lock_enabled(False)
        self.cut_plane_ui = CutPlaneUI(self.frame_cut, self.o3d_widget_cut_plane, self.o3d_widget_result)
        self.o3d_widget_cut_plane.result_widget = self.o3d_widget_result
        self.o3d_widget_cut_plane.ui_manager = self.cut_plane_ui
        self.o3d_widget_result.set_cut_plane_widget(self.o3d_widget_cut_plane)
        self.create_time_stats_widget()
        self.create_footer_branding()
        self.tool_panel_layout.addStretch(1)
        self.update_time_stats()

    def create_view_card(self, title_text, badge_text, widget):
        card = QtWidgets.QFrame(self.workspace)
        card.setObjectName("viewCard")
        card.setMinimumSize(280, 230)
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(8, 7, 8, 8)
        layout.setSpacing(6)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(title_text)
        title.setObjectName("viewTitle")
        title.setStyleSheet("color:#273d4b;font-size:12px;font-weight:700;padding-left:3px;")
        badge = QtWidgets.QLabel(badge_text)
        badge.setObjectName("viewBadge")
        header.addWidget(title)
        header.addStretch(1)
        if widget is self.o3d_widget_result:
            self.slice_lock_switch = QtWidgets.QCheckBox("锁定切面")
            self.slice_lock_switch.setChecked(False)
            self.slice_lock_switch.setEnabled(False)
            self.slice_lock_switch.stateChanged.connect(self.on_slice_lock_toggled)
            header.addWidget(self.slice_lock_switch)
        header.addWidget(badge)
        layout.addLayout(header)
        widget.setParent(card)
        widget.setMinimumSize(220, 170)
        layout.addWidget(widget, 1)
        return card

    def create_frame(self, x, y, width, height, name):
        frame = QtWidgets.QFrame(self.centralwidget)
        frame.setObjectName(name)
        return frame

    def create_frame_labels(self):
        self.label_src.setText("源点云 · 结构光")
        self.label_tgt.setText("目标点云 · MRI参考")
        self.label_result.setText("配准结果")

    def set_slice_lock_enabled(self, enabled: bool):
        """控制切面锁定开关是否可用"""
        if hasattr(self, 'slice_lock_switch') and self.slice_lock_switch is not None:
            self.slice_lock_switch.setEnabled(enabled)
            if not enabled:
                # 关闭可用性时同时取消勾选
                self.slice_lock_switch.setChecked(False)
        if not enabled:
            # 确保结果窗口与切面UI恢复可调状态
            if hasattr(self, 'o3d_widget_result') and self.o3d_widget_result is not None:
                self.o3d_widget_result.set_cut_plane_lock(False)
            if hasattr(self, 'cut_plane_ui') and self.cut_plane_ui is not None:
                self.cut_plane_ui.set_lock_state(False)

    def on_slice_lock_toggled(self, state):
        """切面锁定开关切换"""
        if hasattr(self, 'slice_lock_switch') and self.slice_lock_switch is not None:
            if not self.slice_lock_switch.isEnabled():
                return
        locked = state == QtCore.Qt.Checked
        if hasattr(self, 'o3d_widget_result') and self.o3d_widget_result is not None:
            self.o3d_widget_result.set_cut_plane_lock(locked)
        if hasattr(self, 'cut_plane_ui') and self.cut_plane_ui is not None:
            self.cut_plane_ui.set_lock_state(locked)
        if hasattr(self, 'o3d_widget_cut_plane') and self.o3d_widget_cut_plane is not None:
            # 切换锁定状态后刷新切面显示
            self.o3d_widget_cut_plane.on_point_cloud_loaded()
        if locked:
            self.statusbar.showMessage("切面已锁定", 5000)
        else:
            self.statusbar.showMessage("切面已解锁", 3000)

    def create_menu_bar(self, MainWindow):
        self.menubar = MainWindow.menuBar()
        self.menubar.setStyleSheet("QMenuBar{background:#ffffff;color:#334b59;border-bottom:1px solid #d8e1e7;} QMenuBar::item{padding:7px 12px;} QMenuBar::item:selected{background:#e8f2f3;color:#17656a;} QMenu{background:#ffffff;color:#334b59;border:1px solid #d8e1e7;} QMenu::item{padding:7px 28px 7px 14px;} QMenu::item:selected{background:#e8f2f3;color:#17656a;}")
        file_menu = self.menubar.addMenu("文件")
        source_action = QAction("导入结构光源点云…", MainWindow)
        target_action = QAction("导入MRI参考点云…", MainWindow)
        export_action = QAction("导出变换矩阵…", MainWindow)
        file_menu.addAction(source_action)
        file_menu.addAction(target_action)
        file_menu.addSeparator()
        file_menu.addAction(export_action)
        source_action.triggered.connect(self.load_point_cloud_src)
        target_action.triggered.connect(self.load_point_cloud_tgt)
        export_action.triggered.connect(self.save_registration_matrix)

        registration_menu = self.menubar.addMenu("配准")
        auto_action = QAction("自动配准 · MPG + GICP", MainWindow)
        manual_action = QAction("手动对应点 / GICP…", MainWindow)
        registration_menu.addAction(auto_action)
        registration_menu.addAction(manual_action)
        auto_action.triggered.connect(self.run_auto_registration)
        manual_action.triggered.connect(self.run_manual_registration)

        view_menu = self.menubar.addMenu("视图")
        error_action = QAction("显示最近邻误差热图", MainWindow)
        error_action.triggered.connect(self.display_info)
        view_menu.addAction(error_action)

    def create_button(self):
        self._add_panel_heading("数据输入  ·  INPUT")
        self.import_laser_button = QtWidgets.QPushButton("导入结构光源点云…")
        self.import_mri_button = QtWidgets.QPushButton("导入MRI参考点云…")
        self.import_laser_button.clicked.connect(self.load_point_cloud_src)
        self.import_mri_button.clicked.connect(self.load_point_cloud_tgt)
        self.tool_panel_layout.addWidget(self.import_laser_button)
        self.tool_panel_layout.addWidget(self.import_mri_button)
        self.source_file_label = self._add_file_label("源点云 · 未导入")
        self.target_file_label = self._add_file_label("参考点云 · 未导入")

        self._add_panel_heading("配准  ·  REGISTRATION")
        self.auto_registration_button = QtWidgets.QPushButton("自动配准 · MPG + GICP")
        self.auto_registration_button.setObjectName("primaryAction")
        self.auto_registration_button.setEnabled(False)
        self.auto_registration_button.clicked.connect(self.run_auto_registration)
        self.manual_button = QtWidgets.QPushButton("手动对应点配准…")
        self.manual_button.clicked.connect(self.run_manual_registration)
        self.tool_panel_layout.addWidget(self.auto_registration_button)
        self.tool_panel_layout.addWidget(self.manual_button)

        self._add_panel_heading("分析与导出  ·  REVIEW")
        self.error_display_button = QtWidgets.QPushButton("显示误差热图")
        self.error_display_button.clicked.connect(self.display_info)
        self.save_result_button = QtWidgets.QPushButton("导出变换矩阵…")
        self.save_result_button.clicked.connect(self.save_registration_matrix)
        self.tool_panel_layout.addWidget(self.error_display_button)
        self.tool_panel_layout.addWidget(self.save_result_button)

        self.weights_label = QtWidgets.QLabel("MPG · FaceDLPRealV3 · FT30")
        self.weights_label.setObjectName("viewBadge")
        self.tool_panel_layout.addWidget(self.weights_label)

    def _add_panel_heading(self, text):
        label = QtWidgets.QLabel(text)
        label.setObjectName("mutedText")
        label.setStyleSheet("color:#617985;font-size:10px;font-weight:700;letter-spacing:1px;padding-top:7px;")
        self.tool_panel_layout.addWidget(label)
        return label

    def _add_file_label(self, text):
        label = QtWidgets.QLabel(text)
        label.setObjectName("mutedText")
        label.setWordWrap(True)
        label.setToolTip(text)
        label.setStyleSheet("padding:3px 5px;color:#718391;")
        self.tool_panel_layout.addWidget(label)
        return label

    def create_time_stats_widget(self):
        self._add_panel_heading("模型与耗时  ·  MODEL")
        self.time_stats_label = QtWidgets.QLabel("等待配准")
        self.time_stats_label.setObjectName("mutedText")
        self.time_stats_label.setWordWrap(True)
        self.time_stats_label.setMinimumHeight(44)
        self.tool_panel_layout.addWidget(self.time_stats_label)

    def create_footer_branding(self):
        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(0, 6, 0, 0)
        footer.setSpacing(8)
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo2.png")
        if os.path.exists(logo_path):
            logo = QtWidgets.QLabel()
            logo.setPixmap(QtGui.QPixmap(logo_path).scaledToHeight(28, QtCore.Qt.SmoothTransformation))
            footer.addWidget(logo, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        credit = QtWidgets.QLabel("南京医科大学\n医疗机器人创新实验室")
        credit.setObjectName("mutedText")
        credit.setStyleSheet("color:#84949d;font-size:9px;")
        footer.addWidget(credit, 1)
        self.tool_panel_layout.addLayout(footer)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("MedReg Studio · 医学影像点云配准")

    def load_point_cloud_tgt(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择目标PCD", "", "PCD Files (*.pcd)")
        if file_path:
            try:
                self.o3d_widget_tgt.load_point_cloud(file_path)
            except Exception as error:
                self.statusbar.showMessage(f"参考点云导入失败 · {error}", 8000)
                return
            if self.o3d_widget_tgt.get_origin_pcd() is None or not self.o3d_widget_tgt.get_origin_pcd().has_points():
                self.statusbar.showMessage("参考点云为空或无法读取")
                return
            self.target_pcd_file = file_path
            self.registration_matrices.clear()
            self.pending_auto_pre_transform = None
            count = len(self.o3d_widget_tgt.get_origin_pcd().points)
            self.target_file_label.setText(f"参考点云 · {os.path.basename(file_path)} · {count:,} 点")
            self.target_file_label.setToolTip(file_path)
            self.statusbar.showMessage(f"参考点云已导入 · {count:,} 点")
            self._update_ready_state()
            self.set_slice_lock_enabled(False)

    def load_point_cloud_src(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择源PCD", "", "PCD Files (*.pcd)")
        if file_path:
            try:
                self.o3d_widget_src.load_point_cloud(file_path)
            except Exception as error:
                self.statusbar.showMessage(f"源点云导入失败 · {error}", 8000)
                return
            if self.o3d_widget_src.get_origin_pcd() is None or not self.o3d_widget_src.get_origin_pcd().has_points():
                self.statusbar.showMessage("源点云为空或无法读取")
                return
            self.source_pcd_file = file_path
            self.registration_matrices.clear()
            self.pending_auto_pre_transform = None
            count = len(self.o3d_widget_src.get_origin_pcd().points)
            self.source_file_label.setText(f"源点云 · {os.path.basename(file_path)} · {count:,} 点")
            self.source_file_label.setToolTip(file_path)
            self.statusbar.showMessage(f"源点云已导入 · {count:,} 点")
            self._update_ready_state()
            self.set_slice_lock_enabled(False)

    def _update_ready_state(self):
        ready = bool(self.source_pcd_file and self.target_pcd_file)
        self.auto_registration_button.setEnabled(ready)
        if ready:
            self.header_status.setText("●  已就绪")
            self.header_status.setStyleSheet("color:#147d83;background:#e5f4f2;border:1px solid #c6e5df;border-radius:4px;padding:3px 7px;font-size:10px;font-weight:700;")
        else:
            self.header_status.setText("●  待导入")
            self.header_status.setStyleSheet("color:#718391;background:#eef4f6;border:1px solid #dbe5e9;border-radius:4px;padding:3px 7px;font-size:10px;font-weight:700;")

    def get_combined_registration_matrix(self):
        combined_matrix = np.eye(4)
        for matrix in self.registration_matrices:
            combined_matrix = np.dot(matrix, combined_matrix)
        return combined_matrix

    def create_random_clockwise_z_rotation(self):
        angle_degrees = np.random.uniform(45.0, 135.0)
        angle_radians = -np.deg2rad(angle_degrees)
        cos_value = np.cos(angle_radians)
        sin_value = np.sin(angle_radians)

        rotation_matrix = np.eye(4)
        rotation_matrix[0, 0] = cos_value
        rotation_matrix[0, 1] = -sin_value
        rotation_matrix[1, 0] = sin_value
        rotation_matrix[1, 1] = cos_value
        return rotation_matrix, angle_degrees

    def prepare_auto_registration_source(self):
        source_pcd = copy.deepcopy(self.o3d_widget_src.get_origin_pcd())
        if source_pcd is None:
            raise RuntimeError("Source point cloud is not loaded.")

        self.pending_auto_pre_transform = None

        if not self.registration_matrices:
            return self.source_pcd_file, source_pcd, False

        combined_matrix = self.get_combined_registration_matrix()
        source_pcd.transform(combined_matrix)
        pre_transform, angle_degrees = self.create_random_clockwise_z_rotation()
        source_pcd.transform(pre_transform)
        self.pending_auto_pre_transform = pre_transform
        self.statusbar.showMessage(f"继续自动配准：已先绕Z轴顺时针随机旋转 {angle_degrees:.1f}°")

        fd, temp_source_file = tempfile.mkstemp(prefix="auto_reg_source_", suffix=".pcd")
        os.close(fd)
        if not o3d.io.write_point_cloud(temp_source_file, source_pcd):
            raise RuntimeError(f"Failed to write temporary source point cloud: {temp_source_file}")

        return temp_source_file, source_pcd, True

    def run_manual_registration(self):
        """运行手动配准的主控制函数"""
        # 在新的配准流程开始前禁止切面锁定
        self.set_slice_lock_enabled(False)

        # 1. 检查点云是否已加载
        if self.o3d_widget_tgt.point_cloud is None or self.o3d_widget_src.point_cloud is None:
            msgbox = QtWidgets.QMessageBox(self.centralwidget)
            msgbox.setStyleSheet("background-color: white; color: black;")
            msgbox.setWindowTitle("文件缺失")
            msgbox.setText("请先导入源点云和目标点云！")
            msgbox.setIcon(QtWidgets.QMessageBox.Warning)
            msgbox.exec_()
            return

        # 从widget中获取原始点云的拷贝用于选点，目标点云为颜色更改后的
        source_pcd_orig = copy.deepcopy(self.o3d_widget_src.get_origin_pcd())
        target_pcd_orig = copy.deepcopy(self.o3d_widget_tgt.get_point_cloud())

        # 2. 检查是否为首次配准（需要手动选点）
        if self.o3d_widget_result.source_pcd is None:
            # 首次配准：需要用户手动选择对应点
            self.statusbar.showMessage("首次配准 · 请依次在源点云和参考点云中选择对应点")

            # 弹出窗口让用户选择点
            picked_id_source = pick_points(source_pcd_orig, window_title="请在 源点云(结构光) 中选择对应点")
            self.statusbar.showMessage("请在参考点云中按相同顺序选择对应点；Shift+左键选择，Shift+右键撤销，Q完成")
            picked_id_target = pick_points(target_pcd_orig, window_title="请在 目标点云(MRI) 中选择对应点")

            # 验证选择
            if len(picked_id_source) < 3 or len(picked_id_target) < 3:
                self.statusbar.showMessage("已取消 · 至少需要选择3组对应点")
                return

            if len(picked_id_source) != len(picked_id_target):
                self.statusbar.showMessage(f"对应点数量不一致 · 源 {len(picked_id_source)}，目标 {len(picked_id_target)}")
                return

            # 将选中的点ID保存到widget中，供manual_registration使用
            self.o3d_widget_src.selected_points = picked_id_source
            self.o3d_widget_tgt.selected_points = picked_id_target

            self.statusbar.showMessage("对应点已选择 · 正在执行 GICP")
        else:
            # 增量配准：基于已有配准结果继续优化
            self.statusbar.showMessage("增量精修 · 正在基于当前变换执行 GICP")

        # 3. 调用统一的配准函数
        try:
            start_time = time.time()
            matrix = manual_registration(self)
            end_time = time.time()
            self.local_compute_time = end_time - start_time
            self.update_registration_display(matrix)
            self.statusbar.showMessage("手动配准完成", 5000)
            self.update_time_stats()
        except Exception as e:
            self.statusbar.showMessage(f"配准失败：{str(e)}")
            msgbox = QtWidgets.QMessageBox(self.centralwidget)
            msgbox.setStyleSheet("background-color: white; color: black;")
            msgbox.setWindowTitle("配准错误")
            msgbox.setText(f"配准过程中发生错误：\n{str(e)}")
            msgbox.setIcon(QtWidgets.QMessageBox.Critical)
            msgbox.exec_()

    def run_auto_registration(self):
        # 自动配准前关闭切面锁定
        self.set_slice_lock_enabled(False)

        if not self.source_pcd_file or not self.target_pcd_file:
            msgbox = QtWidgets.QMessageBox(self.centralwidget)
            msgbox.setStyleSheet("background-color: white; color: black;")
            msgbox.setWindowTitle("文件缺失")
            msgbox.setText("请先导入源点云和目标点云！")
            msgbox.setIcon(QtWidgets.QMessageBox.Warning)
            msgbox.exec_()
            return

        self.statusbar.showMessage("正在准备本地 MPG 配准…")
        self.header_status.setText("●  配准中")
        self.auto_registration_button.setEnabled(False)

        # 获取原始点云用于显示
        try:
            source_file_for_registration, source_pcd_for_display, cleanup_source_file = self.prepare_auto_registration_source()
        except Exception as e:
            self.on_registration_error(str(e))
            return

        self.original_source_pcd = source_pcd_for_display
        self.original_target_pcd = copy.deepcopy(self.o3d_widget_tgt.get_origin_pcd())

        # 创建线程和worker
        self.registration_thread = QThread()
        self.registration_worker = RegistrationWorker(
            source_file=source_file_for_registration,
            target_file=self.target_pcd_file,
            checkpoint=CHECKPOINT_PATH,
            cleanup_source_file=cleanup_source_file
        )
        self.registration_worker.moveToThread(self.registration_thread)

        # 连接信号和槽
        self.registration_thread.started.connect(self.registration_worker.run)
        self.registration_worker.progress_update.connect(self.update_registration_display)
        self.registration_worker.finished.connect(self.on_registration_finished)
        self.registration_worker.error.connect(self.on_registration_error)
        self.registration_worker.status_update.connect(self.statusbar.showMessage)

        # 线程结束后进行清理
        self.registration_thread.finished.connect(self.registration_thread.deleteLater)

        # 启动线程
        self.registration_thread.start()

    def update_registration_display(self, transform_matrix):
        """接收来自worker的矩阵并更新UI"""
        source_to_update = copy.deepcopy(self.original_source_pcd)
        source_to_update.transform(transform_matrix)
        #np.savetxt(f"D:\Datas\DLP1.0\pre\\trans_{self.i}.txt", transform_matrix, delimiter=' ', fmt='%.6f')
        #self.i += 1
        self.o3d_widget_result.load_point_clouds(source_to_update, self.original_target_pcd)

        # 更新切面可视化窗口
        if hasattr(self, 'o3d_widget_cut_plane') and self.o3d_widget_cut_plane is not None:
            self.o3d_widget_cut_plane.on_point_cloud_loaded()

    def on_registration_finished(self, matrix, compute_seconds):
        """Run the existing local GICP refinement after MPG inference."""
        self.local_compute_time = compute_seconds
        self.statusbar.showMessage("MPG 完成 · 正在进行 GICP 精修…")
        self.update_registration_display(matrix) # 最后更新一次最精确的位置

        if self.pending_auto_pre_transform is not None:
            self.registration_matrices.append(self.pending_auto_pre_transform)
            self.pending_auto_pre_transform = None
        self.registration_matrices.append(matrix)

        self.registration_thread.quit() # 安全退出线程
        self.registration_thread.wait()
        self.original_source_pcd = copy.deepcopy(self.o3d_widget_src.get_origin_pcd())
        refine_started = time.perf_counter()
        refined_matrix = manual_registration(self)
        self.local_compute_time += time.perf_counter() - refine_started
        self.update_registration_display(refined_matrix)
        self.statusbar.showMessage("MPG + GICP 配准完成", 5000)
        self.header_status.setText("●  已配准")
        self.header_status.setStyleSheet("color:#147d83;background:#e5f4f2;border:1px solid #c6e5df;border-radius:4px;padding:3px 7px;font-size:10px;font-weight:700;")
        self.auto_registration_button.setEnabled(True)
        self.update_time_stats()

    def update_time_stats(self):
        """更新并显示时间统计信息"""
        time_stats = f"MPG + GICP: {self.local_compute_time:.2f} 秒"

        self.time_stats_label.setText(time_stats)

    def on_registration_error(self, error_message):
        self.pending_auto_pre_transform = None
        """处理配准过程中发生的错误"""
        msgbox = QtWidgets.QMessageBox(self.centralwidget)
        msgbox.setStyleSheet("background-color: white; color: black;")
        msgbox.setWindowTitle("配准错误")
        msgbox.setText(f"本地配准失败:\n{error_message}")
        msgbox.setIcon(QtWidgets.QMessageBox.Critical)
        msgbox.exec_()
        self.statusbar.showMessage(f"配准失败: {error_message}", 10000)
        self.header_status.setText("●  失败")
        self.header_status.setStyleSheet("color:#a23c43;background:#fbebec;border:1px solid #f0c8cb;border-radius:4px;padding:3px 7px;font-size:10px;font-weight:700;")
        self._update_ready_state()
        if self.registration_thread and self.registration_thread.isRunning():
            self.registration_thread.quit()
            self.registration_thread.wait()

    def display_info(self):
        # 检查结果窗口中是否有有效的点云
        if self.o3d_widget_result.source_pcd is None or self.o3d_widget_result.target_pcd is None:
            self.statusbar.showMessage("错误：结果窗口中未确定源点云或目标点云，无法计算误差。")
            msgbox = QtWidgets.QMessageBox(self.centralwidget)
            msgbox.setStyleSheet("background-color: white; color: black;")
            msgbox.setWindowTitle("错误")
            msgbox.setText("请先进行配准，再计算误差。")
            msgbox.setIcon(QtWidgets.QMessageBox.Warning)
            msgbox.exec_()
            return

        self.statusbar.showMessage("正在计算最近邻误差…")
        distances = np.asarray(self.o3d_widget_result.source_pcd.compute_point_cloud_distance(self.o3d_widget_result.target_pcd))
        self.max_distance = np.max(distances)
        clipped_distances = np.clip(distances, 0, 5)
        normalized_distances = clipped_distances / 5.0
        self.add_colorbar(self.centralwidget)
        colors = map_colors_based_on_distance(normalized_distances)
        self.o3d_widget_result.source_pcd.colors = o3d.utility.Vector3dVector(colors)
        self.o3d_widget_result.update_source()
        self.o3d_widget_result._update_cut_points_all()
        if hasattr(self, 'o3d_widget_cut_plane') and self.o3d_widget_cut_plane is not None:
            self.o3d_widget_cut_plane.update_current_slice()
        # 允许用户锁定切面
        self.set_slice_lock_enabled(True)
        self.statusbar.showMessage("误差热图已更新")

    def add_colorbar(self, parent_frame):
        if hasattr(self, 'colorbar_widget') and self.colorbar_widget is not None:
            self.colorbar_widget.show()
            return
        colorbar_widget = QtWidgets.QFrame(self.frame_result)
        colorbar_widget.setObjectName("errorScale")
        colorbar_widget.setStyleSheet("QFrame#errorScale { background:#f7f9fa; border:1px solid #e1e8ec; border-radius:4px; }")
        self.frame_result.layout().addWidget(colorbar_widget)
        self.colorbar_widget = colorbar_widget
        layout = QtWidgets.QVBoxLayout(colorbar_widget)
        layout.setContentsMargins(8, 3, 8, 2)
        fig = Figure(figsize=(6, 0.62), dpi=100)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.35)
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(58)
        layout.addWidget(canvas)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.04, 0.43, 0.92, 0.38])
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(vmin=0, vmax=5)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=ax, orientation='horizontal')
        cb.set_label('Nearest target distance (mm)', color='#405967')
        ax.tick_params(axis='x', colors='#526875')
        ticks = np.linspace(0, 5, 6)  # 0到最大值之间均匀分布6个刻度
        cb.set_ticks(ticks)
        cb.ax.set_xticklabels([f"{tick:.1f}" for tick in ticks], color='#526875')
        colorbar_widget.show()

    def save_registration_matrix(self):
        if not self.registration_matrices:
            msgbox = QtWidgets.QMessageBox(self.centralwidget)
            msgbox.setStyleSheet("background-color: white; color: black;")
            msgbox.setWindowTitle("提示")
            msgbox.setText("没有可保存的变换矩阵。请先进行配准。")
            msgbox.setIcon(QtWidgets.QMessageBox.Information)
            msgbox.exec_()
            self.statusbar.showMessage("没有可保存的变换矩阵。")
            return

        combined_matrix = self.get_combined_registration_matrix()

        # 创建自定义文件保存对话框
        dialog = QtWidgets.QFileDialog(self.centralwidget, "保存变换矩阵")
        dialog.setStyleSheet("""
            QDialog {
                background-color: white;
                color: black;
            }
            QLabel, QLineEdit, QComboBox, QPushButton {
                background-color: white;
                color: black;
            }
        """)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setNameFilter("Text Files (*.txt);;Numpy Files (*.npy)")
        dialog.selectFile("transformation_matrix.txt")  # 默认文件名和格式
        dialog.setDefaultSuffix("txt")  # 默认后缀

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            file_name = dialog.selectedFiles()[0]
            try:
                if file_name.endswith('.npy'):
                    np.save(file_name, combined_matrix.astype(dtype=np.float64))
                else: # 保存为txt
                    np.savetxt(file_name, combined_matrix)

                success_box = QtWidgets.QMessageBox(self.centralwidget)
                success_box.setStyleSheet("background-color: white; color: black;")
                success_box.setWindowTitle("成功")
                success_box.setText(f"矩阵已成功保存至\n{file_name}")
                success_box.setIcon(QtWidgets.QMessageBox.Information)
                success_box.exec_()
                self.statusbar.showMessage(f"矩阵已成功保存至 {file_name}")
            except Exception as e:
                error_box = QtWidgets.QMessageBox(self.centralwidget)
                error_box.setStyleSheet("background-color: white; color: black;")
                error_box.setWindowTitle("保存失败")
                error_box.setText(f"无法保存文件！\n错误: {e}")
                error_box.setIcon(QtWidgets.QMessageBox.Critical)
                error_box.exec_()
                self.statusbar.showMessage(f"保存矩阵失败: {e}")

    def closeEvent(self, event):
        # 首先停止并清理定时器，确保不再调用渲染更新
        if hasattr(self, 'render_timer'): # 确保定时器存在
            self.render_timer.stop()
            # 断开定时器的连接，防止进一步调用
            self.render_timer.timeout.disconnect()
            # 确保定时器被正确删除
            self.render_timer.deleteLater()

        # 逐个安全关闭widget
        self.o3d_widget_tgt.close()
        self.o3d_widget_src.close()
        self.o3d_widget_result.close()
        self.o3d_widget_cut_plane.close()

        # 直接接受事件，无需调用super()，因为这个类不是Qt窗口类的子类
        event.accept()


class CutPlaneUI:
    """切面可视化UI组件管理器 - 负责切面可视化的UI部分"""
    def __init__(self, parent_frame, vtk_widget, result_widget):
        self.parent_frame = parent_frame
        self.vtk_widget = vtk_widget  # CutplaneVisualizerWidget实例
        self.result_widget = result_widget

        # 初始化UI组件
        self._setup_ui()

    def _setup_ui(self):
        self.controls = QtWidgets.QFrame(self.parent_frame)
        control_layout = QtWidgets.QVBoxLayout(self.controls)
        control_layout.setContentsMargins(2, 2, 2, 2)
        control_layout.setSpacing(5)
        self.cut_plane_label = QtWidgets.QLabel("切面 · 轴位 / Z")
        self.cut_plane_label.setStyleSheet("color:#526875;font-size:10px;font-weight:700;")
        control_layout.addWidget(self.cut_plane_label)
        self._create_switch_buttons()
        self._create_sliders()
        if self.parent_frame.layout() is not None:
            self.parent_frame.layout().insertWidget(1, self.controls)

    def _create_switch_buttons(self):
        """创建切换切面的按钮"""
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)
        self.plane_group = QtWidgets.QButtonGroup(self.controls)
        self.plane_group.setExclusive(True)
        self.btn_xoy = QtWidgets.QPushButton("轴位 · Z")
        self.btn_xoy.setCheckable(True)
        self.btn_xoy.setChecked(True)
        self.btn_xoy.clicked.connect(lambda: self._switch_plane('xoy'))
        self.btn_xoz = QtWidgets.QPushButton("冠状位 · Y")
        self.btn_xoz.setCheckable(True)
        self.btn_xoz.clicked.connect(lambda: self._switch_plane('xoz'))
        self.btn_yoz = QtWidgets.QPushButton("矢状位 · X")
        self.btn_yoz.setCheckable(True)
        self.btn_yoz.clicked.connect(lambda: self._switch_plane('yoz'))
        for button in (self.btn_xoy, self.btn_xoz, self.btn_yoz):
            button.setMinimumHeight(27)
            button_layout.addWidget(button)
            self.plane_group.addButton(button)
        self.controls.layout().addLayout(button_layout)

    def _switch_plane(self, plane_type):
        """切换显示的切面"""
        # 更新按钮状态
        self.btn_xoy.setChecked(plane_type == 'xoy')
        self.btn_xoz.setChecked(plane_type == 'xoz')
        self.btn_yoz.setChecked(plane_type == 'yoz')

        # 更新标题
        plane_names = {'xoy': '切面 · 轴位 / Z', 'xoz': '切面 · 冠状位 / Y', 'yoz': '切面 · 矢状位 / X'}
        self.cut_plane_label.setText(plane_names[plane_type])

        # 通知VTK控件切换平面
        if hasattr(self.vtk_widget, 'switch_plane'):
            self.vtk_widget.switch_plane(plane_type)
        # 同步UI
        self.update_ui()

    def _create_sliders(self, parent_frame=None):
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._slider_label_base_text = "Position"
        self.slider_label = QtWidgets.QLabel(self._slider_label_base_text)
        self.slider_value_label = QtWidgets.QLabel("0.0")
        self.slider_value_label.setMinimumWidth(54)
        self.slider_value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.cut_plane_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.controls)
        self.cut_plane_slider.setMinimum(0)
        self.cut_plane_slider.setMaximum(100)
        self.cut_plane_slider.setValue(50)
        row.addWidget(self.slider_label)
        row.addWidget(self.cut_plane_slider, 1)
        row.addWidget(self.slider_value_label)
        self.controls.layout().addLayout(row)
        self.cut_plane_slider.valueChanged.connect(self._on_cut_plane_slider_change)

    def set_lock_state(self, locked: bool):
        if not hasattr(self, 'cut_plane_slider'):
            return
        self.cut_plane_slider.setEnabled(not locked)
        self.slider_label.setText("Position · Locked" if locked else "Position")
        self.slider_value_label.setStyleSheet(
            "color:#a36a12;font-weight:700;" if locked else "color:#526875;"
        )
        self.update_ui()

    def _on_cut_plane_slider_change(self, value):
        """滑动条值变化时更新切面位置"""
        if hasattr(self, 'cut_plane_slider') and not self.cut_plane_slider.isEnabled():
            return
        if hasattr(self.vtk_widget, 'on_cut_plane_slider_change'):
            self.vtk_widget.on_cut_plane_slider_change(value)
            # 根据当前平面类型显示正确的切面位置值
            if hasattr(self.vtk_widget, 'current_plane'):
                if self.vtk_widget.current_plane == 'xoy' and hasattr(self.vtk_widget, 'cut_plane_z'):
                    self.slider_value_label.setText(f"{self.vtk_widget.cut_plane_z:.2f}")
                elif self.vtk_widget.current_plane == 'xoz' and hasattr(self.vtk_widget, 'cut_plane_y'):
                    self.slider_value_label.setText(f"{self.vtk_widget.cut_plane_y:.2f}")
                elif self.vtk_widget.current_plane == 'yoz' and hasattr(self.vtk_widget, 'cut_plane_x'):
                    self.slider_value_label.setText(f"{self.vtk_widget.cut_plane_x:.2f}")

    def update_ui(self):
        """更新UI状态"""
        # 同步VTK控件的状态到UI
        if hasattr(self.vtk_widget, 'current_plane'):
            plane_type = self.vtk_widget.current_plane
            self.btn_xoy.setChecked(plane_type == 'xoy')
            self.btn_xoz.setChecked(plane_type == 'xoz')
            self.btn_yoz.setChecked(plane_type == 'yoz')

            # 更新标题
            plane_names = {'xoy': '切面 · 轴位 / Z', 'xoz': '切面 · 冠状位 / Y', 'yoz': '切面 · 矢状位 / X'}
            self.cut_plane_label.setText(plane_names[plane_type])

            # 更新滑动条和值显示，根据当前平面类型
            slider_block_state = None
            if hasattr(self.vtk_widget, 'get_point_cloud_bounds'):
                bounds = self.vtk_widget.get_point_cloud_bounds()
                if bounds is not None:
                    if hasattr(self, 'cut_plane_slider'):
                        slider_block_state = self.cut_plane_slider.blockSignals(True)
                    if plane_type == 'xoy' and hasattr(self.vtk_widget, 'cut_plane_z') and 'z' in bounds:
                        z_min, z_max = bounds['z']
                        if z_max > z_min:
                            slider_value = int((self.vtk_widget.cut_plane_z - z_min) / (z_max - z_min) * 100)
                            if hasattr(self, 'cut_plane_slider'):
                                self.cut_plane_slider.setValue(slider_value)
                            self.slider_value_label.setText(f"{self.vtk_widget.cut_plane_z:.2f}")
                    elif plane_type == 'xoz' and hasattr(self.vtk_widget, 'cut_plane_y') and 'y' in bounds:
                        y_min, y_max = bounds['y']
                        if y_max > y_min:
                            slider_value = int((self.vtk_widget.cut_plane_y - y_min) / (y_max - y_min) * 100)
                            if hasattr(self, 'cut_plane_slider'):
                                self.cut_plane_slider.setValue(slider_value)
                            self.slider_value_label.setText(f"{self.vtk_widget.cut_plane_y:.2f}")
                    elif plane_type == 'yoz' and hasattr(self.vtk_widget, 'cut_plane_x') and 'x' in bounds:
                        x_min, x_max = bounds['x']
                        if x_max > x_min:
                            slider_value = int((self.vtk_widget.cut_plane_x - x_min) / (x_max - x_min) * 100)
                            if hasattr(self, 'cut_plane_slider'):
                                self.cut_plane_slider.setValue(slider_value)
                            self.slider_value_label.setText(f"{self.vtk_widget.cut_plane_x:.2f}")
                    if slider_block_state is not None and hasattr(self, 'cut_plane_slider'):
                        self.cut_plane_slider.blockSignals(slider_block_state)


class RegistrationWorker(QObject):
    """Run local MPG inference away from the Qt event loop."""
    progress_update = pyqtSignal(np.ndarray)
    finished = pyqtSignal(np.ndarray, float)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, source_file, target_file, checkpoint, cleanup_source_file=False):
        super().__init__()
        self.source_file = source_file
        self.target_file = target_file
        self.checkpoint = checkpoint
        self.cleanup_source_file = cleanup_source_file

    def run(self):
        try:
            self.status_update.emit("正在载入本地 MPG 权重并推理...")
            matrix, elapsed = register_point_clouds(
                self.source_file, self.target_file, self.checkpoint,
                update_callback=self.progress_update.emit,
            )
            self.finished.emit(matrix, elapsed)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")

        finally:
            if self.cleanup_source_file:
                try:
                    os.remove(self.source_file)
                except OSError:
                    pass
