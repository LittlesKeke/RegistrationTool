## main.py
import sys
from Ui_registrationWindow import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QObject


class RegistrationMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 创建UI实例
        self.ui = Ui_MainWindow()
        # 设置UI到当前窗口
        self.ui.setupUi(self)

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 调用UI类中的资源清理方法
        self.ui.closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 创建主窗口实例
    window = RegistrationMainWindow()
    window.show()

    sys.exit(app.exec_())
