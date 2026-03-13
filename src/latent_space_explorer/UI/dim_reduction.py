from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPainter, QPixmap, QPen
from PyQt5.QtCore import pyqtSignal, Qt

from latent_space_explorer.app_manager import AppManager

class DimReductionWidget(QLabel):
    """Widget for displaying dimensionality reduction and calling app manager shape reconstruction with mouse click 2D coordinates in reduced dimensions"""
    clicked_dim_reduction = pyqtSignal(float, float, int) # pixel_x, pixel_y, button

    def __init__(self, app_manager : AppManager):
        super().__init__()

        app_manager.dim_reduction_changed.connect(self.update_dim_reduction_type)
        self.clicked_dim_reduction.connect(app_manager.reduced_dim_click)
        
        self.handle_size = 14
        self.left_click_pos, self.right_click_pos = None, None
        self.dim_reduction_plot = None
        
        self.set_dim_reduction_plot(app_manager.plot_paths[app_manager.dim_reduction_type])

    def set_dim_reduction_plot(self, image_path) -> QPixmap:
        self.dim_reduction_plot = QPixmap(image_path)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        black_pen = QPen(Qt.black, 8)
        white_pen = QPen(Qt.white, 4)

        painter.drawPixmap(self.rect(), self.dim_reduction_plot)
        for pen in [black_pen, white_pen]:
            painter.setPen(pen)
            if self.left_click_pos is not None:
                painter.drawEllipse(self.left_click_pos[0] - int(0.5 * self.handle_size), self.left_click_pos[1]  - int(0.5 * self.handle_size), self.handle_size, self.handle_size)
            if self.right_click_pos is not None:
                painter.drawEllipse(self.right_click_pos[0] - int(0.5 * self.handle_size), self.right_click_pos[1]  - int(0.5 * self.handle_size), self.handle_size, self.handle_size)

        painter.end()

    def mousePressEvent(self, event) -> None:
        self.set_clicked_pos(event.x(), event.y(), event.button())

        x_fract = event.pos().x() / self.width()
        y_fract = event.pos().y() / self.height()
        self.clicked_dim_reduction.emit(x_fract, y_fract, event.button()) # Call reconstruction method of app_manager

        self.update()

    def set_clicked_pos(self, x : int, y : int, button : int) -> None:
        if button == 1:
            self.left_click_pos = (x,y)
            self.right_click_pos = None
        elif button == 2:
            self.right_click_pos = (x,y)

    def update_dim_reduction_type(self, image_path : str) -> None:
        self.set_dim_reduction_plot(image_path)
        self.left_click_pos, self.right_click_pos = None, None # Remove handles
        self.update()


if __name__ == '__main__':
    pass