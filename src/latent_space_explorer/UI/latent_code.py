import numpy as np

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap

from latent_space_explorer.app_manager import AppManager
from latent_space_explorer.utils.widget_utils import code_to_img

class CodeImgWidget(QLabel):
	'''Class for widget visualizing latent code as image'''
	code_img_clicked = pyqtSignal(int, int)

	def __init__(self, app_manager : AppManager):
		super().__init__()

		self.code_img_clicked.connect(app_manager.code_click)
		app_manager.code_img_changed.connect(self.update_widget_img)

		self.img = code_to_img(app_manager.active_shape_codes[0])
		self.img_width = self.img.shape[0]
		self.q_pixmap = None

	def update_widget_img(self, img : np.array) -> None:
		if img is None:
			return
		
		q_img = QImage(img.data, img.shape[1], img.shape[0], 8, QImage.Format_Grayscale8)
		self.q_pixmap = QPixmap.fromImage(q_img)
		self.scale_set_pixmap()

	def scale_set_pixmap(self) -> None:
		if self.q_pixmap is None:
			return

		self.q_pixmap = self.q_pixmap.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.FastTransformation) # Stretch and get nearest value 
		self.setPixmap(self.q_pixmap)

	def resizeEvent(self, event) -> None:
		self.scale_set_pixmap()
		# QLabel.resizeEvent(self, event)

	def mousePressEvent(self, event) -> None:
		# Calculate code value index
		x_frac = event.x() / self.width()
		y_frac = event.y() / self.height()
		row = np.floor(y_frac * self.img_width)
		column = np.floor(x_frac * self.img_width)
		latent_value_index = int(row * self.img_width + column)

		# Instruct app_manager to update the latent code
		self.code_img_clicked.emit(latent_value_index, event.button())