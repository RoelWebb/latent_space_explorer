import time

import numpy as np

import moderngl
from pyrr import Matrix44

from PyQt5 import QtCore, QtOpenGL

from latent_space_explorer.app_manager import AppManager
from latent_space_explorer.utils import IO


class MeshWidget(QtOpenGL.QGLWidget):
    def __init__(self, shape_path: str, app_manager: AppManager):
        super(MeshWidget, self).__init__()

        app_manager.rotation_speed_changed.connect(self.set_rotations_per_minute)

        self.base_color = app_manager.base_color
        self.shape_path = shape_path
        self.angle = 0
        self.rotations_per_minute = app_manager.rotations_per_minute
        self.last_timestamp = time.time()
        self.shaders_dir_path = app_manager.shaders_dir_path

        # Timer for updating/triggering to rerender
        self.trigger_timer = QtCore.QTimer(self)
        self.trigger_timer.timeout.connect(self.update_scene)
        self.trigger_timer.start(17)  # 60 fps corresponds to 16.7 milliseconds between frames

    def set_rotations_per_minute(self, rotations_per_minute: int) -> None:
        self.rotations_per_minute = rotations_per_minute

    # Updates render with new angle for shape rotation
    def update_scene(self) -> None:
        self.angle += (time.time() - self.last_timestamp) * self.rotations_per_minute * 2 * np.pi / 60.0

        self.last_timestamp = time.time()
        self.update()

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()

        self.prog_shape = IO.load_program(self.ctx,
                                          self.shaders_dir_path,
                                          vertex_shader_name='shape_visualisation_vertex_shader.glsl',
                                          fragment_shader_name='shape_visualisation_fragment_shader.glsl')
        self.mvp = self.prog_shape['Mvp']
        self.light = self.prog_shape['Light']

        self.vao_mesh = IO.mesh_to_vao(self.ctx, self.shape_path, self.prog_shape, 1.0)

    def paintGL(self) -> None:
        self.ctx.viewport = (0, 0, self.width(), self.height())
        # self.screen = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
        self.render_shape_setup(angle=self.angle)
        self.vao_mesh = IO.mesh_to_vao(self.ctx, self.shape_path, self.prog_shape, 1.0)
        if self.vao_mesh is not None:
            self.vao_mesh.render()

    def render_shape_setup(self,
                           angle: float,
                           camera_height: float = 0.4,
                           lookat_height: float = 0.1,
                           height_offset: float = 0.0) -> None:
        '''Setup for shape animation per frame'''
        self.ctx.clear(*(channel/255.0 for channel in self.base_color))
        self.ctx.enable(moderngl.DEPTH_TEST)

        camera_dist_horizontal = 3.0
        camera_height = camera_height + height_offset
        camera_pos = (np.cos(angle) * camera_dist_horizontal, np.sin(angle) * camera_dist_horizontal, camera_height)

        screen_ratio = self.width()/self.height()
        proj = Matrix44.perspective_projection(45.0, screen_ratio, 0.1, 100.0)
        lookat = Matrix44.look_at(
            camera_pos,
            (0.0, 0.0, lookat_height + height_offset),  # Lookat height
            (0.0, 0.0, 1.0),
        )

        self.mvp.write((proj * lookat).astype('f4'))
        self.light.value = camera_pos
