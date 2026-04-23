import os
import numpy as np
import torch
import trimesh


def load_latent_codes(latent_codes_path: str) -> torch.tensor:
    latent_codes = torch.load(latent_codes_path)
    latent_codes = latent_codes['latent_codes']['weight']
    print(f"loaded {latent_codes.shape[0]} of {latent_codes.shape[1]} dimension")
    return latent_codes


def load_program(ctx, shader_directory: str, vertex_shader_name: str, fragment_shader_name: str):
    with open(os.path.join(shader_directory, vertex_shader_name), 'r') as file:
        vertex_shader = file.read()
    with open(os.path.join(shader_directory, fragment_shader_name), 'r') as file:
        fragment_shader = file.read()
    return ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)


def mesh_to_vao(ctx, mesh_path: str, program, scale: float = 1.0):
    if os.path.exists(mesh_path) is False:
        print("Mesh path does not exist, did you select a latent code? \r", end='')
        return None

    mesh = trimesh.load_mesh(mesh_path)
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[0] < 2:
        print("Detected incorrect mesh")
        return None

    vertex_data = np.array(mesh.vertices, dtype='f4') * scale
    vertex_buffer = ctx.buffer(vertex_data.tobytes())

    index_data = np.array(mesh.faces, dtype='i4').flatten()
    index_buffer = ctx.buffer(index_data.tobytes())

    vertex__normal_data = np.array(mesh.vertex_normals, dtype='f4')
    vertex_normal_buffer = ctx.buffer(vertex__normal_data.tobytes())

    vao = ctx.simple_vertex_array(program, vertex_buffer, 'in_position', index_buffer=index_buffer)
    normal_attr_location = program['in_normal'].location
    vao.bind(normal_attr_location, 'f', vertex_normal_buffer, '3f')

    return vao
