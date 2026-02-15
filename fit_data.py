import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch


# Imports for visualizing 
import matplotlib.pyplot as plt
import mcubes
import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import numpy as np
import imageio
from pytorch3d.renderer import TexturesVertex

import os
from pathlib import Path



from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)


def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    return parser



def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def voxel2mesh(voxels, device=None):
    '''
    Function to visualize source and generated voxel
    '''
    if device is None:
        device = get_device()

    voxel_size = voxels.shape[0]
    
    # Extract mesh from voxel grid
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels.detach().cpu().numpy()), isovalue=0)
    
    # Convert to PyTorch tensors and move to device
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces.astype(np.int64), dtype=torch.long, device=device)

    # Normalize vertex coordinates
    vertices = vertices / voxel_size  
    
    textures = torch.ones_like(vertices)  
    textures = textures * torch.tensor([0.7, 0.7, 1], device=device)  

    mesh = pytorch3d.structures.Meshes(verts=[vertices], faces=[faces], textures=pytorch3d.renderer.TexturesVertex([textures])).to(device)

    # Lighting setup
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device)

    return mesh, lights


def render_pointcloud(points, phi=0, device = None, 
                      image_size=256, background_color=(1, 1, 1), rgb=None):
    if device is None:
        device = get_device()
    
    
    points = points.to(device)
    if rgb is None:
        rgb = torch.tensor([0.7, 0.7, 1], device=device).expand(1, points.shape[1], -1)
    
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=rgb)
    R ,T = pytorch3d.renderer.look_at_view_transform(dist=2, elev=30, azim=phi)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.detach().cpu().numpy()[0, ..., :3]  
    return rend


def render_mesh(mesh, rotation=0, device=None,image_size=256):

    if device is None:
        device = get_device()

    verts_rgb = torch.ones_like(mesh.verts_packed(), device=device) * torch.tensor([0.7, 0.7, 1], device=device)
    mesh.textures = TexturesVertex(verts_features=verts_rgb.unsqueeze(0))  # Ensure batch dim

    # Attempt New Cam rot/translation
    R_, t_ = pytorch3d.renderer.look_at_view_transform(dist=2, elev=0, azim=rotation)
    R_ = R_.to(device)
    t_ = t_.to(device)
    
    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R_, T=t_, fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    renderer = get_mesh_renderer(image_size=image_size)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.detach().cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    
    return rend
    

def render_voxel(mesh, lights, deg, device=None, image_size=256):
    if device is None:
        device = get_device()
    
    # Rotate camera around the object without shifting it
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=2,  
        elev=30,  
        azim=deg
    )
    
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    # Render the mesh
    rend = renderer(mesh, cameras=cameras, lights=lights)
    
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)



def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')

    device = get_device()
    
    images = []
    for angle in range(0, 360+10, 10):
        tgt_image = render_mesh(mesh_tgt, angle, device)
        tgt_image = (tgt_image* 255).astype(np.uint8)

        src_image = render_mesh(mesh_src, angle, device)
        src_image = (src_image* 255).astype(np.uint8)

        comb_imgs = np.hstack([tgt_image, src_image])
        images.append(comb_imgs)
    
    folder = Path("results/")
    folder.mkdir(parents=True, exist_ok=True) 
    path_src = folder/f"fit_data-{args.type}.gif"

    # Save GIF with correct formatting
    imageio.mimsave(path_src, images, format='GIF', duration=len(images) / 1000, loop=0)

    print(f'Gif Saved @{path_src}')


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')
    # Convert voxel to mesh
    device = get_device()
    images = []

    for angle in range(0, 360+10, 10):
        tgt_image = render_pointcloud(pointclouds_tgt, angle, device)
        tgt_image = (np.clip(tgt_image, 0, 1)* 255).astype(np.uint8)

        src_image = render_pointcloud(pointclouds_src, angle, device)
        src_image = (np.clip(src_image, 0, 1)* 255).astype(np.uint8)

        comb_imgs = np.hstack([tgt_image, src_image])
        images.append(comb_imgs)
    
    folder = Path("results/")
    folder.mkdir(parents=True, exist_ok=True) 
    path_src = folder/f"fit_data-{args.type}.gif"

    # Save GIF with correct formatting
    imageio.mimsave(path_src, images, format='GIF', duration=len(images) / 1000, loop=0)

    print(f'Gif Saved @{path_src}')


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Regression Done!')
    # Convert voxel to mesh
    device = get_device()
    tgt_mesh, tgt_lights = voxel2mesh(voxels_tgt.squeeze(0), device)
    src_mesh, src_lights = voxel2mesh(voxels_src.squeeze(0), device)

    images = []

    for angle in range(0, 360+10, 10):
        tgt_image = render_voxel(tgt_mesh, tgt_lights, angle, device)
        tgt_image = (tgt_image * 255).astype(np.uint8)

        src_image = render_voxel(src_mesh, src_lights, angle, device)
        src_image = (src_image * 255).astype(np.uint8)

        img_stacked = np.hstack((tgt_image, src_image))
        images.append(img_stacked)

    

    folder = Path("results/")
    folder.mkdir(parents=True, exist_ok=True) 
    path_src = folder/f"fit_data-{args.type}.gif"

    # Save GIF with correct formatting
    imageio.mimsave(path_src, images, format='GIF', duration=len(images) / 1000, loop=0)

    print(f'Gif Saved @{path_src}')


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)        
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args)        


    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
