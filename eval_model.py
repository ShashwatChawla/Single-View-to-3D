import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
import math
import numpy as np

import torch.nn.functional as F
from fit_data import render_mesh, render_pointcloud, render_voxel, voxel2mesh
from pytorch3d.ops import sample_points_from_meshes

            
import os
from pathlib import Path



import imageio
import cv2

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--interpret', action='store_true', help="Enable interpretation mode")

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=1000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = F.sigmoid(predictions)
        H,W,D = voxels_src.shape
        # H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        if vertices_src.shape[0] < 1:
            return False
    
        
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
        # Apply a rotation transform to align predicted voxels to gt mesh
        angle = -math.pi
        axis_angle = torch.as_tensor(np.array([[0.0, angle, 0.0]]))
        Rot = axis_angle_to_matrix(axis_angle)
        T_transform = Rotate(Rot)
        pred_points = T_transform.transform_points(pred_points)
        # re-center the predicted points
        pred_points = pred_points - pred_points.mean(1, keepdim=True)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    if args.type == "vox":
        gt_points = gt_points - gt_points.mean(1, keepdim=True)
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics



def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)


        if args.type == "vox":
            predictions = predictions.squeeze(0)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)
        if metrics == False:
            continue
        # TODO:
        if (step % args.vis_freq) == 0:
            # Store img
            gt_img = (images_gt.squeeze(0).detach().cpu().numpy()*255).astype(np.uint8)
            # plt.imsave(f'results/{args.type}/{step}_gt-img.png', gt_img)
            
            # GT Mesh pre-processing
            ver = mesh_gt.verts_list()[0].unsqueeze(0)
            faces = mesh_gt.faces_list()[0].unsqueeze(0)
            textures = torch.ones_like(torch.tensor(ver))
            textures = textures * torch.tensor([0.7, 0.7, 1])
            vis_mesh_gt = pytorch3d.structures.Meshes(
                verts=ver,
                faces=faces,
                textures=pytorch3d.renderer.TexturesVertex(textures)
            ).to(torch.device(args.device))


            # Pred Mesh pre-processing
            vis_mesh_pred = None
            if(args.type == 'mesh'):
                pred_ver = predictions.verts_list()[0].unsqueeze(0)
                pred_faces = predictions.faces_list()[0].unsqueeze(0)
                pred_textures = torch.ones_like(torch.tensor(pred_ver))
                pred_textures = pred_textures * torch.tensor([0.7, 0.7, 1], device=args.device)
                vis_mesh_pred = pytorch3d.structures.Meshes(
                    verts=pred_ver,
                    faces=pred_faces,
                    textures=pytorch3d.renderer.TexturesVertex(pred_textures)
                ).to(torch.device(args.device))
            

            images = []
            pred_img = None

            for angle in range(0, 360+10, 10):
                # Store GT Mesh
                gt_mesh = (render_mesh(vis_mesh_gt, angle, args.device)*255).astype(np.uint8)
                
                # Store Pred Point Cloud
                if(args.type == 'point'):
                    pred_point = render_pointcloud(predictions, angle, device=args.device)
                    pred_point = (np.clip(pred_point,0, 1)*255).astype(np.uint8)
                    pred_img = pred_point

                # Store Pred Mesh
                elif(args.type == 'mesh'):
                    # Store gt mesh
                    pred_mesh = (render_mesh(vis_mesh_pred, angle, args.device)*255).astype(np.uint8)
                    pred_img = pred_mesh


                # Store Pred Voxel
                elif(args.type == 'vox'):                    
                    vox_mesh = pytorch3d.ops.cubify(F.sigmoid(predictions.unsqueeze(0)), 0.4, device=torch.device(args.device))
                    vox_ver = vox_mesh.verts_list()[0].unsqueeze(0)
                    vox_faces = vox_mesh.faces_list()[0].unsqueeze(0)
                    
                    vox_textures = torch.ones_like(torch.tensor(vox_ver))
                    vox_textures = vox_textures * torch.tensor([0.7, 0.7, 1], device=torch.device(args.device))
                    
                    pred_vox = pytorch3d.structures.Meshes(
                        verts=vox_ver,
                        faces=vox_faces,
                        textures=pytorch3d.renderer.TexturesVertex(vox_textures)
                    ).to(torch.device(args.device))

                    pred_vox = (render_mesh(pred_vox, angle, args.device)*255).astype(np.uint8)
                    pred_img = pred_vox
                
                
                if(args.interpret==True):
                    gt_pts = (sample_points_from_meshes(mesh_gt, args.n_points)).to(args.device) #[1, 1000, 3]
                    
                    # Compute nearest GT point for each predicted point
                    dists, _, _ = knn_points(predictions, gt_pts, K=1)
                    
                    # Normalize distances for coloring (min-max normalization)
                    
                    dists = dists.squeeze().detach().cpu().numpy()
                    
                    min_dist, max_dist = dists.min(), dists.max()
                    normalized_dists = (dists - min_dist) / (max_dist - min_dist)  # Scale to [0,1]
                    
                    # Convert normalized distances to colors (e.g., Blue (low) → Red (high))
                    
                    colors = np.zeros((dists.shape[0], 3))
                    colors[:, 0] = normalized_dists      # Red increases with error
                    colors[:, 2] = 1 - normalized_dists  # Blue decreases with error

                    # Convert to tensor
                    colors = torch.tensor(colors, dtype=torch.float32, device=args.device)
                    colors = colors.unsqueeze(0)
                    
                    vis_point1 = render_pointcloud(gt_pts, angle, device=args.device, rgb=colors)
                    vis_point1 = (np.clip(vis_point1,0, 1)*255).astype(np.uint8)

                    vis_point2 = render_pointcloud(predictions, angle, device=args.device, rgb=colors)
                    vis_point2 = (np.clip(vis_point2,0, 1)*255).astype(np.uint8)
                    # pred_img = pred_point

                
                gt_img = cv2.resize(gt_img, (gt_mesh.shape[1], gt_mesh.shape[0]))

                if(args.interpret==True):
                    com_imgs = np.hstack([gt_mesh, pred_img, vis_point1, vis_point2])
                else:
                    # com_imgs = pred_img
                    com_imgs = np.hstack([gt_img, gt_mesh, pred_img])
                
                images.append(com_imgs)



            if(args.interpret==True):
                folder = Path("results/interpret/")
                folder.mkdir(parents=True, exist_ok=True) 
                path_= folder/f'{step}_{args.type}.gif'
            else:
                folder = Path(f"results/{args.type}/extended_eval/")
                folder.mkdir(parents=True, exist_ok=True) 
                path_= folder/f'{step}_{args.type}.gif'
            
            imageio.mimsave(path_, images, format='GIF', duration=len(images) / 1000, loop=0)
            
            print(f"Saved Visualizations !!")
      

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
