import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	loss = F.binary_cross_entropy_with_logits(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	
	# Nearest neigbours from src to target
	dists_src, _, _ = knn_points(point_cloud_src, point_cloud_tgt, K=1)
	loss_src_to_tgt = dists_src.squeeze(-1).mean()

	# Nearest neigbours from target to src
	dists_tgt, _, _ = knn_points(point_cloud_tgt, point_cloud_src, K=1)
	loss_tgt_to_src = dists_tgt.squeeze(-1).mean()

	# Chamfer loss
	loss_chamfer = loss_src_to_tgt + loss_tgt_to_src

	return loss_chamfer

def smoothness_loss(mesh_src):
	# implement laplacian smoothening loss
	loss_laplacian = mesh_laplacian_smoothing(mesh_src, method="uniform")

	return loss_laplacian