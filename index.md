# Single View to 3D

## 1. Exploring loss functions

---

## 1.1. Fitting a voxel grid

Left is ground-truth, while right is fitted voxel

![img](results/fit_data-vox.gif "Fitting Voxel")

## 1.2. Fitting a point cloud

Left is ground-truth, while right is fitted point-cloud

![img](results/fit_data-point.gif "Point Cloud")

## 1.3. Fitting a mesh

Left is ground-truth, while right is fitted mesh

![img](results/fit_data-mesh.gif "Mesh")

# 2. Reconstructing 3D from single view

---

## 2.1. Image to voxel grid

    From left to right: RGB image, ground-truth rendered mesh, predicted rendered mesh

![img](results/vox/100_vox.gif)

![img](results/vox/200_vox.gif)

![img](results/vox/400_vox.gif)

## 2.2. Image to point cloud

From left to right: RGB image, ground-truth rendered mesh, predicted rendered pointcloud

![img](results/point/100_point.gif)

![img](results/point/200_point.gif)

![img](results/point/400_point.gif)

## 2.3. Image to mesh

From left to right: RGB image, ground-truth rendered mesh, predicted rendered mesh

![img](results/mesh/100_mesh.gif)

![img](results/mesh/200_mesh.gif)

![img](results/mesh/400_mesh.gif)

## 2.4. Quantitative comparisions

#### F1 score for predicted voxel

Final avg score: 51.1

![img](results/eval_vox.png)

#### F1 score for predicted point cloud

Final avg score: 75.65

![img](results/eval_point.png)

#### F1 score for predicted mesh

Final avg score: 72.2

![img](results/eval_mesh.png)

**Explanation:**

* **Point Clouds:** They are a sparse representation of 3D geometry, capturing key surface points without needing to fill an entire volume. Because they focus on essential geometric features, they’re easier to align with ground truth, leading to a highest F1 score.
* **Meshes:** These represent surfaces with vertices and faces, providing more structure than point clouds but still focusing on the object's boundary. While more detailed, they can introduce errors in surface connectivity or topology, thus the F1 score is lower than point-cloud.
* **Voxels:** Voxel grids discretize 3D space into small cubes, requiring dense, volumetric predictions. This introduces ambiguity and high false positives/negatives, especially with limited input views, leading to the lowest the F1 score.

## 2.5. Analyse effects of hyperparams variations

I varied the *'n_points'* hyperparameter and analysed the the effects

| n_points | F1 Score | F1 plot                                  | Gif@300 Image                           |
| -------- | -------- | ---------------------------------------- | --------------------------------------- |
| 512      | 70.65    | ![img](results/point/_512_eval_point.png)  | ![img](results/point/_512_300_point.gif)  |
| 1000     | 75.66    | ![img](results/eval_point.png)             | ![img](results/point/_1000_300_point.gif) |
| 2048     | 80.83    | ![img](results/point/_2048_eval_point.png) | ![img](results/point/_2048_300_point.gif) |

**Analysis:** As the number of points increases, the average F1 score improves because more samples help the model converge closer to the optimal result. However, this also increases GPU memory usage. The 1000-point model offers a balanced tradeoff between achieving a high F1 score and managing memory consumption.

## 2.6. Interpret your model

To better understand precision, I visualized the nearest neighbors (k-NN) from the ground truth mesh to the predicted point cloud. Similarly, to gain insight into recall, I visualized the nearest neighbors from the predicted point cloud to the ground truth mesh. The GIFs below illustrate these visualizations:

From left to right:

1. Original Mesh
2. Predicted Point Cloud
3. Nearest Neighbors (Ground Truth → Predicted)
4. Nearest Neighbors (Predicted → Ground Truth)

The color gradient represents the distance error:

* **Blue:** Points with low error (closer match)
* **Red:** Points with high error (larger distance)

![img](results/interpret/0_point.gif)

![img](results/interpret/400_point.gif)

# 3. Exploring other architectures / datasets

---

## 3.3 Extended dataset for training

I trained the point-cloud decoder on the extended r2n2_shapenet_dataset model

##### **Visualizations**

From left to right: original Image, original Mesh, predicted point cloud

As shown in the third example, for the same chair class, the predicted point cloud appears more geometrically aligned with the original mesh. This suggests that the model learns the characteristics of a chair more effectively, likely due to the increased diversity in the training data.

![img](results/point/extended/0_point.gif)

![img](results/point/extended/500_point.gif)

![img](results/point/extended/800_point.gif)

**Failure Example:** The predicted mesh does not align with the original mesh. This discrepancy may stem from a lack of similar example images in the dataset, or it could indicate the need for a more complex model to better capture intricate geometric features.![img](results/point/extended/700_point.gif)

##### Model comparison  "training on one class" VS "training on three classes"

| Training Set  | F1 Score | F1 plot                                     |
| ------------- | -------- | ------------------------------------------- |
| One class     | 75.66    | ![img](results/eval_point.png)                |
| Three classes | 85.73    | ![img](results/point/extended/eval_point.png) |

The average F1 score significantly increases to 85.73, indicating that the model learns more effectively about the object's general shape, resulting in a better fit. Greater class diversity enhances the model's ability to understand and represent each object more accurately.
