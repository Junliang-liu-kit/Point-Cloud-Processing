## PCA_Voxel Grid

### PCA
PCA is to find the dominant directions of the point cloud, perform PCA by:
1. Normalized by the center
2. Compute SVD
3. The principle vectors are the columns of 𝑈𝑟 (Eigenvector of 𝑋 = Eigenvector of 𝐻)

Applications:
* Dimensionality reduction
* Surface normal estimation

Surface normal on 3D point cloud
1. Select a point P
2. Find the neighborhood that defines the surface
3. PCA
4. Normal -> the least significant vector
5. Curvature -> ratio between eigen values 𝜆3/(𝜆1 + 𝜆2 + 𝜆3)

### Voxel Grid Downsampling

Pseudo code description the algorithm:
1. Compute the min or max of the point set
2. Determine the voxel grid size 𝑟
3. Compute the dimension of the voxel grid
4. Compute voxel index for each point
5. Sort the points according to the index in Step 4
6. Iterate the sorted points, select points according to Centroid / Random method

