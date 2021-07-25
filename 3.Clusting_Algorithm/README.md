## Clusting_Algorithm

Python implementation of K-means, GMM and Spectral Clustering algorithm and compare their effects Object.

### K-Means

wenn 𝑁 input data points, find 𝐾 clusters.
1. Randomly select K center points
2. Each data point is assigned to one of the K centers.
3. Re-compute the K centers by the mean of each group
4. Iterate step 2 & 3.

### GMM

1. Initialize the means, covariances and weights
2. E-step. Evaluate the posterior, intuitively this is the probability of 𝑥𝑛 being assigned to each of the K clusters.
3. M-Step.Estimate the parameters using MLE.
4. Evaluate the log likelihood, if converges, stop. Otherwise go back to E-step

### Spectral Clustering

1. Build the graph to get adjacency matrix 𝑊∈ℝ𝑛×𝑛
2. Compute unnormalized Laplacian 𝐿
3. Compute the first (smallest) 𝑘eigenvectors 𝑣1,⋯,𝑣𝑘of 𝐿
4. Let 𝑉∈ℝ𝑛×𝑘 be the matrix contraining the vectors𝑣1,⋯,𝑣𝑘 as columns
5.For 𝑖=1,⋯𝑛, let 𝑦𝑖∈ℝ𝑘 be the vector corresponding to the 𝑖 th row of 𝑉
6.Cluster the points {𝑦𝑖∈ℝ𝑘} with k means algorithm into clusters 𝐶1,⋯,𝐶𝑘
7.The final output clusters are 𝐴1,⋯,𝐴𝑘where 𝐴𝑖={𝑗|𝑦𝑗∈𝐶𝑖}
