## Clusting_Algorithm

Python implementation of K-means, GMM and Spectral Clustering algorithm and compare their effects Object.

### K-Means

wenn π input data points, find πΎ clusters.
1. Randomly select K center points
2. Each data point is assigned to one of the K centers.
3. Re-compute the K centers by the mean of each group
4. Iterate step 2 & 3.

### GMM

1. Initialize the means, covariances and weights
2. E-step. Evaluate the posterior, intuitively this is the probability of π₯π being assigned to each of the K clusters.
3. M-Step.Estimate the parameters using MLE.
4. Evaluate the log likelihood, if converges, stop. Otherwise go back to E-step

### Spectral Clustering

1. Build the graph to get adjacency matrix πββπΓπ
2. Compute unnormalized Laplacian πΏ
3. Compute the first (smallest) πeigenvectors π£1,β―,π£πof πΏ
4. Let πββπΓπ be the matrix contraining the vectorsπ£1,β―,π£π as columns
5.For π=1,β―π, let π¦πββπ be the vector corresponding to the π th row of π
6.Cluster the points {π¦πββπ} with k means algorithm into clusters πΆ1,β―,πΆπ
7.The final output clusters are π΄1,β―,π΄πwhere π΄π={π|π¦πβπΆπ}
