## RANSAC_DBSCNA

Python implementation to remove the ground from the lidar points using RANSAC; then Clustering over the remaining points using DBSCNA. <br>

### RANSAC

1. Randomly select a minimal subset of points required to solve the model
2. Solve the model
3. Compute error function for each point `p𝑖=(𝑥𝑖,y𝑖)`
4. Count the points consistent with the model, `di` < `𝜏`
5. Repeat step 1-4 for N-iterations, choose the model with most inlier points

### DBSCNA

1. Randomly select a unvisited point `p`, find its neighborhood with in `r`
2. determine whether number of points within `r` >= `min_samples`? if Yes, `p` is a core point, Create a cluster `C`, go to step 3, mark `p` as visited; if not, Mark `p` as noise and visited.<br>
3. Go through points within its `r`-neighborhood, label it as 'C'. If it is a core point, set it as the “`new p`”, repeat step-3
4. Remove cluster `C` from the database, go to step-1
5. Terminate when all points are visited.
