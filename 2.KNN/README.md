## NN-Trees

### Kd-tree

How to construct a Kd-tree:
1. If there is only one point, or number of points < leaf_size, build a leaf
2. Otherwise, divide the points in half by ahyperplaneperpendicular to the selected splitting axis
3. Recursively repeat the first two steps.

### OCT-tree
* Each node has 8 children
* oct–tree
* Specifically for 3D, 23=8
* In kd-tree, it is non-trivial to determine whether the NN search is done, so we have to go back to root every time
* Octree is more efficient because we can stop without going back to root
