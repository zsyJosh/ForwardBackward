Logic Message Passing
=====================

# Coding Logic

- Enable flexible ablation of graph construction.
- Allow subsampling of triangles.
- Avoid full line graph / all triangles.

# Pipeline

- Input graph

### For this part, we would like to keep flexibility

- Create fact variables.
- Remove easy edges.
- Create query variables (NBFNet-like).
- (Optional) Create unary variables.
- (Optional) Create auxiliary variables (other than fact / query variables).

The number of variables should be bounded by O(|E| + d|V|). We may pick d = O(|E| / |V|)

### For this part, we would like to opt for efficiency

- Convert to triangle list and downsample the triangles for each edge.
  Fuse the operations to avoid materializing full triangle list, which might be O(|V|^2) in the worst case.

The number of triangles should be bounded by O(k|E|). k is the maximum number of triangles allowed for each variable. 