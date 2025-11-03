# CVclass — Assignment 1

## 1. Building the equation system for a projective transformation

We want to express the homography as:

```
A h = 0
```

where the homography vector is:

```
h = [h00 h01 h02 h10 h11 h12 h20 h21 h22]^T
```

Given a source point:

```
x = [x, y, 1]
```

and destination point:

```
x_dst = [x_dst, y_dst]
```

the projective mapping is:

```
w = h20*x + h21*y + h22

x_dst = (h00*x + h01*y + h02) / w
y_dst = (h10*x + h11*y + h12) / w
```

Rearranging:

```
x_dst*(h20*x + h21*y + h22) - (h00*x + h01*y + h02) = 0
y_dst*(h20*x + h21*y + h22) - (h10*x + h11*y + h12) = 0
```

Each correspondence gives two rows in A:

```
[-x  -y  -1   0   0   0   x_dst*x   x_dst*y   x_dst]
[ 0   0   0  -x  -y  -1   y_dst*x   y_dst*y   y_dst]
```

Thus:

```
A h = 0
```

To avoid the trivial solution **h = 0**, we constrain:

```
||h|| = 1
```

and solve:

```
min ||A h||  subject to  ||h|| = 1
```

The loss function:

```
(1 - Γ) h^T h  -  h^T A^T A h
```

Derivative:

```
-2 Γ h - 2 A^T A h = 0
```

Rearranged:

```
(Γ I - A^T A) h = 0
```

This is an eigenvalue problem.  
The solution is the eigenvector corresponding to the smallest eigenvalue of:

```
A^T A
```

which is the right singular vector of A (SVD):

```
A = U Σ V^T
h = last column of V
```
