# CVclass

assignment 1 :

1. Build a system of equations of the form 𝐴𝑥 = 𝑏, as learned in class, for projective
   transformation. Attach the formula development to your exercise solution. How do we get the
   conversion matrix from the equation system?
2. Write a function that estimates the transformation coefficients from source (src) to destination
   (dst), from the equation system in section 1.

---

## Answer:

### 1. Ax = b when:

```text
A = [[h00 h01 h02]
     [h10 h11 h12]
     [h20 h21 h22]]

b = X_hat(without normalize) = [
    x_h_raw
    y_h_raw
    w
]

x = source point = [
    x
    y
    1
]
```

```
w = h20*x + h21*y + h22
x_dst = x_h_raw/w = h00*x + h01*y + h02*1 / (h20*x + h21*y + h22)
y_dst = y_h_raw/w = h10*x + h11*y + h12*1 / (h20*x + h21*y + h22)
```

### arrangment

```text
x_dst*(h20*x + h21*y + h22) = h00*x + h01*y + h02*1
y_dst*(h20*x + h21*y + h22) = h10*x + h11*y + h12*1

x_dst*(h20*x + h21*y + h22) - (h00*x + h01*y + h02*1) = 0
y_dst*(h20*x + h21*y + h22) - (h10*x + h11*y + h12*1) = 0
```

```text
h20*x_dst*x + h21*x_dst*y + h22*x_dst - h00*x - h01*y - h02*1 = 0
h20*y_dst*x + h21*y_dst*y + h22*y_dst - h10*x - h11*y - h12*1 = 0
```

We want to have h vector such that:

```text
h = [h00 h01 h02 h10 h11 h12 h20 h21 h22]^T
```

Therefore:

```text
[[-x, -y, -1,  0,  0,  0,  x_dst*x,  x_dst*y,  x_dst],
 [ 0,  0,  0, -x, -y, -1,  y_dst*x,  y_dst*y,  y_dst]]
```

---

From this equation = Ah = 0  
We want to find the smallest value h such that:

```
min ||Ah||
and
||h|| = 1
```

so that **h = 0** is not a trivial solution.

---

### The loss function:

```
(1 - Gamma)*hTh - hTATAh = 0
```

Gamma — penalty for h constrain  
hTATAh = such that the norm goes to 0

### the derivative:

```
-2Gamma*h - 2ATAh = 0
```

### and the solution:

```
(Gamma - ATA) * h = 0
```

which means the solution is the eigen vector of the smallest eigen value:

```
h = min(V)
```
