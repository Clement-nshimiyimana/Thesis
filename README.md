# Thesis

We introduce two proximal operator-based algorithms, namely Chambolle-Pock and Primal-Dual Douglas-Rachfold for non convex optimization problems with the objective functions f(x) + g(Dx), where f is convex and g is non convex function. We implement our methods to signal reconstruction problem with l0 norm regularization. Numerical results show that both algorithms similarly converge to critical points and provide good results.

Problem: 
```math 
$`\min 1/2 ||Ax-b||^2 + \lambda ||Dx||_0`$
```
![\min 1/2 ||Ax-b||^2 + \lambda ||Dx||_0]%5Cmin%201%2F2%20%7C%7CAx-b%7C%7C%5E2%20%2B%20%5Clambda%20%7C%7CDx%7C%7C_0

## Run

```bash
cd Thesis
Main.py

Plot.py
```
