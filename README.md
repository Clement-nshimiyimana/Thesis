# Thesis

We introduce two proximal operator-based algorithms, namely Chambolle-Pock and Primal-Dual Douglas-Rachfold for non convex optimization problems with the objective functions f(x) + g(Dx), where f is convex and g is non convex function. We implement our methods to signal reconstruction problem with l0 norm regularization. Numerical results show that both algorithms similarly converge to critical points and provide good results.

Problem: 
```math 
$\min 1/2 ||Ax-b||^2 + \lambda ||Dx||_0$
```

## Run

```bash
cd Thesis
Main.py

Plot.py
```
