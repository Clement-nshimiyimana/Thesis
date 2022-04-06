# Thesis

We introduce two proximal operator-based algorithms, namely Chambolle-Pock and Primal-Dual Douglas-Rachfold for non convex optimization problems with the objective functions f(x) + g(Dx), where f is convex and g is non convex function. We implement our methods to signal reconstruction problem with l0 norm regularization. Numerical results show that both algorithms similarly converge to critical points and provide good results.

Problem: 
![\min \frac{1}{2}||Ax-b||^2 + \lambda ||Dx||_0](http://www.sciweavers.org/tex2img.php?eq=%5Cmin%20%5Cfrac%7B1%7D%7B2%7D%7C%7CAx-b%7C%7C%5E2%20%2B%20%5Clambda%20%7C%7CDx%7C%7C_0&fc=Black&im=jpg&fs=12&ff=arev&edit=)

## Run

```bash
cd Thesis
Main.py

Plot.py
```
