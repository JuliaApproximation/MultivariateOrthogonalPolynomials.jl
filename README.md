# MultivariateOrthogonalPolynomials.jl

[![Build Status](https://travis-ci.org/JuliaApproximation/MultivariateOrthogonalPolynomials.jl.svg?branch=master)](https://travis-ci.org/JuliaApproximation/MultivariateOrthogonalPolynomials.jl)

This is an experimental package to add support for multivariate orthogonal polynomials on disks, spheres, triangles, and other simple
geometries to [ApproxFun.jl](https://github.com/JuliaApproximation/ApproxFun.jl). At the moment it primarily supports triangles. For example,
we can solve variable coefficient Helmholtz on the triangle with zero Dirichlet conditions as follows:
```julia
using ApproxFun, MultivariateOrthogonalPolynomials
x,y = Fun(Triangle())
Δ = Laplacian() : TriangleWeight(1.0,1.0,1.0,JacobiTriangle(1.0,1.0,1.0))
V = x*y^2
L = Δ + 200^2*V
u = \(L, ones(Triangle()); tolerance=1E-5)
```
See the examples folder for more examples, including non-zero Dirichlet conditions, Neumann conditions, and piecing together multiple triangles. In particular, the [examples](examples/triangleexamples.jl) from Olver, Townsend & Vasil 2019.


This code relies on Slevinsky's [FastTransforms](https://github.com/MikaelSlevinsky/FastTransforms) C library for calculating transforms between values and coefficients. At the moment the path to the compiled FastTransforms library is hard coded in [c_transforms.jl](src/c_transforms.jl). 

## References


S. Olver, A. Townsend & G.M. Vasil (2019), [A sparse spectral method on triangles](https://arxiv.org/pdf/1902.04863.pdf), arXiv:1902.04
S. Olver & Y. Xuan (2019), [Orthogonal polynomials in and on a quadratic surface of revolution](https://arxiv.org/abs/1906.12305.pdf), arXiv:1906.12305
G.M. Vasil, K.J. Burns, D. Lecoanet, S. Olver, B.P. Brown & J.S. Oishi (2016), [Tensor calculus in polar coordinates using Jacobi polynomials](http://arxiv.org/pdf/1509.07624.pdf), J. Comp. Phys., 325: 53–73
