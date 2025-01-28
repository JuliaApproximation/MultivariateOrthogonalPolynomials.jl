# MultivariateOrthogonalPolynomials.jl

[![Build Status](https://github.com/JuliaApproximation/MultivariateOrthogonalPolynomials.jl/workflows/CI/badge.svg)](https://github.com/JuliaApproximation/MultivariateOrthogonalPolynomials.jl/actions)
[![codecov](https://codecov.io/gh/JuliaApproximation/MultivariateOrthogonalPolynomials.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaApproximation/MultivariateOrthogonalPolynomials.jl)

This is an experimental package to add support for multivariate orthogonal polynomials on disks, spheres, triangles, and other simple
geometries to [ContinuumArrays.jl](https://github.com/JuliaApproximation/ContinuumArrays.jl). At the moment it primarily supports triangles. For example,
we can solve variable coefficient Helmholtz on the triangle with zero Dirichlet conditions as follows:
```julia
julia> using MultivariateOrthogonalPolynomials, StaticArrays, LinearAlgebra

julia> P = JacobiTriangle()
JacobiTriangle(0, 0, 0)

julia> x,y = coordinates(P);

julia> u = P * (P \ (exp.(x) .* cos.(y))) # Expand in Triangle OPs
JacobiTriangle(0, 0, 0) * [1.3365085377830084, 0.5687967596428205, -0.22812040274224554, 0.07733064070637755, 0.016169744493985644, -0.08714886622738759, 0.00338435674992512, 0.01220019521126353, -0.016867598915573725, 0.003930461395801074  …  ]

julia> u[SVector(0.1,0.2)] # Evaluate expansion
1.083141079608063
```
See the examples folder for more examples, including non-zero Dirichlet conditions, Neumann conditions, and piecing together multiple triangles. In particular, the [examples](examples/triangleexamples.jl) from Olver, Townsend & Vasil 2019.


This code relies on Slevinsky's [FastTransforms](https://github.com/MikaelSlevinsky/FastTransforms) C library for calculating transforms between values and coefficients. At the moment the path to the compiled FastTransforms library is hard coded in [c_transforms.jl](src/c_transforms.jl). 

## References


- S. Olver, A. Townsend & G.M. Vasil (2019), [A sparse spectral method on triangles](https://arxiv.org/pdf/1902.04863.pdf), arXiv:1902.04
- S. Olver & Y. Xuan (2019), [Orthogonal polynomials in and on a quadratic surface of revolution](https://arxiv.org/abs/1906.12305.pdf), arXiv:1906.12305
- G.M. Vasil, K.J. Burns, D. Lecoanet, S. Olver, B.P. Brown & J.S. Oishi (2016), [Tensor calculus in polar coordinates using Jacobi polynomials](http://arxiv.org/pdf/1509.07624.pdf), J. Comp. Phys., 325: 53–73
