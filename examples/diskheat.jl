using MultivariateOrthogonalPolynomials, DifferentialEquations

Z = Zernike(1)
W = Weighted(Z)
xy = axes(W,1)
Δ = Z \ Laplacian(xy) * W