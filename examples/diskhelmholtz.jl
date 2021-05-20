using MultivariateOrthogonalPolynomials, FastTransforms, BlockBandedMatrices, Plots
plotly()

Z = Zernike(1)
W = Weighted(Z)
xy = axes(Z,1); x,y = first.(xy),last.(xy)
Δ = Z \ (Laplacian(xy) * W)
S = Z \ W

k = 2
f = @.(cos(x*exp(y)))
F = factorize(Δ + k^2 * S)
c = Z \ f
F \ c

u = W * ((Δ + k^2 * S) \ (Z \ f))



