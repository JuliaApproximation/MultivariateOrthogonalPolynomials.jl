using MultivariateOrthogonalPolynomials

WZ = Weighted(Zernike(1))
xy = axes(WZ,1)
x,y = first.(xy),last.(xy)
u = WZ \ @. (exp(x*cos(y))*(1-x^2-y^2))

Δ = (Zernike(1) \ (Laplacian(xy) * WZ))
C = Zernike(1) \ WZ
N = 40; @time C[Block.(1:N), Block.(1:N)];
k = 100
L = Δ - k^2 * C

MemoryLayout(L)
