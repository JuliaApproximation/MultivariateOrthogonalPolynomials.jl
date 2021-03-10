using MultivariateOrthogonalPolynomials, FastTransforms, BlockBandedMatrices, Plots

WZ = Weighted(Zernike(1))
xy = axes(WZ,1)
x,y = first.(xy),last.(xy)
u = Zernike(1) \ @. exp(x*cos(y))

N = 50
KR = Block.(Base.OneTo(N))
Δ = BandedBlockBandedMatrix((Zernike(1) \ (Laplacian(xy) * WZ))[KR,KR],(0,0),(0,0))
C = (Zernike(1) \ WZ)[KR,KR]
k = 5
L = Δ - k^2 * C

v = (L \ u[KR])

F = factorize(Zernike(1)[:,KR])
F.plan \ v
F |> typeof |> fieldnames

grid(WZ[:,KR])

F |>typeof |> fieldnames
F.F * v

m = DiskTrav(v).matrix

plan_disk2cxf(m, 0, 0) * m


