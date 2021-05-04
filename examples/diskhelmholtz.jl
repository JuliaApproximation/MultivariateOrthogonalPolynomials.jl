using MultivariateOrthogonalPolynomials, FastTransforms, BlockBandedMatrices, WGLMakie
import MultivariateOrthogonalPolynomials: ZernikeITransform, grid

Z = Zernike()[:,Block.(Base.OneTo(10))]
G = grid(Z)

contourf(first.(G), last.(G), ones(size(G)...))


scatter(vec(first.(G)), vec(last.(G)))
G[1,1].θ

WZ = Weighted(Zernike(1))
xy = axes(WZ,1)
x,y = first.(xy),last.(xy)
f = Zernike(1) \ @. exp(x*cos(y))

N = 50
KR = Block.(Base.OneTo(N))
Δ = (Zernike(1) \ (Laplacian(xy) * WZ))[KR,KR]
C = (Zernike(1) \ WZ)[KR,KR]
k = 5
L = Δ - k^2 * C

v = f[KR]
@time u = (L \ v);

g = MultivariateOrthogonalPolynomials.grid(Zernike(1)[:,KR])
U = ZernikeITransform{Float64}(N, 0, 1) * u

plot(first.(g), last.(g), U)



F = factorize(Zernike(1)[:,KR]).plan
F \ u
u


F*u


F.plan \ v
F |> typeof |> fieldnames

grid(WZ[:,KR])

F |>typeof |> fieldnames
F.F * v

m = DiskTrav(v).matrix

plan_disk2cxf(m, 0, 0) * m


