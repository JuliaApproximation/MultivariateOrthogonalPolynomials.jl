using MultivariateOrthogonalPolynomials, DifferentialEquations

Z = Zernike(1)
W = Weighted(Z)
xy = axes(W,1)
x,y = first.(xy),last.(xy)
Δ = Z \ Laplacian(xy) * W
S = Z \ W

# initial condition is (1-r^2) * exp(-(x-0.1)^2 - (y-0.2)^2)
c₀ = (Z \ @.(exp(-(x-0.1)^2 - (y-0.2)^2)))

K = Block.(Base.OneTo(101))
Δₙ = Δ[K,K]
Sₙ = S[K,K]

diskheat(u, (Δₙ, Sₙ), t) = Sₙ \ (Δₙ * u)

u = solve(ODEProblem(diskheat, ModalTrav(c₀[K]), (0.,1.), (Δₙ, Sₙ)), Tsit5(), reltol=1e-8, abstol=1e-8)
