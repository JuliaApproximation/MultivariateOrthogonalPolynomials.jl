using MultivariateOrthogonalPolynomials, DifferentialEquations, Plots
pyplot() # pyplot supports disks

Z = Zernike(1)
W = Weighted(Z)
xy = axes(W,1)
x,y = first.(xy),last.(xy)
Δ = Z \ Laplacian(xy) * W
S = Z \ W

# initial condition is (1-r^2) * exp(-(x-0.1)^2 - (y-0.2)^2)

K = Block.(Base.OneTo(11))

Δₙ = Δ[K,K]
Sₙ = S[K,K]
Zₙ = Z[:,K]
Wₙ = W[:,K]

diskheat(c, (Δₙ, Sₙ), t) = Sₙ \ (Δₙ * c)

c₀ = Zₙ \ @.(exp(-(x-0.1)^2 - (y-0.2)^2))
u = solve(ODEProblem(diskheat, c₀, (0.,1.), (Δₙ, Sₙ)), Tsit5(), reltol=1e-8, abstol=1e-8)


surface(Wₙ * u(1.0))