using MultivariateOrthogonalPolynomials, Test, ForwardDiff
using ForwardDiff: gradient


k = 0; m = 0; n = 2

Z_x = 𝐱 -> gradient(𝐱 -> zernikez(n,m,𝐱), 𝐱)[1]
Z_y = 𝐱 -> gradient(𝐱 -> zernikez(n,m,𝐱), 𝐱)[2]


𝐱 = SVector(0.1,0.2)
r = norm(𝐱); θ = atan(𝐱[2], 𝐱[1])
z = 2r^2 - 1
zernikez(n,m,𝐱)

W = (n,a,b) -> 2^(a+b+1)/(2n+a+b+1) * gamma(n+a+1)gamma(n+b+1)/(gamma(n+a+b+1)factorial(n))

@test jacobip(n,k, m,z) / sqrt(W(n,k,m)) ≈ normalizedjacobip(n,k,m,z)

r^m * cos(m*θ) * jacobip(n,k, m,z) / sqrt(W(n,k,m) / 2^(2+k+m))

sqrt(π) * zernikez(n,m,𝐱)

@time expand(Zernike(), Z_x)

