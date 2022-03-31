using MultivariateOrthogonalPolynomials, FastTransforms, BlockBandedMatrices, Plots, LinearAlgebra, StaticArrays
plotly()


####
# Solving
#
#   (Δ + k^2*I) * u = f
#
# in a unit disk |𝐱| ≤ 1, 
# using orthogonal polynomials (in x and y) in a disk
# with the weight (1-|𝐱|^2) = (1-x^2-y^2)
####

Z = Zernike(1)
W = Weighted(Z) # w*Z
xy = axes(Z, 1);
x, y = first.(xy), last.(xy);
Δ = Z \ (Laplacian(xy) * W)
S = Z \ W # identity


f = @.(cos(x * exp(y)))

f[SVector(0.1, 0.2)]
g = ((1 .- x .^ 2 .- y .^ 2) .* f)
@time W \ g

Z \ g
N = 100
S[Block.(1:N), Block.(1:N)] * (W\g)[Block.(1:N)]


# F = factorize(Δ + k^2 * S)
# c = Z \ f
# F \ c

# u = W * ((Δ + k^2 * S) \ (Z \ f))


N = 20
k = 20
f = @.(cos(x * exp(y)))
c = Z \ f
𝐮 = (Δ+k^2*S)[Block.(1:N), Block.(1:N)] \ c[Block.(1:N)]
u = W[:, Block.(1:N)] * 𝐮
axes(u)


ũ = Z / Z \ u

ũ = Z / Z \ u

ũ = (Z / Z) \ u
ũ = inv(Z * inv(Z)) * u
ũ = Z * (inv(Z) * u)
ũ = Z * (Z \ u)
# Z \ u means Find c s.t. Z*c == u

sum(ũ .* f)

W \ f


sum(u .^ 2 * W \ f)
norm(u)

surface(u)

# Δ*u == λ*u
# Z\Δ*W*𝐮 == λ*Z\W*𝐮
# Δ*𝐮 == λ*S*𝐮
Matrix(Δ[Block.(1:N), Block.(1:N)])
eigvals(Matrix(Δ[Block.(1:N), Block.(1:N)]), Matrix(S[Block.(1:N), Block.(1:N)]))

Z \ (x .* Z)



# u = (1-x^2) * P^(1,1) * 𝐮 = W * 𝐮
# v = (1-x^2) * P^(1,1) * 𝐯 = W * 𝐯
# -<D*v,D*u>
# -(D*v)'(D*u) == -𝐯'*(D*W)'D*W*𝐮
# <v,u> == 𝐯'*W'W*𝐮

P¹ = Jacobi(1, 1)
W = Weighted(P¹)
x = axes(W, 1)
D = Derivative(x)
-(D * W)' * (D * W)
W'W

# p-FEM 

P = Legendre()
u = P * [randn(5); zeros(∞)]
u' * u

T[0.1, 1:10]
T'[1:10, 0.1]
axes(T')

