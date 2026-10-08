using MultivariateOrthogonalPolynomials, FastTransforms, BlockBandedMatrices, Plots, LinearAlgebra, StaticArrays
pyplot()


####
# Solving
#
#   (Δ + k^2*I) * u = f
#
# in a unit disk |𝐱| ≤ 1, 
# using orthogonal polynomials (in x and y) in a disk
# with the weight (1-|𝐱|^2) = (1-x^2-y^2)
####

Z = Normalized(Zernike(1)) # orthonormal so that the mass matrix below is symmetric
W = Weighted(Z) # w*Z
x, y = coordinates(W)
Δ = Z \ laplacian(W)
S = Z \ W # identity


k = 20
L = Δ + k^2 * S # discretisation of Helmholtz
f = @.(cos(x * exp(y)))

u = W * (L \ (Z \ f))
contourf(u)


# One can also fix the discretisation size

N = 20
Zₙ = Z[:,Block.(1:N)]
Wₙ = W[:,Block.(1:N)]
Lₙ = L[Block.(1:N),Block.(1:N)]

u = Wₙ * (Lₙ \ (Zₙ \ f))
contourf(u)


# We can also do eigenvalues of the Laplacian
N = 20
Δₙ = Δ[Block.(1:N),Block.(1:N)]
Sₙ = S[Block.(1:N),Block.(1:N)]

λ,Q = eigen(Symmetric(Matrix(Δₙ)), Symmetric(Matrix(Sₙ)))

contourf(Wₙ * Q[:,end-10])