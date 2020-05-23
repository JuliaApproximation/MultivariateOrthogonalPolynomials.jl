using MultivariateOrthogonalPolynomials, StaticArrays, BlockArrays, Test


P = JacobiTriangle()
xy = axes(P,1)
@test xy[SVector(0.1,0.2)] == SVector(0.1,0.2)

∂ˣ = PartialDerivative{(1,0)}(xy)
∂ʸ = PartialDerivative{(0,1)}(xy)

Dˣ = JacobiTriangle(1,0,1) \ (∂ˣ * P)
Dʸ = JacobiTriangle(0,1,1) \ (∂ʸ * P)

M = P'P;

Rx = JacobiTriangle(1,0,0) \ P
Lx = P \ WeightedTriangle(1,0,0)
Ry = JacobiTriangle(0,1,0) \ P
Ly = P \ WeightedTriangle(0,1,0)


N = 50

X = Lx[Block.(1:N), Block.(1:N)] * Rx[Block.(1:N), Block.(1:N)];
Y = Ly[Block.(1:N), Block.(1:N)] * Ry[Block.(1:N), Block.(1:N)];

K = sqrt.(M[Block.(1:N), Block.(1:N)])

X̃ = (K * X * inv(K));
Ỹ = (K * Y * inv(K));
A = X̃[Block(N-2,N-2)]
B = X̃[Block(N-2,N-1)][:,1:end-1]

C = Ỹ[Block(N-2,N-2)]
D = Ỹ[Block(N-2,N-1)][:,1:end-1]

x = z -> A + B/z + B'*z
y = z -> C + D/z + D'*z

z = exp(0.1im); x(z) * y(z) - y(z) * x(z)

z = exp(0.2im);
eigvals(Hermitian(x(z)))
eigvals(Hermitian(y(z)))

z=1; eigvals(A + B*z + B'/z)
z=1; eigvals(A + B*z + B'/z)




using Plots, BandedMatrices

scatter(diag(X̃[Block(N-2,N-2)]))
λ = X̃[Block(N-2,N-2)][band(0)]
μ = X̃[Block(N-1,N-1)][band(0)]

plot((Diagonal(inv.(sqrt.(λ))) * Ỹ[Block(N-2,N-2)] * Diagonal(inv.(sqrt.(λ))))[band(0)])
plot!((Diagonal(inv.(sqrt.(λ))) * Ỹ[Block(N-2,N-2)] * Diagonal(inv.(sqrt.(λ))))[band(1)])

Diagonal(inv.(sqrt.(μ))) * X̃[Block(N-1,N-2)]  * Diagonal(inv.(sqrt.(λ)))
Diagonal(inv.(sqrt.(μ))) * Ỹ[Block(N-1,N-2)]  * Diagonal(inv.(sqrt.(λ)))

plot(X̃[Block(N-2,N-2)][band(0)])
plot!(X̃[Block(N-2,N-1)][band(0)])
plot!(Ỹ[Block(N-2,N-2)][band(0)])
plot!(Ỹ[Block(N-2,N-2)][band(1)])

