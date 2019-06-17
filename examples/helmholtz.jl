using ApproxFun, MultivariateOrthogonalPolynomials, BlockArrays, FillArrays
using Plots
import PyPlot
x,y = Fun(Triangle())
Δ = Laplacian() : TriangleWeight(1.0,1.0,1.0,JacobiTriangle(1.0,1.0,1.0))
V = x*y^2
L = Δ + 200^2*V
M = L[Block.(1:50),Block.(1:50)];

PyPlot.spy(M)

@time cfs = M \ [1; Zeros(size(M,1)-1)];
u = Fun(domainspace(Δ), cfs)
p = MultivariateTriangle.contourf(u)

u(0.1,0.2)

cfs1000 = cfs
cfs1001 = cfs