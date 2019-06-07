using ApproxFun, MultivariateOrthogonalPolynomials, BlockArrays, FillArrays

x,y = Fun(Triangle())
Δ = Laplacian() : TriangleWeight(1.0,1.0,1.0,JacobiTriangle(1.0,1.0,1.0))
V = x*y^2
L = Δ + V
M = L[Block.(1:500),Block.(1:500)];
@time cfs = M \ [1; Zeros(size(M,1)-1)];
u = Fun(domainspace(Δ), cfs)
u(0.1,0.2)
