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

Debugger.@enter P'P