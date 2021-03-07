using MultivariateOrthogonalPolynomials, DifferentialEquations

N = 20
WZ = Weighted(Zernike(1))[:,Block.(Base.OneTo(N))]
Δ = Laplacian(axes(WZ,1))
(Δ*WZ).args[3].data |> typeof
using LazyArrays: arguments, ApplyLayout
arguments(ApplyLayout{typeof(*)}(), WZ)[2].data
@ent arguments(ApplyLayout{typeof(*)}(), WZ)

function heat!(du, u, (R,Δ), t)

end