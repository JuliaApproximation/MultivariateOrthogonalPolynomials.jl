ClassicalOrthogonalPolynomials.checkpoints(d::UnitDisk{T}) where T = [SVector{2,T}(0.1,0.2), SVector{2,T}(0.2,0.3)]

struct Zernike{T} <: BivariateOrthogonalPolynomial{T} end

Zernike() = Zernike{Float64}()

axes(P::Zernike{T}) where T = (Inclusion(UnitDisk{T}()),blockedrange(oneto(∞)))

copy(A::Zernike) = A

# function getindex(Z::Zernike, rθ::RadialCoordinate, B::BlockIndex{1})
#     r,θ = rθ.r, rθ.θ
#     if isodd(block(B))
#         ℓ = blockindex(B)-1
#     m = block(B)
#     Normalized(Jacobi(0,m)
# end
