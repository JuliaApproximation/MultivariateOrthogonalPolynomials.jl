"""
    DunklXuDisk(β)

are the orthogonal polynomials on the unit disk with respect to (1-x^2-y^2)^β, defined as `P_{n-k}^{k+β+1/2,k+β+1/2}(x)*(1-x^2)^{k/2}*P_k^{β,β}(y/sqrt{1-x^2})`.
"""
struct DunklXuDisk{T, V} <: BivariateOrthogonalPolynomial{T}
    β::V
end

DunklXuDisk(β::T) where T = DunklXuDisk{float(T), T}(β)
DunklXuDisk() = DunklXuDisk(0)

==(D1::DunklXuDisk, D2::DunklXuDisk) = D1.β == D2.β

axes(P::DunklXuDisk{T}) where T = (Inclusion(UnitDisk{T}()),blockedrange(oneto(∞)))

copy(A::DunklXuDisk) = A

Base.summary(io::IO, P::DunklXuDisk) = print(io, "DunklXuDisk($(P.β))")


"""
    DunklXuDiskWeight(β)

is a quasi-vector representing `(1-x^2-y^2)^β`.
"""
struct DunklXuDiskWeight{T, V} <: Weight{T}
    β::V
end

DunklXuDiskWeight(β::T) where T = DunklXuDiskWeight{float(T),T}(β)

axes(P::DunklXuDiskWeight{T}) where T = (Inclusion(UnitDisk{T}()),)

Base.summary(io::IO, P::DunklXuDiskWeight) = print(io, "(1-x^2-y^2)^$(P.β) on the unit disk")

const WeightedDunklXuDisk{T} = WeightedBasis{T,<:DunklXuDiskWeight,<:DunklXuDisk}

WeightedDunklXuDisk(β) = DunklXuDiskWeight(β) .* DunklXuDisk(β)

@simplify function *(Dx::PartialDerivative{1}, P::DunklXuDisk)
    β = P.β
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockBroadcastArray(hcat,
        ((k .+ (β - 1)) ./ (2k .+ (2β - 1)) .* (n .- k .+ 1)),
        Zeros((axes(n,1),)),
        (((k .+ 2β) .* (k .+ (2β + 1))) ./ ((2k .+ (2β - 1)) .* (2k .+ 2β)) .* (n .+ k .+ 2β) ./ 2)
        )
    DunklXuDisk(β+1) * _BandedBlockBandedMatrix(dat', axes(k,1), (-1,1), (0,2))
end

@simplify function *(Dy::PartialDerivative{2}, P::DunklXuDisk)
    β = P.β
    k = mortar(Base.OneTo.(oneto(∞)))
    T = promote_type(eltype(Dy), eltype(P)) # avoid bug in convert
    DunklXuDisk(β+1) * _BandedBlockBandedMatrix(((k .+ T(2β)) ./ 2)', axes(k,1), (-1,1), (-1,1))
end

@simplify function *(Dx::PartialDerivative{1}, w_P::WeightedDunklXuDisk)
    wP, P = w_P.args
    @assert P.β == wP.β
    β = P.β
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockBroadcastArray(hcat,
        (-4 .* (k .+ (β - 1)).*(n .- k .+ 1)./(2k .+ (2β - 1))),
        Zeros((axes(n,1),)),
        (-k .* (k .+ 1) ./ ((2k .+ (2β - 1)) .* (k .+ β)) .* (n .+ k .+ 2β))
        )
    WeightedDunklXuDisk(β-1) * _BandedBlockBandedMatrix(dat', axes(k,1), (1,-1), (2,0))
end

@simplify function *(Dy::PartialDerivative{2}, w_P::WeightedDunklXuDisk)
    wP, P = w_P.args
    @assert P.β == wP.β
    k = mortar(Base.OneTo.(oneto(∞)))
    T = promote_type(eltype(Dy), eltype(P)) # avoid bug in convert
    WeightedDunklXuDisk(P.β-1) * _BandedBlockBandedMatrix((T(-2).*k)', axes(k,1), (1,-1), (1,-1))
end

# P^{(β)} ↗ P^{(β+1)}
function dunklxu_raising(β)
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockHcat(
        BlockBroadcastArray(hcat,
            (-(k .+ (β - 1)) .* (2n .+ (2β - 1)) ./ ((2k .+ (2β - 1)) .* (4n .+ 4β))), # n-2, k-2
            Zeros((axes(n,1),)), # n-2, k-1
            (-(k .+ 2β) .* (k .+ (2β + 1)) ./ ((2k .+ (2β - 1)) .* (2k .+ 2β)) .* (2n .+ (2β-1)) ./ (8n .+ 8β))), # n-2, k
        Zeros((axes(n,1), Base.OneTo(3))),
        BlockBroadcastArray(hcat,
            (2 .* (k .+ (β - 1)) .* (n .- k .+ 1) .* (n .- k .+ 2) ./ ((2k .+ (2β-1)) .* (2n .+ 2β) .* (2n .+ (2β + 1)))), # n, k-2
            Zeros((axes(n,1),)), # n, k-1
            ((k .+ 2β) .* (k .+ (2β + 1)) ./ ((2k .+ (2β - 1)) .* (2k .+ 2β)) .* (n .+ k .+ 2β) .* (n .+ k .+ (2β + 1)) ./ ((2n .+ 2β) .* (2n .+ (2β + 1)))) # n, k
        ))
    _BandedBlockBandedMatrix(dat', axes(k,1), (0,2), (0,2))
end

function \(A::DunklXuDisk, B::DunklXuDisk)
    if A.β == B.β
        Eye((axes(B,2),))
    elseif A.β == B.β + 1
        dunklxu_raising(B.β)
    else
        error("not implemented for $A and $B")
    end
end

# (1-x^2-y^2) P^{(β)} ↘ P^{(β-1)}
function dunklxu_lowering(β)
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockHcat(
        BlockBroadcastArray(hcat,
            ((2n .+ (2β - 1)) ./ (n .+ β) .* (k .+ (β - 1)) ./ (2k .+ (2β - 1))), # n, k
            Zeros((axes(n,1),)), # n, k+1
            ((n .+ (β - 1/2)) ./ (n .+ β) .* k .* (k .+ 1) ./ ((2k .+ (2β - 1)) .* (2k .+ 2β)))), # n, k+2
        Zeros((axes(n,1), Base.OneTo(3))),
        BlockBroadcastArray(hcat,
            (-8 .* (n .- k .+ 1) .* (n .- k .+ 2) ./ ((2n .+ 2β) .* (2n .+ (2β + 1))) .* (k .+ (β - 1)) ./(2k .+ (2β - 1))), # n+2, k
            Zeros((axes(n,1),)), # n+2, k+1
            (-4 .* (n .+ k .+ 2β) .* (n .+ k .+ (2β + 1)) ./ ((2n .+ 2β) .* (2n .+ (2β + 1))) .* k .* (k .+ 1) ./ ((2k .+ (2β - 1)) .* (2k .+ 2β))) # n+2, k+2
        ))
    _BandedBlockBandedMatrix(dat', axes(k,1), (2,0), (2,0))
end

function \(w_A::WeightedDunklXuDisk, w_B::WeightedDunklXuDisk)
    wA,A = w_A.args
    wB,B = w_B.args

    @assert wA.β == A.β
    @assert wB.β == B.β

    if A.β == B.β
        Eye((axes(B,2),))
    elseif A.β + 1 == B.β
        dunklxu_lowering(B.β)
    else
        error("not implemented for $A and $B")
    end
end

\(w_A::DunklXuDisk, w_B::WeightedDunklXuDisk) =
    (DunklXuDiskWeight(0) .* w_A) \ w_B

# Actually Jxᵀ
function jacobimatrix(::Val{1}, P::DunklXuDisk)
    β = P.β
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockHcat(
        ((2n .+ (2β - 1)) ./ (4n .+ 4β)), # n-1, k
        Zeros((axes(n,1),)), # n, k
        ((n .- k .+ 1) .* (n .+ k .+ 2β) ./ ((n .+ β) .* (2n .+ (2β + 1)))), # n+1, k
        )
    _BandedBlockBandedMatrix(dat', axes(k,1), (1,1), (0,0))
end

# Actually Jyᵀ
function jacobimatrix(::Val{2}, P::DunklXuDisk)
    β = P.β
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockHcat(
        BlockBroadcastArray(hcat,
            ((k .+ (β - 1)) .* (2n .+ (2β - 1)) ./ ((2k .+ (2β - 1)) .* (2n .+ 2β))), # n-1, k-1
            Zeros((axes(n,1),)), # n-1, k
            (-k .* (k .+ 2β) .* (2n .+ (2β - 1)) ./ ((2k .+ (2β - 1)) .* (2k .+ 2β) .* (4n .+ 4β)))), # n-1, k+1
        Zeros((axes(n,1),Base.OneTo(3))),
        BlockBroadcastArray(hcat,
            (-(2k .+ (2β - 2)) .* (n .- k .+ 1) .* (n .- k .+ 2) ./ ((2k .+ (2β - 1)) .* (n .+ β) .* (2n .+ (2β + 1)))), # n+1, k-1
            Zeros((axes(n,1),)), # n+1, k
            (k .* (k .+ 2β) .* (n .+ k .+ 2β) .* (n .+ k .+ (2β + 1)) ./ ((2k .+ (2β - 1)) .* (2k .+ 2β) .* (n .+ β) .* (2n .+ (2β + 1)))))) # n+1, k+1
    _BandedBlockBandedMatrix(dat', axes(k,1), (1,1), (1,1))
end

@simplify function *(A::AngularMomentum, P::DunklXuDisk)
    β = P.β
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockBroadcastArray(hcat,
        (2 .* (k .+ (β - 1)) .* (n .- k .+ 1) ./ (2k .+ (2β - 1))), # n, k-1
        Zeros((axes(n,1),)), # n, k
        (-k .* (k .+ 2β) .* (n .+ k .+ 2β) ./ ((2k .+ (2β - 1)) .* (2k .+ 2β)))) # n, k+1
        
    DunklXuDisk(β) * _BandedBlockBandedMatrix(dat', axes(k,1), (0,0), (1,1))
end
