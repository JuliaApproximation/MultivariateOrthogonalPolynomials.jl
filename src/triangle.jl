const UnitTriangle{T} = EuclideanUnitSimplex{2,T,:closed}

ClassicalOrthogonalPolynomials.checkpoints(d::UnitTriangle{T}) where T = [SVector{2,T}(0.1,0.2), SVector{2,T}(0.2,0.3)]

struct Triangle{T} <: Domain{SVector{2,T}}
    a::SVector{2,T}
    b::SVector{2,T}
    c::SVector{2,T}
    Triangle{T}(a::SVector{2,T}, b::SVector{2,T}, c::SVector{2,T}) where T = new{T}(a, b, c)
end

Triangle() = Triangle(SVector(0,0), SVector(1,0), SVector(0,1))
Triangle(a, b, c) = Triangle{promote_type(eltype(eltype(a)), eltype(eltype(b)), eltype(eltype(c)))}(a,b, c)
Triangle{T}(d::Triangle) where T = Triangle{T}(d.a, d.b, d.c)
"""
    equilateraltriangle([T=Float64])

The equilateral triangle centred at the origin whose vertices are the cube roots of unity
`(1,0)`, `(-1/2,√3/2)`, `(-1/2,-√3/2)` (i.e. `1, ω, ω²` for `ω = exp(2πi/3)`).
"""
function equilateraltriangle(::Type{T}=Float64) where T
    s = sqrt(T(3))/2
    Triangle(SVector{2,T}(1,0), SVector{2,T}(-1/2,s), SVector{2,T}(-1/2,-s))
end

Triangle{T}(a, b, c) where T = Triangle{T}(convert(SVector{2,T}, a), convert(SVector{2,T}, b), convert(SVector{2,T}, c))

==(A::Triangle, B::Triangle) = A.a == B.a && A.b == B.b && A.c == B.c

Inclusion(d::Triangle{T}) where T = Inclusion{SVector{2,float(T)}}(d)

function tocanonical(d::Triangle, 𝐱::AbstractVector)
    if d.a == SVector(0,0)
        [d.b d.c] \ 𝐱
    else
        tocanonical(d-d.a, 𝐱-d.a)
    end
end


function fromcanonical(d::Triangle, 𝐱::AbstractVector)
    if d.a == SVector(0,0)
        [d.b d.c]*𝐱
    else
        fromcanonical(d-d.a, 𝐱) + d.a
    end
end

fromcanonical(d::UnitTriangle, 𝐱::AbstractVector) = 𝐱
tocanonical(d::UnitTriangle, 𝐱::AbstractVector) = 𝐱

function getindex(a::ContinuumArrays.AffineMap{<:Any, <:Inclusion{<:Any,<:Union{Triangle,UnitTriangle}}, <:Inclusion{<:Any,<:Union{Triangle,UnitTriangle}}}, x::SVector{2})
    checkbounds(a, x)
    fromcanonical(a.range.domain, tocanonical(a.domain.domain, x))
end


# canonicaldomain(::Triangle) = Triangle()
function in(p::SVector{2}, d::Triangle)
    x,y = tocanonical(d, p)
    0 ≤ x ≤ x + y ≤ 1
end


for op in (:-, :+)
    @eval begin
        $op(d::Triangle, x::SVector{2}) = Triangle($op(d.a,x), $op(d.b,x), $op(d.c,x))
        $op(x::SVector{2}, d::Triangle) = Triangle($op(x,d.a), $op(x,d.b), $op(x,d.c))
    end
end

for op in (:*, :/)
    @eval $op(d::Triangle, x::Number) = Triangle($op(d.a,x), $op(d.b,x), $op(d.c,x))
end

for op in (:*, :\)
    @eval $op(x::Number, d::Triangle) = Triangle($op(x,d.a), $op(x,d.b), $op(x,d.c))
end



# expansion in OPs orthogonal to
# x^a*y^b*(1-x-y)^c
# defined as
# P_{n-k}^{2k+b+c+1,a}(2x-1)*(1-x)^k*P_k^{c,b}(2y/(1-x)-1)


struct JacobiTriangle{T,V} <: BivariateOrthogonalPolynomial{T}
    a::V
    b::V
    c::V
end


JacobiTriangle{T}(a::V,b::V,c::V) where {T,V} = JacobiTriangle{T,V}(a, b, c)
JacobiTriangle(a::T, b::T, c::T) where T = JacobiTriangle{float(T),T}(a, b, c)
JacobiTriangle() = JacobiTriangle(0,0,0)
JacobiTriangle{T}() where T = JacobiTriangle{T}(0,0,0)
==(K1::JacobiTriangle, K2::JacobiTriangle) = K1.a == K2.a && K1.b == K2.b && K1.c == K2.c

axes(P::JacobiTriangle{T}) where T = (Inclusion(UnitTriangle{T}()),blockedrange(oneto(∞)))

copy(A::JacobiTriangle) = A

show(io::IO, P::JacobiTriangle) = summary(io, P)
summary(io::IO, P::JacobiTriangle) = print(io, "JacobiTriangle($(P.a), $(P.b), $(P.c))")


basis_axes(::Inclusion{<:Any,<:UnitTriangle}, v) = JacobiTriangle()

"""
    TriangleWeight(a, b, c)

is a quasi-vector representing `x^a * y^b * (1-x-y)^c`.
"""
struct TriangleWeight{T,V} <: Weight{T}
    a::V
    b::V
    c::V
end

TriangleWeight(a::T, b::T, c::T) where T = TriangleWeight{float(T),T}(a, b, c)

const WeightedTriangle{T,V} = Weighted{T,JacobiTriangle{T,V}}

WeightedTriangle(a, b, c) = Weighted(JacobiTriangle(a,b,c))

axes(P::TriangleWeight{T}) where T = (Inclusion(UnitTriangle{T}()),)
function getindex(P::TriangleWeight, xy::StaticVector{2})
    x,y = xy
    x^P.a * y^P.b * (1-x-y)^P.c
end

all(::typeof(isone), w::TriangleWeight) = iszero(w.a) && iszero(w.b) && iszero(w.c)
==(w::TriangleWeight, v::TriangleWeight) = w.a == v.a && w.b == v.b && w.c == v.c

==(wA::WeightedTriangle, B::JacobiTriangle) = wA.P == B == JacobiTriangle(0,0,0)
==(B::JacobiTriangle, wA::WeightedTriangle) = wA == B

==(w_A::WeightedBasis{<:Any,<:TriangleWeight,<:JacobiTriangle}, w_B::WeightedBasis{<:Any,<:TriangleWeight,<:JacobiTriangle}) = arguments(w_A) == arguments(w_B)

function ==(w_A::WeightedBasis{<:Any,<:TriangleWeight,<:JacobiTriangle}, B::JacobiTriangle)
    w,A = arguments(w_A)
    all(isone,w) && A == B
end

==(B::JacobiTriangle, w_A::WeightedBasis{<:Any,<:TriangleWeight,<:JacobiTriangle}) = w_A == B

function ==(w_A::WeightedBasis{<:Any,<:TriangleWeight,<:JacobiTriangle}, wB::WeightedTriangle)
    w,A = arguments(w_A)
    w.a == A.a && w.b == A.b && w.c == A.c && A == wB.P
end

==(wB::WeightedTriangle, w_A::WeightedBasis{<:Any,<:TriangleWeight,<:JacobiTriangle}) = w_A == wB

show(io::IO, P::TriangleWeight) = summary(io, P)
summary(io::IO, P::TriangleWeight) = print(io, "x^$(P.a)*y^$(P.b)*(1-x-y)^$(P.c) on the unit triangle")

orthogonalityweight(P::JacobiTriangle) = TriangleWeight(P.a, P.b, P.c)

function diff(P::JacobiTriangle{T}, ::Val{(1,0)}; dims=1) where T
    @assert dims == 1
    a,b,c = P.a,P.b,P.c
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockBroadcastArray(hcat,
        ((k .+ (b-1)) .* (n .+ k .+ (b+c-1)) ./ (2k .+ (b+c-1))),
        ((n .+ k .+ (a+b+c)) .* (k .+ (b+c)) ./ (2k .+ (b+c-1)))
        )
    JacobiTriangle(a+1,b,c+1) * _BandedBlockBandedMatrix(dat', axes(k,1), (-1,1), (0,1))
end

function diff(P::JacobiTriangle{T}, ::Val{(0,1)}; dims=1) where T
    @assert dims == 1
    a,b,c = P.a,P.b,P.c
    k = mortar(Base.OneTo.(oneto(∞)))
    JacobiTriangle(a,b+1,c+1) * _BandedBlockBandedMatrix((k .+ convert(T, b+c))', axes(k,1), (-1,1), (-1,1))
end

###
# Affinely mapped Koornwinder bases
#
# If `kr = affine(d, UnitTriangle())` then `Q = P[kr, :]` is `P` composed with the map `u = kr(x)`.
# With `J = ∂u/∂x` constant, the chain rule gives ∂Q/∂xⱼ = Σᵢ J[i,j] (∂P/∂uᵢ)∘kr and
# ∫_d Q dx = ∫_{unit} P du / |det J|.
###

const TriangleDomain = Union{Triangle,UnitTriangle}
const TriangleMap = ContinuumArrays.AffineMap{<:Any,<:Inclusion{<:Any,<:TriangleDomain},<:Inclusion{<:Any,<:TriangleDomain}}

_vertices(d::Triangle) = (d.a, d.b, d.c)
_vertices(::UnitTriangle{T}) where T = (SVector{2,T}(0,0), SVector{2,T}(1,0), SVector{2,T}(0,1))
function _edgematrix(d)
    a,b,c = _vertices(d)
    [b-a c-a]
end

# the (constant) Jacobian ∂(range)/∂(domain) of an affine map between triangles
triangle_jacobian(kr) = _edgematrix(kr.range.domain) / _edgematrix(kr.domain.domain)

ContinuumArrays.affinemap_A(m::TriangleMap) = triangle_jacobian(m)

# TODO: replace this with a generic version overloading 
#  diff_mapped(kr::AbstractAffineQuasiVector, V, order::NTuple{2,Int}; dims...)
# possibly using the definition
#   (J[1,i] *diff(P, Val((1,0))) + J[2,i] * diff(P, Val((0,1))))[basismap(V), :]

function ContinuumArrays.diff_mapped(kr::AbstractAffineQuasiVector, V, (k,j)::NTuple{2,Int}; dims...)
    (k < 0 || j < 0) && throw(ArgumentError("order must be non-negative"))
    k == j == 0 && return V
    if k + j == 1
        P = demap(V)
        J = kr.A
        i = k == 1 ? 1 : 2 # differentiate with respect to xᵢ
        Pu, Mu = arguments(diff(P, Val((1,0))))
        Pv, Mv = arguments(diff(P, Val((0,1))))
        Pc = broadcastbasis(+, Pu, Pv)
        return Pc[kr, :] * (J[1,i] * ((Pc \ Pu) * Mu) + J[2,i] * ((Pc \ Pv) * Mv))
    end
    k ≥ j && return diff(diff(V, (1,0)), (k-1,j))
    diff(diff(V, (0,1)), (k,j-1))
end

diff_layout(::MappedBasisLayouts, V, order::NTuple{2,Int}; dims...) = diff_mapped(basismap(V), V, order; dims...)

# The weighted derivative formulae lower the parameters, which is only valid if the
# weight vanishes on the relevant edges. Otherwise we avoid negative parameters by
# converting to JacobiTriangle(0,0,0) and then differentiating.
function _unweighted_diff(w_P::WeightedTriangle{T}, order) where T
    P = JacobiTriangle{T}()
    Q, D = arguments(diff(P, order))
    Q * (D * (P \ w_P))
end

function diff(w_P::WeightedTriangle{T}, order::Val{(1,0)}; dims=1) where T
    @assert dims == 1
    a, b, c = w_P.P.a, w_P.P.b, w_P.P.c
    (iszero(a) || iszero(c)) && return _unweighted_diff(w_P, order)
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))    
    scale = -(2k .+ (b + c - 1))
    coeff1 = (k .+ (c - 1)) .* (n .- k .+ 1) ./ scale 
    coeff2 = k .* (n .- k .+ a) ./ scale 
    dat = BlockBroadcastArray(hcat, coeff1, coeff2)
    WeightedTriangle(a-1, b, c-1) * _BandedBlockBandedMatrix(dat', axes(k, 1), (1, -1), (1, 0))
end

function diff(w_P::WeightedTriangle{T}, order::Val{(0,1)}; dims=1) where T
    @assert dims == 1
    a, b, c = w_P.P.a, w_P.P.b, w_P.P.c 
    (iszero(b) || iszero(c)) && return _unweighted_diff(w_P, order)
    k = mortar(Base.OneTo.(oneto(∞)))
    WeightedTriangle(a, b-1, c-1) * _BandedBlockBandedMatrix(-one(T) * k', axes(k, 1), (1, -1), (1, -1))
end

function grammatrix(A::JacobiTriangle)
    @assert A == JacobiTriangle()
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    _BandedBlockBandedMatrix(BroadcastVector{eltype(A)}((n,k) -> exp(loggamma(n+k)+loggamma(n-k+1)+loggamma(k)+loggamma(k)-loggamma(n+k)-loggamma(k)-loggamma(n-k+1)-loggamma(k))/((2n)*(2k-1)), n, k)',
    axes(k,1), (0,0), (0,0))
end

function weightedgrammatrix(A::JacobiTriangle)
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    a,b,c = A.a,A.b,A.c
    k = mortar(Base.OneTo.(oneto(∞)))
    _BandedBlockBandedMatrix(BroadcastVector{eltype(A)}((n,k,a,b,c) -> exp(loggamma(n+k+b+c)+loggamma(n-k+a+1)+loggamma(k+b)+loggamma(k+c)-loggamma(n+k+a+b+c)-loggamma(k+b+c)-loggamma(n-k+1)-loggamma(k))/((2n+a+b+c)*(2k+b+c-1)), n, k, a, b, c)',
    axes(k,1), (0,0), (0,0))
end

"""
    Conjugate(A,B)

represents the matrix `Q'A*Q`. `Q` may not be orthogonal.
"""
struct Conjugate{T, AA, QQ} <: LazyMatrix{T}
    A::AA
    Q::QQ
end

Conjugate{T}(A, Q) where T = Conjugate{T,typeof(A),typeof(Q)}(A, Q)
Conjugate(A, Q) = Conjugate{promote_type(eltype(A), eltype(Q))}(A, Q)

Base.array_summary(io::IO, B::Conjugate{T}, inds) where T = print(io, Base.dims2string(length.(inds)), " Conjugate{$T}")


MemoryLayout(::Type{<:Conjugate}) = ApplyBandedBlockBandedLayout{typeof(*)}()
function axes(A::Conjugate)
    _,ax = axes(A.Q)
    (ax,ax)
end
function size(A::Conjugate)
    _,sz = size(A.Q)
    (sz,sz)
end

@inline arguments(::Union{ApplyLayout{typeof(*)},ApplyBandedBlockBandedLayout{typeof(*)}}, A::Conjugate) = (A.Q', A.A, A.Q)
@inline ApplyMatrix(A::Conjugate{T}) where T = ApplyMatrix{T}(*, arguments(A)...)
blockbandwidths(A::Conjugate) = blockbandwidths(ApplyMatrix(A))
subblockbandwidths(A::Conjugate) = subblockbandwidths(ApplyMatrix(A))

getindex(A::Conjugate{T}, k::Int, j::Int) where T = ApplyMatrix(A)[k,j]::T
function getindex(A::Conjugate, KR::BlockRange{1,Tuple{OneTo{Int}}}, JR::BlockRange{1,Tuple{OneTo{Int}}})
    KR ≠ JR && return ApplyMatrix(A)[KR,JR]
    MR = blockcolsupport(A.Q, KR)
    B = BandedBlockBandedMatrix(A.Q[MR, KR])
    B' * BandedBlockBandedMatrix(A.A[MR,MR]) * B
end


function grammatrix(W::Weighted{T,<:JacobiTriangle}) where T
    P = JacobiTriangle{T}()
    L = P \ W
    Conjugate(grammatrix(P), L)
end

function Wy(a,b,c)
    k = mortar(Base.OneTo.(oneto(∞)))
    _BandedBlockBandedMatrix((-k)', axes(k,1), (1,-1), (1,-1))
end
function Wx(a,b,c)
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockVcat(
        ((k .+ (c-1)) .* ( k .- n .- 1 ) ./ (2k .+ (b+c-1)))',
        (k .* (k .- n .- a) ./ (2k .+ (b+c-1)))'
        )
    _BandedBlockBandedMatrix(dat, axes(k,1), (1,-1), (1,0))
end

 function Rx(a,b,c)
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockHcat(
        BroadcastVector((n,k,bc1,abc) -> (n + k +  bc1) / (2n + abc), n, k, b+c-1, a+b+c),
        BroadcastVector((n,k,abc) -> (n + k +  abc) / (2n + abc), n, k, a+b+c)
        )
    _BandedBlockBandedMatrix(dat', axes(k,1), (0,1), (0,0))
end
function Ry(a,b,c)
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockHcat(
        BlockBroadcastArray(hcat,
            ((k .+ (c-1) ) .* (n .+ k .+ (b+c-1)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) ))),
            ((k .- n .- a ) .* (k .+ (b+c)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))),
        BlockBroadcastArray(hcat,
            ((k .+ (c-1) ) .* (k .- n .- 1) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) ))),
            ((n .+ k .+ (a+b+c) ) .* (k .+ (b+c)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))))
    _BandedBlockBandedMatrix(dat', axes(k,1), (0,1), (0,1))
end

function Rz(a,b,c)
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockHcat(
        BlockBroadcastArray(hcat,
            (-(k .+ (b-1) ) .* (n .+ k .+ (b+c-1)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) ))),
            ((k .- n .- a ) .* (k .+ (b+c)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))),
        BlockBroadcastArray(hcat,
            (-(k .+ (b-1) ) .* (k .- n .- 1) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) ))),
            ((n .+ k .+ (a+b+c) ) .* (k .+ (b+c)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))))
    _BandedBlockBandedMatrix(dat', axes(k,1), (0,1), (0,1))
end

function Lx(a,b,c)
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockHcat(
        ((n .- k .+ a) ./ (2n .+ (a+b+c))),
        ((n .- k .+ 1) ./ (2n .+ (a+b+c))))
    _BandedBlockBandedMatrix(dat', axes(k,1), (1,0), (0,0))
end
function Ly(a,b,c)
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockHcat(
        BlockBroadcastArray(hcat,
            ((k .+ (b-1) ) .* (n .+ k .+ (b+c-1)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) ))),
            (k .* (k .- n .- a ) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))),
        BlockBroadcastArray(hcat,
            ((k .+ (b-1) ) .* (k .- n .- 1) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) ))),
            ( k .* (n .+ k .+ (a+b+c) ) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))))
    _BandedBlockBandedMatrix(dat', axes(k,1), (1,0), (1,0))
end
function Lz(a,b,c)
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockHcat(
        BlockBroadcastArray(hcat,
            ((k .+ (c-1) ) .* (n .+ k .+ (b+c-1)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) ))),
            (k .* (n .- k .+ a ) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))),
        BlockBroadcastArray(hcat,
            ((k .+ (c-1) ) .* (k .- n .- 1) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) ))),
            ( (-k) .* (n .+ k .+ (a+b+c) ) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))))
    _BandedBlockBandedMatrix(dat', axes(k,1), (1,0), (1,0))
end


function \(w_A::WeightedTriangle, w_B::WeightedTriangle)
    A,B = w_A.P,w_B.P

    if A.a == B.a && A.b == B.b && A.c == B.c
        Eye{promote_type(eltype(A),eltype(B))}((axes(B,2),))
    elseif A.a + 1 == B.a && A.b == B.b && A.c == B.c
        Lx(B.a, B.b, B.c)
    elseif A.a == B.a && A.b + 1 == B.b && A.c == B.c
        Ly(B.a, B.b, B.c)
    elseif A.a == B.a && A.b == B.b && A.c + 1 == B.c
        Lz(B.a, B.b, B.c)
    elseif A.a < B.a
        w_C = WeightedTriangle(A.a+1,A.b,A.c)
        (w_A \ w_C) * (w_C \ w_B)
    elseif A.b < B.b
        w_C = WeightedTriangle(A.a,A.b+1,A.c)
        (w_A \ w_C) * (w_C \ w_B)
    elseif A.c < B.c
        w_C = WeightedTriangle(A.a,A.b,A.c+1)
        (w_A \ w_C) * (w_C \ w_B)
    else
        error("not implemented for $w_A and $w_B")
    end
end

function \(w_A::WeightedBasis{<:Any,<:TriangleWeight,<:JacobiTriangle}, w_B::WeightedTriangle)
    wA,A = w_A.args
    w_A == Weighted(A) && Weighted(A) \ w_B
    all(isone,wA) && return A \ w_B
    error("Not implemented")
end

function \(w_A::WeightedTriangle, w_B::WeightedBasis{<:Any,<:TriangleWeight,<:JacobiTriangle})
    wB,B = w_B.args
    w_B == Weighted(B) && w_A \ Weighted(B)
    all(isone,wB) && return w_A \ B
    error("Not implemented")
end


function \(w_A::WeightedBasis{<:Any,<:TriangleWeight,<:JacobiTriangle}, w_B::WeightedBasis{<:Any,<:TriangleWeight,<:JacobiTriangle})
    wA,A = w_A.args
    w_A == Weighted(A) && Weighted(A) \ w_B
    all(isone,wA) && return A \ w_B
    error("Not implemented")
end

function \(w_A::WeightedBasis{<:Any,<:TriangleWeight,<:JacobiTriangle}, B::JacobiTriangle)
    wA,A = w_A.args
    w_A == Weighted(A) && return Weighted(A) \ B
    all(isone,wA) && return A \ B
    error("Not implemented")
end

function \(A::JacobiTriangle, w_B::WeightedBasis{<:Any,<:TriangleWeight,<:JacobiTriangle})
    wB,B = w_B.args
    w_B == Weighted(B) && return A \ Weighted(B)
    all(isone,wB) && return A \ B
    error("Not implemented")
end

function \(A::JacobiTriangle, w_B::WeightedTriangle)
    w_B.P == JacobiTriangle() && return A \ w_B.P
    A == JacobiTriangle() && return Weighted(A) \ w_B
    P = JacobiTriangle{eltype(A)}()
    (A \ P) * (Weighted(P) \ w_B) # lower to Legendre then raise
end
function \(w_A::WeightedTriangle, B::JacobiTriangle)
    w_A.P == JacobiTriangle() && return w_A.P \ B
    w_A \ (TriangleWeight(0,0,0) .* B)
end

@simplify function \(A::JacobiTriangle, B::JacobiTriangle)
    if A == B
        Eye{promote_type(eltype(A),eltype(B))}((axes(B,2),))
    elseif A.a == B.a + 1 && A.b == B.b && A.c == B.c
        Rx(B.a, B.b, B.c)
    elseif A.a == B.a && A.b == B.b + 1 && A.c == B.c
        Ry(B.a, B.b, B.c)
    elseif A.a == B.a && A.b == B.b && A.c == B.c + 1
        Rz(B.a, B.b, B.c)
    elseif A.a > B.a
        C = JacobiTriangle(B.a+1,B.b,B.c)
        (A \ C) * (C \ B)
    elseif A.b > B.b
        C = JacobiTriangle(B.a,B.b+1,B.c)
        (A \ C) * (C \ B)
    elseif A.c > B.c
        C = JacobiTriangle(B.a,B.b,B.c+1)
        (A \ C) * (C \ B)
    else
        error("not implemented for $A and $B")
    end
end


# the basis for a sum is the one all terms can be converted to with banded-block-banded operators:
# raise the parameters for polynomials, lower them for weighted polynomials.
broadcastbasis(::typeof(+), A::JacobiTriangle{T}, B::JacobiTriangle{V}) where {T,V} =
    JacobiTriangle{promote_type(T,V)}(promote(max(A.a,B.a), max(A.b,B.b), max(A.c,B.c))...)
broadcastbasis(::typeof(+), w_A::WeightedTriangle{T}, w_B::WeightedTriangle{V}) where {T,V} =
    Weighted(JacobiTriangle{promote_type(T,V)}(promote(min(w_A.P.a,w_B.P.a), min(w_A.P.b,w_B.P.b), min(w_A.P.c,w_B.P.c))...))
broadcastbasis(::typeof(+), A::JacobiTriangle, w_B::WeightedTriangle) = broadcastbasis(+, A, JacobiTriangle{eltype(w_B)}())
broadcastbasis(::typeof(+), w_A::WeightedTriangle, B::JacobiTriangle) = broadcastbasis(+, B, w_A)


## Jacobi matrix

jacobimatrix(::Val{1}, P::JacobiTriangle) = Lx(P.a+1, P.b, P.c) * Rx(P.a, P.b, P.c)
jacobimatrix(::Val{2}, P::JacobiTriangle) = Ly(P.a, P.b+1, P.c) * Ry(P.a, P.b, P.c)




## Evaluation

"""
Represents B⁺, the pseudo-inverse defined in forward recurrence.
"""
struct TriangleRecurrenceA{T} <: AbstractMatrix{T}
    bˣ::Vector{T}
    d::NTuple{3,T}  # last row entries
end

"""
Represents -B⁺*[Aˣ; Aʸ].
"""
struct TriangleRecurrenceB{T} <: AbstractMatrix{T}
    aˣ::Vector{T}
    bˣ::Vector{T}
    d::NTuple{2,T}
end

"""
Represents B⁺*[Cˣ; Cʸ].
"""
struct TriangleRecurrenceC{T} <: AbstractMatrix{T}
    bˣ::Vector{T}
    cˣ::Vector{T}
    d::T  # last row entries
end

function TriangleRecurrenceA(n, X, Y)
    bˣ = view(X,Block(n+1,n))[band(0)]
    Bʸ = view(Y,Block(n+1,n))'
    d₂ = inv(Bʸ[n,n+1])
    d₁ = -Bʸ[n,n] * d₂/bˣ[n]
    # max avoids bounds error, result unused
    d₀ = -Bʸ[n,max(1,n-1)] * d₂/bˣ[max(1,n-1)]
    TriangleRecurrenceA(bˣ, (d₀, d₁, d₂))
end

function TriangleRecurrenceB(n, X, Y)
    aˣ = view(X,Block(n,n))[band(0)]
    bˣ = view(X,Block(n+1,n))[band(0)]
    Aʸ = view(Y,Block(n,n))'
    Bʸ = view(Y,Block(n+1,n))'
    d₂ = inv(Bʸ[n,n+1])
    d₁ = Bʸ[n,n] * d₂/bˣ[n]*aˣ[n]-Aʸ[n,n]*d₂
    # max avoids bounds error, result unused
    d₀ = Bʸ[n,max(1,n-1)] * d₂ * aˣ[max(1,n-1)]/bˣ[max(1,n-1)] - Aʸ[n,max(1,n-1)]*d₂
    TriangleRecurrenceB(aˣ, bˣ, (d₀, d₁))
end

function TriangleRecurrenceC(n, X, Y)
    cˣ = view(X,Block(n-1,n))[band(0)]
    bˣ = view(X,Block(n+1,n))[band(0)]
    Cʸ = view(Y,Block(n-1,n))'
    Bʸ = view(Y,Block(n+1,n))'
    d₂ = inv(Bʸ[n,n+1])
    d = -Bʸ[n,n-1]/bˣ[n-1]*cˣ[n-1]*d₂ + Cʸ[n,n-1]*d₂
    TriangleRecurrenceC(bˣ, cˣ, d)
end

function size(A::TriangleRecurrenceA)
    n = length(A.bˣ)
    (n+1,2n)
end

function size(BA::TriangleRecurrenceB)
    n = length(BA.bˣ)
    (n+1, n)
end


function size(BA::TriangleRecurrenceC)
    n = length(BA.bˣ)
    (n+1, n-1)
end



function getindex(A::TriangleRecurrenceA{T}, k::Int, j::Int) where T
    n,m = size(A)
    if k < n && j == k
        return inv(A.bˣ[k])
    elseif k == n
        d₀, d₁, d₂ = A.d
        if j == m
            return d₂
        elseif j == n-1
            return d₁
        elseif j == n-2
            return d₀
        end
    end
    return zero(T)
end

function getindex(A::TriangleRecurrenceB{T}, k::Int, j::Int) where T
    n,m = size(A)
    if k < n && j == k
        return -A.aˣ[k] / A.bˣ[k]
    elseif k == n
        d₀, d₁ = A.d
        if j == m
            return d₁
        elseif j == m-1
            return d₀
        end
    end
    return zero(T)
end

function getindex(A::TriangleRecurrenceC{T}, k::Int, j::Int) where T
    n,m = size(A)
    if k ≤ m && j == k
        return A.cˣ[k] / A.bˣ[k]
    elseif k == n && j == m
        return A.d
    end
    return zero(T)
end



function xy_muladd!((x,y), A::TriangleRecurrenceA, v::AbstractVector, β, w::AbstractVector)
    n = size(A,1)
    d₀, d₁, d₂ = A.d
    w_1 = view(w, 1:n-1)
    w_1 .= A.bˣ .\ x .* v .+ β .* w_1
    if n > 2
        w[n] = d₀*x*v[n-2] + (d₁*x+d₂*y)*v[n-1] + β*w[n]
    else
        w[n] = (d₁*x+d₂*y)*v[n-1] + β*w[n]
    end
    w
end

function xy_muladd!(xy, Ac::Adjoint{<:Any,<:TriangleRecurrenceA}, v::AbstractVector, β, w::AbstractVector)
    x,y = xy
    A = parent(Ac)
    n = size(A,1)
    d₀, d₁, d₂ = A.d
    w .= A.bˣ .\ x .* view(v,1:n-1) .+ β .* w
    if n > 2
        w[n-2] += d₀*x*v[n]
        w[n-1] += (d₁*x+d₂*y)*v[n]
    else
        w[n-1] += (d₁*x+d₂*y)*v[n]
    end
    w
end


function ArrayLayouts.muladd!(α::T, BA::TriangleRecurrenceB{T}, c::AbstractVector{T}, β::T, dest::AbstractVector{T}) where T
    @boundscheck (size(BA,2) == length(c) && size(BA,1) == length(dest)) || throw(DimensionMismatch())
    aˣ,bˣ = BA.aˣ,BA.bˣ
    Ba_1, Ba_2 = BA.d

    n = length(bˣ)
    w = view(dest,1:n)
    w .= (-α) .* (bˣ .\ aˣ .* c) .+ β .* w
    dest[n+1] = α*(n > 1 ? Ba_1 * c[end-1] + Ba_2 * c[end] : Ba_2 * c[end]) + β*dest[n+1]
    dest
end

function ArrayLayouts.muladd!(α::T, BAc::Adjoint{T,<:TriangleRecurrenceB{T}}, c::AbstractVector{T}, β::T, dest::AbstractVector{T}) where T
    @boundscheck (size(BAc,2) == length(c) && size(BAc,1) == length(dest)) || throw(DimensionMismatch())
    BA = parent(BAc)
    aˣ,bˣ = BA.aˣ,BA.bˣ
    Ba_1, Ba_2 = BA.d

    n = length(bˣ)
    dest .= (-α) .* (bˣ .\ aˣ .* view(c,1:n)) .+ β .* dest
    if n > 1
        dest[end-1] += α*Ba_1*c[n+1]
    end
    dest[end] += α*Ba_2*c[n+1]
    dest
end


function ArrayLayouts.muladd!(α::T, BA::TriangleRecurrenceC{T}, u::AbstractVector{T}, β::T, dest::AbstractVector{T}) where T
    @boundscheck (size(BA,2) == length(u) && size(BA,1) == length(dest)) || throw(DimensionMismatch())
    bˣ,cˣ = BA.bˣ,BA.cˣ
    d = BA.d

    n = length(bˣ)
    w = view(dest,1:n-1)
    w .= α .* (view(bˣ,1:n-1) .\ cˣ .* u) .+ β .* w
    dest[n] *= β
    dest[n+1] = α*d * u[end] + β*dest[n+1]
    dest
end

function ArrayLayouts.muladd!(α::T, BAc::Adjoint{T,TriangleRecurrenceC{T}}, u::AbstractVector{T}, β::T, w::AbstractVector{T}) where T
    @boundscheck (size(BAc,2) == length(u) && size(BAc,1) == length(w)) || throw(DimensionMismatch())
    BA = parent(BAc)
    bˣ,cˣ = BA.bˣ,BA.cˣ
    d = BA.d

    n = length(bˣ)
    w .= α .* (view(bˣ,1:n-1) .\ cˣ .* view(u,1:n-1)) .+ β .* w
    w[n-1] += α*d*u[n+1]

    w
end

# following also resizes
function LinearAlgebra.lmul!(BAc::Adjoint{T,TriangleRecurrenceC{T}}, u::AbstractVector{T}) where T
    @boundscheck size(BAc,2) == length(u) || throw(DimensionMismatch())
    BA = parent(BAc)
    bˣ,cˣ = BA.bˣ,BA.cˣ
    d = BA.d

    n = length(bˣ)
    v = view(u,1:n-1)
    v .= view(bˣ,1:n-1) .\ cˣ .* v
    u[n-1] += d*u[n+1]

    resize!(u, n-1)
end

function LinearAlgebra.mul!(dest::AbstractVector{T}, BA::Union{TriangleRecurrenceB,TriangleRecurrenceC,Adjoint{<:Any,<:TriangleRecurrenceB},Adjoint{<:Any,<:TriangleRecurrenceC}}, c::AbstractVector) where T
    ArrayLayouts.zero!(dest) # avoid NaN
    muladd!(one(T), BA, c, zero(T), dest)
end

##
# TODO: almost generic, but left specialised for now because we need
# to make block indexing of infinite Jacobi operators fast.
# This has to do with recognizing a block-row vector with banded blocks
# times a block-col vec with banded blocks is banded.
# The functionality will need to be added to LazyBandedMatrices.jl
function tri_forwardrecurrence(N::Int, X, Y, x, y)
    T = promote_type(eltype(X),eltype(Y))
    ret = BlockedVector{T}(undef, 1:N)
    N < 1 && return ret
    ret[1] = 1
    N < 2 && return ret

    X_N = X[Block.(1:N), Block.(1:N-1)]
    Y_N = Y[Block.(1:N), Block.(1:N-1)]

    let n = 1
        A = TriangleRecurrenceA(n, X_N, Y_N)
        B = TriangleRecurrenceB(n, X_N, Y_N)
        ret[Block(2)] .= A*[x; y] + vec(B)
    end

    for n = 2:N-1
        # P[n+1,xy] == (A*[x*Eye(n); y*Eye(n)] + B)*P[n,xy] - C*P[n-1,xy]
        A = TriangleRecurrenceA(n, X_N, Y_N)
        B = TriangleRecurrenceB(n, X_N, Y_N)
        C = TriangleRecurrenceC(n, X_N, Y_N)

        p_0 = view(ret,Block(n-1))
        p_1 = view(ret,Block(n))
        p_2 = view(ret,Block(n+1))

        mul!(p_2, B, p_1)
        xy_muladd!((x,y), A, p_1, one(T), p_2)
        muladd!(-one(T), C, p_0, one(T), p_2)
    end
    ret
end



###
# Evaluation
###

# Block(n+1)[k+1] of JacobiTriangle(a,b,c) is P_{n-k}^{(2k+b+c+1,a)}(2x-1) * Q_k(x,y) where
# Q_k(x,y) = (1-x)^k * P_k^{(c,b)}(2y/(1-x)-1). These are stored as the entry [n-k+1,k+1] of the matrix of a DiagTrav.
# On the triangle 2y/(1-x)-1 is in [-1,1], except at the corner x = 1 (where y = 0) where Q_0 = 1 and Q_k = 0 for k > 0.

# The first N blocks: one forward recurrence for the Q_k and one for each column, costing O(N^2) operations.
function getindex(P::JacobiTriangle{T}, xy::SVector{2}, JR::BlockOneTo) where T
    x, y = xy
    N = length(JR)
    R = polynomialtype(T, eltype(xy))
    M = zeros(R, N, N)
    N == 0 && return DiagTrav(M)
    # Q_k in the first row
    W = 1-x
    if iszero(W)
        M[1,1] = one(R)
    else
        copyto!(view(M, 1, :), view(Jacobi{R}(P.c, P.b), 2y/W-1, oneto(N)))
        view(M, 1, :) .*= W .^ (0:N-1)
    end
    for k = 0:N-1
        Qₖ = M[1,k+1]
        col = view(M, 1:N-k, k+1)
        copyto!(col, view(Jacobi{R}(2k+P.b+P.c+1, P.a), 2x-1, oneto(N-k)))
        lmul!(Qₖ, col)
    end
    DiagTrav(M)
end

# Points given as tuples are converted first, so that indexing by ranges of blocks/entries uses the evaluation of
# the first N blocks above, instead of falling back to evaluating entry-by-entry.
Base.@propagate_inbounds function _getindex(::Type{Tuple{IND1,IND2}}, P::JacobiTriangle, (xy,J)::Tuple{Tuple,Any}) where {IND1<:SVector{2},IND2}
    𝐱 = convert(IND1, xy)
    @boundscheck checkbounds(P, to_indices(P, (𝐱, J))...)
    P[𝐱, J]
end

###
# Clenshaw
###

# The coefficients as the matrix of a DiagTrav. This is free when they are a DiagTrav of a padded matrix, as
# produced by the transform, and otherwise costs a copy of the first N blocks.
function _diagtravdata(P, c::AbstractVector)
    c isa DiagTrav && MemoryLayout(c.array) isa AbstractPaddedLayout && return paddeddata(c.array)
    kr = colsupport(c)
    N = isempty(kr) ? 1 : Int(findblock(axes(P,2), last(kr)))
    v = c[1:sum(1:N)]
    M = zeros(eltype(c), N, N)
    for n = 0:N-1, k = 0:n
        M[n-k+1,k+1] = v[sum(1:n)+k+1]
    end
    M
end

# Evaluate the expansion with coefficients in the matrix M of a DiagTrav: for each k we sum column k+1 against
# P_j^{(2k+b+c+1,a)}(2x-1) using Clenshaw's algorithm, giving the coefficient g_k of Q_k, and then sum these against
# Q_k = (1-x)^k * P_k^{(c,b)}(2y/(1-x)-1) using Clenshaw's algorithm again. This costs O(N^2) operations.
function _triangle_clenshaw(P::JacobiTriangle{T}, M::AbstractMatrix, xy::SVector{2}) where T
    x, y = xy
    R = polynomialtype(T, eltype(xy))
    g(k) = clenshaw(view(M, :, k+1), recurrencecoefficients(Jacobi{R}(2k+P.b+P.c+1, P.a))..., 2x-1)
    W = 1-x
    iszero(W) && return g(0)
    clenshaw(BroadcastVector(k -> W^(k-1) * g(k-1), oneto(size(M,2))), recurrencecoefficients(Jacobi{R}(P.c, P.b))..., 2y/W-1)
end

getindex(f::ApplyQuasiVector{T,typeof(*),<:Tuple{JacobiTriangle,AbstractVector}}, xy::SVector{2}) where T =
    convert(T, _triangle_clenshaw(f.args[1], _diagtravdata(f.args...), xy))::T

getindex(f::ApplyQuasiVector{T,typeof(*),<:Tuple{JacobiTriangle,AbstractVector}}, xys::AbstractArray{<:SVector{2}}) where T =
    [f[xy] for xy in xys]

####
# Grammatrices
####

"""
    legendre_triangle_grammatrix

computes the grammatrix by first re-expanding in Legendre
"""
function legendre_triangle_grammatrix(A, B)
    P = JacobiTriangle{eltype(B)}()
    (P\A)'*grammatrix(P)*(P\B)
end

@simplify *(Ac::QuasiAdjoint{<:Any,<:JacobiTriangle}, B::Weighted{<:Any,<:JacobiTriangle}) = legendre_triangle_grammatrix(parent(Ac),B)


# getindex(f::Expansion{T,<:JacobiTriangle}, x::AbstractVector{<:Number}) where T =
#     copyto!(Vector{T}(undef, length(x)), view(f, x))

# function copyto!(dest::AbstractVector{T}, v::SubArray{<:Any,1,<:Expansion{<:Any,<:JacobiTriangle}, <:Tuple{AbstractVector{<:Number}}}) where T
#     f = parent(v)
#     (x,) = parentindices(v)
#     P,c = arguments(f)
#     clenshaw!(paddeddata(c), recurrencecoefficients(P)..., x, Fill(_p0(P), length(x)), dest)
# end

lgamma(x) = logabsgamma(x)[1]

# FastTransforms uses normalized Jacobi polynomials so this corrects the normalization
function _ft_trinorm(n,k,a,b,c)
    if a == 0 && b == c == -0.5
        k == 0 ? sqrt((2n+1)) /sqrt(2π) : k*sqrt(2n+1)*exp(lgamma(k)-lgamma(k+0.5))
    else
        sqrt((2n+b+c+a+2)*(2k+c+b+1))*exp(k*log(2)+(lgamma(n+k+b+c+a+2)+lgamma(n-k+1)-log(2)*(2k+a+b+c+2)-
                lgamma(n+k+b+c+2)-lgamma(n-k+a+1) + lgamma(k+c+b+1)+lgamma(k+1)-
                    log(2)*(c+b+1)-lgamma(k+c+1)-lgamma(k+b+1))/2)
    end
end

function tridenormalize!(F̌,a,b,c)
    for n = 0:size(F̌,1)-1, k = 0:n
        F̌[n-k+1,k+1] *= _ft_trinorm(n,k,a,b,c)
    end
    F̌
end

function trinormalize!(F̌,a,b,c)
    for n = 0:size(F̌,1)-1, k = 0:n
        F̌[n-k+1,k+1] /= _ft_trinorm(n,k,a,b,c)
    end
    F̌
end

function trigrid(N::Integer)
    M = N
    x = [sinpi((2N-2n-1)/(4N))^2 for n in 0:N-1]
    w = [sinpi((2M-2m-1)/(4M))^2 for m in 0:M-1]
    [SVector(x[n+1], x[N-n]*w[m+1]) for n in 0:N-1, m in 0:M-1]
end 
grid(P::JacobiTriangle, B::Block{1}) = trigrid(Int(B))

struct TriPlan{T}
    tri2cheb::FastTransforms.FTPlan{T,2,FastTransforms.TRIANGLE}
    grid2cheb::FastTransforms.FTPlan{T,2,FastTransforms.TRIANGLEANALYSIS}
    a::T
    b::T
    c::T
end

TriPlan{T}(F::AbstractMatrix{T}, a, b, c) where T =
    TriPlan{T}(plan_tri2cheb(F, a, b, c), plan_tri_analysis(F), a, b, c)

TriPlan{T}(N::Block{1}, a, b, c) where T = TriPlan{T}(Matrix{T}(undef, Int(N), Int(N)), a, b, c)

*(T::TriPlan, F::AbstractMatrix) = DiagTrav(tridenormalize!(T.tri2cheb\(T.grid2cheb*F),T.a,T.b,T.c))

function plan_transform(P::JacobiTriangle, (N,)::Tuple{Block{1}}, dims=1)
    T = eltype(P)
    TriPlan{T}(N, P.a, P.b, P.c)
end

struct TriIPlan{T}
    tri2cheb::FastTransforms.FTPlan{T,2,FastTransforms.TRIANGLE}
    cheb2grid::FastTransforms.FTPlan{T,2,FastTransforms.TRIANGLESYNTHESIS}
    a::T
    b::T
    c::T
end

TriIPlan{T}(F::AbstractMatrix{T}, a, b, c) where T =
    TriIPlan{T}(plan_tri2cheb(F, a, b, c), plan_tri_synthesis(F), a, b, c)

TriIPlan{T}(N::Block{1}, a, b, c) where T = TriIPlan{T}(Matrix{T}(undef, Int(N), Int(N)), a, b, c)

*(T::TriIPlan, F::DiagTrav) = T.cheb2grid*(T.tri2cheb*trinormalize!(Matrix(F.array),T.a,T.b,T.c))


function plotgrid(S::JacobiTriangle{T}, B::Block{1}) where T
    N = min(2Int(B), MAX_PLOT_BLOCKS)
    grid(S, Block(N)) # double sampling
end

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{JacobiTriangle, AbstractVector}}, x...) where T
    P,c = u.args
    B = findblock(axes(P,2), last(colsupport(c)))

    N = min(2Int(B), MAX_PLOT_BLOCKS)
    F = TriIPlan{T}(Block(N), P.a, P.b, P.c)
    C = F * DiagTrav(invdiagtrav(c)[1:N,1:N]) # transform to grid
    C
end



function Base._sum(P::JacobiTriangle{T}, dims) where T
    @assert dims == 1
    @assert P.a == P.b == P.c == 0
    Hcat(one(T)/2, Zeros{T}(1,∞))
end