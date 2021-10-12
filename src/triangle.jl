const UnitTriangle{T} = EuclideanUnitSimplex{2,T,:closed}

ClassicalOrthogonalPolynomials.checkpoints(d::UnitTriangle{T}) where T = [SVector{2,T}(0.1,0.2), SVector{2,T}(0.2,0.3)]

# expansion in OPs orthogonal to
# x^a*y^b*(1-x-y)^c
# defined as
# P_{n-k}^{2k+b+c+1,a}(2x-1)*(1-x)^k*P_k^{c,b}(2y/(1-x)-1)


struct JacobiTriangle{T,V} <: BivariateOrthogonalPolynomial{T}
    a::V
    b::V
    c::V
end


JacobiTriangle(a::T, b::T, c::T) where T = JacobiTriangle{float(T),T}(a, b, c)
JacobiTriangle() = JacobiTriangle(0,0,0)
==(K1::JacobiTriangle, K2::JacobiTriangle) = K1.a == K2.a && K1.b == K2.b && K1.c == K2.c

axes(P::JacobiTriangle{T}) where T = (Inclusion(UnitTriangle{T}()),blockedrange(oneto(∞)))

copy(A::JacobiTriangle) = A

Base.summary(io::IO, P::JacobiTriangle) = print(io, "JacobiTriangle($(P.a), $(P.b), $(P.c))")


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

Base.summary(io::IO, P::TriangleWeight) = print(io, "x^$(P.a)*y^$(P.b)*(1-x-y)^$(P.c) on the unit triangle")

orthogonalityweight(P::JacobiTriangle) = TriangleWeight(P.a, P.b, P.c)

@simplify function *(Dx::PartialDerivative{1}, P::JacobiTriangle)
    a,b,c = P.a,P.b,P.c
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    dat = BlockBroadcastArray(hcat,
        ((k .+ (b-1)) .* (n .+ k .+ (b+c-1)) ./ (2k .+ (b+c-1))),
        ((n .+ k .+ (a+b+c)) .* (k .+ (b+c)) ./ (2k .+ (b+c-1)))
        )
    JacobiTriangle(a+1,b,c+1) * _BandedBlockBandedMatrix(dat', axes(k,1), (-1,1), (0,1))
end

@simplify function *(Dy::PartialDerivative{2}, P::JacobiTriangle)
    a,b,c = P.a,P.b,P.c
    k = mortar(Base.OneTo.(oneto(∞)))
    T = promote_type(eltype(Dy), eltype(P)) # avoid bug in convert
    JacobiTriangle(a,b+1,c+1) * _BandedBlockBandedMatrix((k .+ convert(T, b+c))', axes(k,1), (-1,1), (-1,1))
end


# @simplify function *(Δ::Laplacian, P)
#     _lap_mul(P, eltype(axes(P,1)))
# end



@simplify function *(Ac::QuasiAdjoint{<:Any,<:JacobiTriangle}, B::JacobiTriangle)
    A = parent(Ac)
    @assert A == B == JacobiTriangle(0,0,0)
    a,b,c = A.a,A.b,A.c
    n = mortar(Fill.(oneto(∞),oneto(∞)))
    k = mortar(Base.OneTo.(oneto(∞)))
    _BandedBlockBandedMatrix((@. exp(loggamma(n+k+b+c)+loggamma(n-k+a+1)+loggamma(k+b)+loggamma(k+c)-loggamma(n+k+a+b+c)-loggamma(k+b+c)-loggamma(n-k+1)-loggamma(k))/((2n+a+b+c)*(2k+b+c-1)))',
                                axes(k,1), (0,0), (0,0))
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
    (TriangleWeight(0,0,0) .* A) \ w_B
end
function \(w_A::WeightedTriangle, B::JacobiTriangle)
    w_A.P == JacobiTriangle() && return w_A.P \ B
    w_A \ (TriangleWeight(0,0,0) .* B)
end

function \(A::JacobiTriangle, B::JacobiTriangle)
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
    ret = PseudoBlockVector{T}(undef, 1:N)
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



getindex(P::JacobiTriangle, xy::SVector{2}, JR::BlockOneTo) =
    tri_forwardrecurrence(Int(JR[end]), jacobimatrix(Val(1), P), jacobimatrix(Val(2), P), xy...)

###
# Clenshaw
###

function getindex(f::ApplyVector{T,typeof(*),<:Tuple{JacobiTriangle,AbstractVector}}, xy::SVector{2}) where T
    P,c∞ = arguments(f)
    c = paddeddata(c∞)
    N = blocksize(c,1)

    N == 1 && return c[1]
    X = jacobimatrix(Val(1), P)
    Y = jacobimatrix(Val(2), P)
    X_N = X[Block.(1:N), Block.(1:N-1)]
    Y_N = Y[Block.(1:N), Block.(1:N-1)]
    γ1 = Array{T}(undef, N-1)
    γ2 = Array{T}(undef, N)
    A = TriangleRecurrenceA(N-1, X_N, Y_N)
    B = TriangleRecurrenceB(N-1, X_N, Y_N)
    copyto!(γ2, view(c,Block(N)))
    copyto!(γ1, view(c,Block(N-1)))
    muladd!(one(T), B', γ2, one(T), γ1)
    xy_muladd!(xy, A', γ2, one(T), γ1)

    for n = N-2:-1:1
        # P[n+1,xy] == (A[n]*[x*Eye(n); y*Eye(n)] + B[n])*P[n,xy] - C[n]*P[n-1,xy]
        A,B,C = TriangleRecurrenceA(n, X_N, Y_N),TriangleRecurrenceB(n, X_N, Y_N),TriangleRecurrenceC(n+1, X_N, Y_N)
        # some magic! C can be done in-place, otherwise
        # we would need a secondary buffer.
        # this also resizes γ2
        lmul!(C', γ2)
        # Need to swap sign since it should have been -C
        γ2 .= (-).(γ2) .+ view(c,Block(n))
        γ2,γ1 = γ1,γ2
        muladd!(one(T), B', γ2, one(T), γ1)
        xy_muladd!(xy, A', γ2, one(T), γ1)
    end

    γ1[1]
end

getindex(f::ApplyVector{T,typeof(*),<:Tuple{JacobiTriangle,AbstractVector}}, xys::AbstractArray{<:SVector{2}}) where T =
    [f[xy] for xy in xys]

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

function _trigrid(N::Integer)
    M = N
    x = [sinpi((2N-2n-1)/(4N))^2 for n in 0:N-1]
    w = [sinpi((2M-2m-1)/(4M))^2 for m in 0:M-1]
    [SVector(x[n+1], x[N-n]*w[m+1]) for n in 0:N-1, m in 0:M-1]
end
trigrid(N::Block{1}) = _trigrid(Int(N))


function grid(Pn::SubQuasiArray{T,2,<:JacobiTriangle,<:Tuple{Inclusion,BlockSlice{<:BlockRange{1}}}}) where T
    kr,jr = parentindices(Pn)
    trigrid(maximum(jr.block))
end

struct TriTransform{T}
    tri2cheb::FastTransforms.FTPlan{T,2,FastTransforms.TRIANGLE}
    grid2cheb::FastTransforms.FTPlan{T,2,FastTransforms.TRIANGLEANALYSIS}
    a::T
    b::T
    c::T
end

TriTransform{T}(F::AbstractMatrix{T}, a, b, c) where T =
    TriTransform{T}(plan_tri2cheb(F, a, b, c), plan_tri_analysis(F), a, b, c)

TriTransform{T}(N::Block{1}, a, b, c) where T = TriTransform{T}(Matrix{T}(undef, Int(N), Int(N)), a, b, c)

*(T::TriTransform, F::AbstractMatrix) = DiagTrav(tridenormalize!(T.tri2cheb\(T.grid2cheb*F),T.a,T.b,T.c))

function factorize(V::SubQuasiArray{T,2,<:JacobiTriangle,<:Tuple{Inclusion,BlockSlice{BlockOneTo}}}) where T
    P = parent(V)
    _,jr = parentindices(V)
    N = maximum(jr.block)
    TransformFactorization(grid(V), TriTransform{T}(N, P.a, P.b, P.c))
end