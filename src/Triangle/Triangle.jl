## Triangle Def
# currently right trianglel
struct Triangle <: EuclideanDomain{2,Float64}
    a::SVector{2,Float64}
    b::SVector{2,Float64}
    c::SVector{2,Float64}
end

Triangle() = Triangle(SVector(0,0), SVector(1,0), SVector(0,1))
function in(p::SVector{2,Float64}, d::Triangle)
    x,y = p
    0 ≤ x ≤ x + y ≤ 1
end

# issubset(a::Segment{<:SVector{2}}, b::Triangle) = all(in.(endpoints(a), Ref(b)))


#canonical is rectangle [0,1]^2
# with the map (x,y)=(s,(1-s)*t)
iduffy(st::SVector) = SVector(st[1],(1-st[1])*st[2])
iduffy(s,t) = SVector(s,(1-s)*t)
duffy(xy::SVector) = SVector(xy[1],xy[1]==1 ? zero(eltype(xy)) : xy[2]/(1-xy[1]))
duffy(x::T,y::T) where T = SVector(x,x == 1 ? zero(Y) : y/(1-x))

boundary(d::Triangle) = PiecewiseSegment([d.a,d.b,d.c,d.a])


Base.isnan(::Triangle) = false

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

axes(P::JacobiTriangle) = (Inclusion(Triangle()),blockedrange(Base.OneTo(∞)))

copy(A::JacobiTriangle) = A

struct TriangleWeight{T,V} <: Weight{T}
    a::V
    b::V
    c::V
end

TriangleWeight(a::T, b::T, c::T) where T = TriangleWeight{float(T),T}(a, b, c)

const WeightedTriangle{T} = WeightedBasis{T,<:TriangleWeight,<:JacobiTriangle}

WeightedTriangle(a, b, c) = TriangleWeight(a,b,c) .* JacobiTriangle(a,b,c)

axes(P::TriangleWeight) = (Inclusion(Triangle()),)

@simplify function *(Dx::PartialDerivative{1}, P::JacobiTriangle)
    a,b,c = P.a,P.b,P.c
    n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
    k = mortar(Base.OneTo.(Base.OneTo(∞)))
    dat = Vcat(
        ((k .+ (b-1)) .* (n .+ k .+ (b+c-1)) ./ (2k .+ (b+c-1)))',
        ((n .+ k .+ (a+b+c)) .* (k .+ (b+c)) ./ (2k .+ (b+c-1)))'
        )
    JacobiTriangle(a+1,b,c+1) * _BandedBlockBandedMatrix(dat, axes(k,1), (-1,1), (0,1))
end

@simplify function *(Dy::PartialDerivative{2}, P::JacobiTriangle)
    a,b,c = P.a,P.b,P.c
    k = mortar(Base.OneTo.(Base.OneTo(∞)))
    JacobiTriangle(a,b+1,c+1) * _BandedBlockBandedMatrix((k .+ (b+c))', axes(k,1), (-1,1), (-1,1))
end


# @simplify function *(Δ::Laplacian, P)
#     _lap_mul(P, eltype(axes(P,1)))
# end



@simplify function *(Ac::QuasiAdjoint{<:Any,<:JacobiTriangle}, B::JacobiTriangle)
    A = parent(Ac)
    @assert A == B == JacobiTriangle(0,0,0)
    a,b,c = A.a,A.b,A.c
    n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
    k = mortar(Base.OneTo.(Base.OneTo(∞)))    
    _BandedBlockBandedMatrix((@. exp(loggamma(n+k+b+c)+loggamma(n-k+a+1)+loggamma(k+b)+loggamma(k+c)-loggamma(n+k+a+b+c)-loggamma(k+b+c)-loggamma(n-k+1)-loggamma(k))/((2n+a+b+c)*(2k+b+c-1)))',
                                axes(k,1), (0,0), (0,0))
end

 function Wy(a,b,c)
    k = mortar(Base.OneTo.(Base.OneTo(∞)))
    _BandedBlockBandedMatrix((-k)', axes(k,1), (1,-1), (1,-1))
end
function Wx(a,b,c)
    n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
    k = mortar(Base.OneTo.(Base.OneTo(∞)))
    dat = Vcat(
        ((k .+ (c-1)) .* ( k .- n .- 1 ) ./ (2k .+ (b+c-1)))',
        (k .* (k .- n .- a) ./ (2k .+ (b+c-1)))'
        )
    _BandedBlockBandedMatrix(dat, axes(k,1), (1,-1), (1,0))
end

 function Rx(a,b,c)
    n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
    k = mortar(Base.OneTo.(Base.OneTo(∞)))
    dat = PseudoBlockArray(Vcat(        
        ((n .+ k .+ (b+c-1) ) ./ (2n .+ (a+b+c)))',
        ((n .+ k .+ (a+b+c) ) ./ (2n .+ (a+b+c)))'
        ), (blockedrange(Fill(1,2)), axes(n,1)))
    _BandedBlockBandedMatrix(dat, axes(k,1), (0,1), (0,0))
end
function Ry(a,b,c)
    n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
    k = mortar(Base.OneTo.(Base.OneTo(∞)))
    dat = PseudoBlockArray(Vcat(        
        ((k .+ (c-1) ) .* (n .+ k .+ (b+c-1)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))',
        ((k .- n .- a ) .* (k .+ (b+c)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))',
        ((k .+ (c-1) ) .* (k .- n .- 1) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))',
        ((n .+ k .+ (a+b+c) ) .* (k .+ (b+c)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))'
        ), (blockedrange(Fill(2,2)), axes(n,1)))
    _BandedBlockBandedMatrix(dat, axes(k,1), (0,1), (0,1))
end

function Lx(a,b,c)
    n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
    k = mortar(Base.OneTo.(Base.OneTo(∞)))
    dat = PseudoBlockArray(Vcat(        
        ((n .- k .+ a) ./ (2n .+ (a+b+c)))',
        ((n .- k .+ 1) ./ (2n .+ (a+b+c)))'
        ), (blockedrange(Fill(1,2)), axes(n,1)))
    _BandedBlockBandedMatrix(dat, axes(k,1), (1,0), (0,0))
end
function Ly(a,b,c)
    n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
    k = mortar(Base.OneTo.(Base.OneTo(∞)))
    dat = PseudoBlockArray(Vcat(        
        ((k .+ (b-1) ) .* (n .+ k .+ (b+c-1)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))',
        (k .* (k .- n .- a ) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))',
        ((k .+ (b-1) ) .* (k .- n .- 1) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))',
        ( k .* (n .+ k .+ (a+b+c) ) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))'
        ), (blockedrange(Fill(2,2)), axes(n,1)))
    _BandedBlockBandedMatrix(dat, axes(k,1), (1,0), (1,0))
end
function Lz(a,b,c)
    n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
    k = mortar(Base.OneTo.(Base.OneTo(∞)))
    dat = PseudoBlockArray(Vcat(        
        ((k .+ (c-1) ) .* (n .+ k .+ (b+c-1)) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))',
        (k .* (n .- k .+ a ) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))',
        ((k .+ (c-1) ) .* (k .- n .- 1) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))',
        ( (-k) .* (n .+ k .+ (a+b+c) ) ./ ((2n .+ (a+b+c)) .* (2k .+ (b+c-1) )))'
        ), (blockedrange(Fill(2,2)), axes(n,1)))
    _BandedBlockBandedMatrix(dat, axes(k,1), (1,0), (1,0))
end


function \(w_A::WeightedTriangle, w_B::WeightedTriangle)
    wA,A = w_A.args
    wB,B = w_B.args

    @assert wA.a == A.a && wA.b == A.b && wA.c == A.c
    @assert wB.a == B.a && wB.b == B.b && wB.c == B.c

    if A.a + 1 == B.a && A.b == B.b && A.c == B.c
        Lx(B.a, B.b, B.c)
    elseif A.a == B.a && A.b + 1 == B.b && A.c == B.c
        Ly(B.a, B.b, B.c)
    elseif A.a == B.a && A.b == B.b && A.c + 1 == B.c
        Lz(B.a, B.b, B.c)
    else
        error("not implemented for $A and $wB")
    end
end

\(w_A::JacobiTriangle, w_B::WeightedTriangle) = 
    (TriangleWeight(0,0,0) .* w_A) \ w_B

function \(A::JacobiTriangle, B::JacobiTriangle)
    if A.a == B.a && A.b == B.b && A.c == B.c
        Eye((axes(B,2),))
    elseif A.a == B.a + 1 && A.b == B.b && A.c == B.c
        Rx(B.a, B.b, B.c)
    elseif A.a == B.a && A.b == B.b + 1 && A.c == B.c
        Ry(B.a, B.b, B.c)
    # elseif A.a == B.a && A.b == B.b && A.c == B.c + 1
    #     Rz(B.a, B.b, B.c)
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


function LinearAlgebra.mul!(dest::AbstractVector{T}, BA::TriangleRecurrenceB{T}, c::AbstractVector{T}) where T
    @boundscheck (size(BA,2) == length(c) && size(BA,1) == length(dest)) || throw(DimensionMismatch())
    aˣ,bˣ = BA.aˣ,BA.bˣ
    Ba_1, Ba_2 = BA.d

    n = length(bˣ)
    dest[1:n] .= (-).(bˣ .\ aˣ .* c)
    dest[n+1] = Ba_1 * c[end-1] + Ba_2 * c[end]
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
    dest[n+1] = d * u[end] + β*dest[n+1]
    dest
end


# Following gives data for applying 
# 
# and 
# B⁺ * [Cˣ; Cʸ] == [Diagonal(Bˣ[band(0)] .\ (x .- Aˣ[band(0)])); [zeros(1,N-2) Ba_1 Ba_2]]

function tri_forwardrecurrence(N::Int, X, Y, x, y)
    T = promote_type(eltype(X),eltype(Y))
    ret = PseudoBlockVector{T}(undef, 1:N)
    N < 1 && return ret
    ret[1] = 1
    N < 2 && return ret
    ret[2] = x/X[1,1]-1
    ret[3] = -Y[2,1]/(Y[3,1]*X[1,1]) * (x-X[1,1]) + (y-Y[1,1])/Y[3,1]

    X_N = X[Block.(1:N), Block.(1:N-1)]
    Y_N = Y[Block.(1:N), Block.(1:N-1)]

    for n = 2:N-1
        a_x = view(X_N,Block(n,n))[band(0)]
        Aʸ = view(Y_N,Block(n,n))'
        b_x = view(X_N,Block(n+1,n))[band(0)]
        Bʸ = view(Y_N,Block(n+1,n))'
        c_x = view(X_N,Block(n-1,n))[band(0)]
        Cʸ = view(Y_N,Block(n-1,n))'
        b₂ = Bʸ[n,n+1]
        Bc = -Bʸ[n,n-1]/(b₂*b_x[n-1])*c_x[n-1] + Cʸ[n,n-1]/b₂
        Ba_1 = -Bʸ[n,n-1]/(b₂*b_x[n-1]) * (x - a_x[n-1]) + (-Aʸ[n,n-1])/b₂
        Ba_2 = -Bʸ[n,n]/(b₂*b_x[n]) * (x - a_x[n]) + (y-Aʸ[n,n])/b₂
        P_1 = view(ret,Block(n-1))
        P_2 = view(ret,Block(n))
        w = view(ret,Block(n+1))
        w[1:n] .= b_x .\ (x .- a_x) .* P_2
        w[n+1] = Ba_1 * P_2[end-1] + Ba_2 * P_2[end]
        w[1:n-1] .-= (view(b_x,1:n-1) .\ c_x .* P_1)
        w[n+1] -= Bc * P_1[end]
    end
    ret
end

getindex(P::JacobiTriangle, xy::SVector{2}, JR::BlockOneTo) = 
    tri_forwardrecurrence(Int(JR[end]), jacobimatrix(Val(1), P), jacobimatrix(Val(2), P), xy...)


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
    tri2cheb::FastTransforms.FTPlan{T,2,14}
    grid2cheb::FastTransforms.FTPlan{T,2,24}
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