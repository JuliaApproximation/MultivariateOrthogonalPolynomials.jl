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

getindex(P::JacobiTriangle, xy::SVector{2}, J::Block{1}) = tri_forwardrecurrence(Int(J), jacobimatrix(Val(1), P), jacobimatrix(Val(2), P), xy...)[J]
getindex(P::JacobiTriangle, xy::SVector{2}, JR::BlockRange{1}) = tri_forwardrecurrence(Int(JR[end]), jacobimatrix(Val(1), P), jacobimatrix(Val(2), P), xy...)[JR]
getindex(P::JacobiTriangle, xy::SVector{2}, Jj::BlockIndex{1}) = P[xy, block(Jj)][blockindex(Jj)]
getindex(P::JacobiTriangle, xy::SVector{2}, j::Integer) = P[xy, findblockindex(axes(P,2), j)]