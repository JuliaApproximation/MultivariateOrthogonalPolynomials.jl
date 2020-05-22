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


struct JacobiTriangle{T,V} <: Basis{T}
    a::V
    b::V
    c::V
end


JacobiTriangle(a::T, b::T, c::T) where T = JacobiTriangle{float(T),T}(a, b, c)
JacobiTriangle() = JacobiTriangle(0,0,0)
==(K1::JacobiTriangle, K2::JacobiTriangle) = K1.a == K2.a && K1.b == K2.b && K1.c == K2.c

axes(P::JacobiTriangle) = (Inclusion(Triangle()),blockedrange(Base.OneTo(∞)))

struct TriangleWeight{T,V} <: Weight{T}
    a::V
    b::V
    c::V
end

TriangleWeight(a::T, b::T, c::T) where T = TriangleWeight{float(T),T}(a, b, c)

const WeightedTriangle{T} = WeightedBasis{T,<:TriangleWeight,<:JacobiTriangle}

@simplify function *(Dy::PartialDerivative{(0,1)}, P::JacobiTriangle)
    a,b,c = P.a,P.b,P.c
    k = mortar(Base.OneTo.(Base.OneTo(∞)))
    JacobiTriangle(a,b+1,c+1) * _BandedBlockBandedMatrix((k .+ (b+c))', axes(k,1), (-1,1), (-1,1))
end

@simplify function *(Dx::PartialDerivative{(1,0)}, P::JacobiTriangle)
    a,b,c = P.a,P.b,P.c
    n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
    k = mortar(Base.OneTo.(Base.OneTo(∞)))
    dat = Vcat(
        ((k .+ (b-1)) .* (n .+ k .+ (b+c-1)) ./ (2k .+ (b+c-1)))',
        ((n .+ k .+ (a+b+c)) .* (k .+ (b+c)) ./ (2k .+ (b+c-1)))'
        )
    JacobiTriangle(a+1,b,c+1) * _BandedBlockBandedMatrix(dat, axes(k,1), (-1,1), (0,1))
end

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


@simplify function \(w_A::WeightedTriangle, w_B::WeightedTriangle)
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

@simplify function \(A::JacobiTriangle, B::JacobiTriangle)
    if A.a == B.a + 1 && A.b == B.b && A.c == B.c
        Rx(B.a, B.b, B.c)
    elseif A.a == B.a && A.b == B.b + 1 && A.c == B.c
        Ry(B.a, B.b, B.c)
    # elseif A.a == B.a && A.b == B.b && A.c == B.c + 1
    #     Rz(B.a, B.b, B.c)
    else
        error("not implemented for $A and $wB")
    end
end