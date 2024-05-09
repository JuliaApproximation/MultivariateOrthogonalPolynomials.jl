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