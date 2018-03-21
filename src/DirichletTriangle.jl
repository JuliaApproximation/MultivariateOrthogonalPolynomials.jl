# this is TriangleWeight(a,b,c,KoornwinderTriangle(a,b,c)) with some extra columns to span
# all the polynomials
immutable DirichletTriangle{a,b,c} <: Space{Triangle,Float64}
    domain::Triangle
end

DirichletTriangle{a,b,c}() where {a,b,c} = DirichletTriangle{a,b,c}(Triangle())

canonicalspace(D::DirichletTriangle) = KoornwinderTriangle(0,0,0,domain(D))

spacescompatible(A::DirichletTriangle{a,b,c}, B::DirichletTriangle{a,b,c}) where {a,b,c} =
    domainscompatible(A,B)

domain(D::DirichletTriangle) = D.domain
canonicaldomain(sp::DirichletTriangle) = Triangle()
setdomain(D::DirichletTriangle{a,b,c}, d::Triangle) where {a,b,c} = DirichletTriangle{a,b,c}(d)

# TODO: @tensorspace
tensorizer(K::DirichletTriangle) = Tensorizer((ApproxFun.repeated(true),ApproxFun.repeated(true)))

# we have each polynomial
blocklengths(K::DirichletTriangle) = 1:∞

for OP in (:block,:blockstart,:blockstop)
    @eval begin
        $OP(s::DirichletTriangle, M::Block) = $OP(tensorizer(s),M)
        $OP(s::DirichletTriangle, M) = $OP(tensorizer(s),M)
    end
end


function maxspace_rule(A::DirichletTriangle, B::KoornwinderTriangle)
    domainscompatible(A,B) && return B
    NoSpace()
end

function conversion_rule(A::DirichletTriangle, B::KoornwinderTriangle)
    domainscompatible(A,B) && return A
    NoSpace()
end


Conversion(A::DirichletTriangle{1,0,0}, B::KoornwinderTriangle) = ConcreteConversion(A,B)
Conversion(A::DirichletTriangle{0,1,0} ,B::KoornwinderTriangle) = ConcreteConversion(A,B)
Conversion(A::DirichletTriangle{0,0,1}, B::KoornwinderTriangle) = ConcreteConversion(A,B)
function Conversion(A::DirichletTriangle{a,b,c}, B::DirichletTriangle{d,e,f}) where {a,b,c,d,e,f}
    @assert a ≥ d && b ≥ e && c ≥ f
    @assert domainscompatible(A,B)
    # if only one is bigger, we can do a concrete conversion
    a+b+c-d-e-f == 1 && return ConcreteConversion(A,B)
    a > d && return Conversion(A,DirichletTriangle{a-1,b,c}(domain(A)),B)
    b > e && return Conversion(A,DirichletTriangle{a,b-1,c}(domain(A)),B)
    #  c > f &&
    return Conversion(A,DirichletTriangle{a,b,c-1}(domain(A)),B)
end
function Conversion(A::DirichletTriangle{a,b,c}, B::KoornwinderTriangle) where {a,b,c}
    @assert a ≥ 0 && b ≥ 0 && c ≥ 0
    @assert domainscompatible(A,B)
    # if only one is bigger, we can do a concrete conversion
    a+b+c == 1 && return ConcreteConversion(A,B)
    a > 0 && return Conversion(A,DirichletTriangle{a-1,b,c}(domain(A)),B)
    b > 0 && return Conversion(A,DirichletTriangle{a,b-1,c}(domain(A)),B)
    #  c > 0 &&
    return Conversion(A,DirichletTriangle{a,b,c-1}(domain(A)),B)
end


isbandedblockbanded(::ConcreteConversion{<:DirichletTriangle,<:DirichletTriangle}) = true



isbandedblockbanded(::ConcreteConversion{<:DirichletTriangle,KoornwinderTriangle})= true

blockbandinds(::ConcreteConversion{DirichletTriangle{1,0,0},KoornwinderTriangle}) = (0,1)
blockbandinds(::ConcreteConversion{DirichletTriangle{0,1,0},KoornwinderTriangle}) = (0,1)
blockbandinds(::ConcreteConversion{DirichletTriangle{0,0,1},KoornwinderTriangle}) = (0,1)




subblockbandinds(::ConcreteConversion{DirichletTriangle{1,0,0},KoornwinderTriangle}) = (0,0)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,0,0},KoornwinderTriangle},k::Integer) = 0

subblockbandinds(::ConcreteConversion{DirichletTriangle{0,1,0},KoornwinderTriangle}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{0,1,0},KoornwinderTriangle},k::Integer) = k==1 ? 0 : 1


subblockbandinds(::ConcreteConversion{DirichletTriangle{0,0,1},KoornwinderTriangle}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{0,0,1},KoornwinderTriangle},k::Integer) = k==1 ? 0 : 1

function getindex(R::ConcreteConversion{DirichletTriangle{1,0,0},KoornwinderTriangle},k::Integer,j::Integer)
    T=eltype(R)
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = 2J-1

    if K==J==1
        one(T)
    elseif K==J-1 && κ == ξ
        T((J-ξ)/s)
    elseif K==J && κ == ξ == J
        one(T)
    elseif K==J && κ == ξ
        T((J-ξ)/s)
    else
        zero(T)
    end
end



function getindex{T}(R::ConcreteConversion{DirichletTriangle{0,1,0},KoornwinderTriangle,T},k::Integer,j::Integer)::T
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = (2(ξ-1))*(2J-1)

    if K==J && κ==ξ==1
        T(K)/(2K-1)
    elseif K==J-1 && κ==ξ==1
        -T(K)/(2K+1)
    elseif K==J-1 && κ == ξ-1
        T((ξ-1)*(J+ξ-2)/s)
    elseif K==J-1 && κ == ξ
        T(-(ξ-1)*(J-ξ)/s)
    elseif K==J && κ == ξ-1
        T(-(ξ-1)*(J-ξ+1)/s)
    elseif K==J && κ == ξ
        T((ξ-1)*(J+ξ-1)/s)
    else
        zero(T)
    end
end


function getindex(R::ConcreteConversion{DirichletTriangle{0,0,1},KoornwinderTriangle},k::Integer,j::Integer)
    T=eltype(R)
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = (2)*(2J-1)

    if K==J && κ==ξ==1
        T(K)/(2K-1)
    elseif K==J-1 && κ==ξ==1
        -T(K)/(2K+1)
    elseif K==J-1 && κ == ξ-1
        T((J+ξ-2)/s)
    elseif K==J-1 && κ == ξ
        T((J-ξ)/s)
    elseif K==J && κ == ξ-1
        T(-(J-ξ+1)/s)
    elseif K==J && κ == ξ
        T(-(J+ξ-1)/s)
    else
        zero(T)
    end
end


blockbandinds(::ConcreteConversion{DirichletTriangle{1,1,0},DirichletTriangle{0,1,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,1,0},DirichletTriangle{0,1,0}}) = (0,0)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,1,0},DirichletTriangle{0,1,0}},k::Integer) = 0

blockbandinds(::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{0,0,1}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{0,0,1}}) = (0,0)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{0,0,1}},k::Integer) = 0




blockbandinds(::ConcreteConversion{DirichletTriangle{1,1,0},DirichletTriangle{1,0,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,1,0},DirichletTriangle{1,0,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,1,0},DirichletTriangle{1,0,0}},k::Integer) = k==1 ? 0 : 1

blockbandinds(::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{1,0,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{1,0,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{1,0,0}},k::Integer) = k==1 ? 0 : 1


blockbandinds(::ConcreteConversion{DirichletTriangle{0,1,1},DirichletTriangle{0,0,1}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{0,1,1},DirichletTriangle{0,0,1}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{0,1,1},DirichletTriangle{0,0,1}},k::Integer) = k==1 ? 0 : 1



blockbandinds(::ConcreteConversion{DirichletTriangle{0,1,1},DirichletTriangle{0,1,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{0,1,1},DirichletTriangle{0,1,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{0,1,1},DirichletTriangle{0,1,0}},k::Integer) = k==1 ? 0 : 1



function getindex(R::ConcreteConversion{DirichletTriangle{1,1,0},DirichletTriangle{0,1,0}},k::Integer,j::Integer)
    T=eltype(R)
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = 2(J-1)

    if K == J && κ == ξ == J
        one(T)
    elseif K == J && κ == ξ == 1
        one(T)/2   # JacobiWeight(1,0,Jacobi(1,0))/2 -> Legendre
    elseif K == J-1 && κ == ξ == 1
        one(T)/2   # JacobiWeight(1,0,Jacobi(1,0))/2 -> Legendre
    elseif K==J-1 && κ == ξ && (1 < κ < J)
        T((J-ξ)/s) # Lowering{1} for interior term
    elseif K==J && κ == ξ && (1 < κ < J)
        T((J-ξ)/s) # Lowering{2} for interior term
    else
        zero(T)
    end
end


function getindex(R::ConcreteConversion{DirichletTriangle{1,1,0},DirichletTriangle{1,0,0},T},k::Integer,j::Integer)::T where T
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = 2(2J-2)


    if K == J == 1
        one(T)
    elseif K == J && κ == ξ == 1
        T((J)/(2(J-1)))  # Jacobi(1,0) -> Jacobi(1,1)
    elseif K ≠ 1 && K == J-1 && κ == ξ == 1
        -one(T)/2      # Jacobi(1,0) -> Jacobi(1,1)
    elseif K == J && κ == ξ == J
        one(T)/2 # Special lowering
    elseif K == J && ξ == J && κ == J-1
        -one(T)/2 # Special lowering
    elseif K == J-1 && ξ == J && κ == J-1
        one(T)/2 # Special lowering
    elseif K==J-1 && κ == ξ-1
        (J+ξ-3)/s # Lowering{2}
    elseif K==J-1 && κ == ξ && 1 < ξ < J-1
        -(J-ξ)/s # Lowering{2}
    elseif K==J && κ == ξ-1
        -(J-ξ)/s # Lowering{2}
    elseif K==J && κ == ξ && 1 < ξ < J
        (J+ξ-1)/s # Lowering{2}
    else
        zero(T)
    end
end


function getindex(R::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{0,0,1}},k::Integer,j::Integer)
    T=eltype(R)
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = 2(J-1)

    if K == J && κ == ξ == J
        one(T)
    elseif K == J && κ == ξ == 1
        one(T)/2   # JacobiWeight(1,0,Jacobi(1,0))/2 -> Legendre
    elseif K == J-1 && κ == ξ == 1
        one(T)/2   # JacobiWeight(1,0,Jacobi(1,0))/2 -> Legendre
    elseif K==J-1 && κ == ξ && (1 < κ < J)
        T((J-ξ)/s) # Lowering{1} for interior term
    elseif K==J && κ == ξ && (1 < κ < J)
        T((J-ξ)/s) # Lowering{1} for interior term
    else
        zero(T)
    end
end


function getindex(R::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{1,0,0},T},k::Integer,j::Integer)::T where {T}
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    # J->J-2 and ξ->ξ-1
    s = 2*(2J-2)


    if K == J == 1
        one(T)
    elseif K == J && κ == ξ == 1
        T((J)/(2(J-1)))  # Jacobi(1,0) -> Jacobi(1,1)
    elseif K ≠ 1 && K == J-1 && κ == ξ == 1
        -one(T)/2      # Jacobi(1,0) -> Jacobi(1,1)
    elseif K == J && κ == ξ == J
        -one(T)/2 # Special lowering
    elseif K == J && ξ == J && κ == J-1
        -one(T)/2 # Special lowering
    elseif K == J-1 && ξ == J && κ == J-1
        one(T)/2 # Special lowering
    elseif K==J-1 && κ == ξ-1
        (J+ξ-3)/s # Lowering{3}
    elseif K==J-1 && κ == ξ && 1 < ξ < J-1
        (J-ξ)/s # Lowering{3}
    elseif K==J && κ == ξ-1
        -(J-ξ)/s # Lowering{3}
    elseif K==J && κ == ξ && 1 < ξ < J
        -(J+ξ-1)/s # Lowering{3}
    else
        zero(T)
    end
end


function getindex(R::ConcreteConversion{DirichletTriangle{0,1,1},DirichletTriangle{0,1,0},T},k::Integer,j::Integer)::T where {T}
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    # J->J-2 and ξ->ξ-2
    # K->K-1 and k->κ-1
    s = (2ξ-3)*(2J-2)


    if K == J == 1
        one(T)
    elseif K == J && κ == ξ == 1
        -one(T)/2   # JacobiWeight(0,1,Jacobi(0,1))/2 -> Legendre
    elseif K == J-1 && κ == ξ == 1
        one(T)/2   # JacobiWeight(0,1,Jacobi(0,1))/2 -> Legendre
    elseif K == J && κ == ξ == 2
        -T(J)/(J-1)  #  (1-x-2y)*P^{(0,0,0)} -> * + y*P^{(0,1,0)}
    elseif K == J-1 && κ == ξ == 2
        T(J-2)/(J-1) #  (1-x-2y)*P^{(0,0,0)} -> * + y*P^{(0,1,0)}
    elseif K == J && ξ == 2 && κ == 1
        -one(T)/2 #  (1-x-2y)*P^{(0,0,0)} -> P^{(0,0)} + *
    elseif K == J-1 && ξ == 2 && κ == 1
        one(T)/2 #  (1-x-2y)*P^{(0,0,0)} -> P^{(0,0)} + *
    elseif K==J-1 && κ == ξ-1
        (ξ-2)*(J+ξ-3)/s # Lowering{3}
    elseif K==J-1 && κ == ξ && 2 < ξ ≤ J
        (ξ-2)*(J-ξ)/s # Lowering{3}
    elseif K==J && κ == ξ-1
        -(ξ-2)*(J-ξ+1)/s # Lowering{3}
    elseif K==J && κ == ξ && 2 < ξ ≤ J
        -(ξ-2)*(J+ξ-2)/s # Lowering{3}
    else
        zero(T)
    end
end



function getindex(R::ConcreteConversion{DirichletTriangle{0,1,1},DirichletTriangle{0,0,1},T},k::Integer,j::Integer)::T where {T}
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    # J->J-2 and ξ->ξ-2
    # K->K-1 and k->κ-1
    s = (2ξ-3)*(2J-2)


    if K == J == 1
        one(T)
    elseif K == J && κ == ξ == 1
        -one(T)/2   # JacobiWeight(0,1,Jacobi(0,1))/2 -> Legendre
    elseif K == J-1 && κ == ξ == 1
        one(T)/2   # JacobiWeight(0,1,Jacobi(0,1))/2 -> Legendre
    elseif K == J && κ == ξ == 2
        T(J)/(J-1)  #  (1-x-2y)*P^{(0,0,0)} -> * + (1-x-y)*P^{(0,0,1)}
    elseif K == J-1 && κ == ξ == 2
        -T(J-2)/(J-1) #  (1-x-2y)*P^{(0,0,0)} -> * + (1-x-y)*P^{(0,0,1)}
    elseif K == J && ξ == 2 && κ == 1
        one(T)/2 #  (1-x-2y)*P^{(0,0,0)} -> P^{(0,0)} + *
    elseif K == J-1 && ξ == 2 && κ == 1
        -one(T)/2 #  (1-x-2y)*P^{(0,0,0)} -> P^{(0,0)} + *
    elseif K==J-1 && κ == ξ-1
        (ξ-2)*(J+ξ-3)/s # Lowering{2}
    elseif K==J-1 && κ == ξ && 2 < ξ ≤ J
        -(ξ-2)*(J-ξ)/s # Lowering{2}
    elseif K==J && κ == ξ-1
        -(ξ-2)*(J-ξ+1)/s # Lowering{2}
    elseif K==J && κ == ξ && 2 < ξ ≤ J
        (ξ-2)*(J+ξ-2)/s # Lowering{2}
    else
        zero(T)
    end
end





blockbandinds(::ConcreteConversion{DirichletTriangle{1,1,1},DirichletTriangle{0,1,1}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,1,1},DirichletTriangle{0,1,1}}) = (0,0)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,1,1},DirichletTriangle{0,1,1}},k::Integer) = 0


blockbandinds(::ConcreteConversion{DirichletTriangle{1,1,1},DirichletTriangle{1,0,1}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,1,1},DirichletTriangle{1,0,1}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,1,1},DirichletTriangle{1,0,1}},k::Integer) = k==1 ? 0 : 1


blockbandinds(::ConcreteConversion{DirichletTriangle{1,1,1},DirichletTriangle{1,1,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,1,1},DirichletTriangle{1,1,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,1,1},DirichletTriangle{1,1,0}},k::Integer) = k==1 ? 0 : 1






function getindex(R::ConcreteConversion{DirichletTriangle{1,1,1},DirichletTriangle{0,1,1},T},k::Integer,j::Integer)::T where {T}
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = 2J-3

    if K == J == 1
        one(T)
    elseif K == J == 2 && κ == ξ == 1
        T(2)  # 1-2x = 1-x + *
    elseif J == 2 && K == 1 && κ == ξ == 1
        -one(T) # 1-2x
    elseif K == J == 2  && κ == ξ == 2
        one(T) #1-x-2y
    elseif K == J && κ == ξ == 1
        T(J-2)/s   # JacobiWeight(1,0,Jacobi(1,1))/2 -> Jacobi(0,1)
    elseif K == J-1 && κ == ξ == 1
        T(J-2)/s # JacobiWeight(1,0,Jacobi(1,1))/2 -> Jacobi(0,1)
    elseif K == J && κ == ξ == 2
        T(J-2)/s # x*(1-x-2y)*P_{n-2}^{(1,1)} -> (1-x-2y)*P_{n-1}^{(1,0)}
    elseif K == J-1 && κ == ξ == 2
        T(J-2)/s # x*(1-x-2y)*P_{n-2}^{(1,1)} -> (1-x-2y)*P_{n-1}^{(1,0)}
    elseif K == J && κ == ξ == J
        one(T) # y*(1-x-y)*P_{n-2,n-2}^{0,1,1} stays the same
    elseif K==J-1 && κ == ξ && (2 < κ < J)
        T(J-ξ)/s # Lowering{1} for interior term
    elseif K==J && κ == ξ && (2 < κ < J)
        T(J-ξ)/s # Lowering{1} for interior term
    else
        zero(T)
    end
end

function getindex(R::ConcreteConversion{DirichletTriangle{1,1,1},DirichletTriangle{1,0,1},T},k::Integer,j::Integer)::T where {T}
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = 2J-3

    if K == J == 1
        one(T)
    elseif K == J == 2 && κ == ξ == 1
        -T(2)  # 1-2x = x + *
    elseif J == 2 && K == 1 && κ == ξ == 1
        one(T) # 1-2x
    elseif K == J == 2  && κ == ξ == 2
        2one(T) #1-x-2y = 2(1-x-y) + *
    elseif K == J == 2  && ξ == 2 && κ == 1
        one(T) #1-x-2y = * + x-1
    elseif J == 2 && K == 1 && ξ == 2 && κ == 1
        -one(T) #1-x-2y = * + x-1
    elseif K == J && κ == ξ == 1
        -T(J-2)/s # x*(1-x)*P^{(1,1)} -> x*P^{(0,1)}
    elseif K == J-1 && κ == ξ == 1
        T(J-2)/s # x*(1-x)*P^{(1,1)} -> x*P^{(0,1)}
    elseif K == J && κ == ξ == 2
        2T(J)/s # x*(1-x-2y)*P_{n-2}^{(1,0,0)} -> 2x*(1-x-y)*P_{n-2}^{(1,0,1)} + *
    elseif K == J-1 && κ == ξ == 2 && J > 3
        -2T(J-2)/s # x*(1-x-2y)*P_{n-2}^{(1,0,0)} -> 2x*(1-x-y)*P_{n-2}^{(1,0,1)} + *
    elseif K == J && ξ == 2 && κ == 1
        T(J-2)/s # x*(1-x-2y)*P_{n-2}^{(1,0,0)} -> * + x*P^{(0,1)}
    elseif K == J-1 && ξ == 2 && κ == 1
        -T(J-2)/s # x*(1-x-2y)*P_{n-2}^{(1,0,0)} -> * + x*P^{(0,1)}
    elseif K == J && κ == ξ == J
        T(J-2)/s # y*(1-x-y)*P_{n-2,n-2}^{0,1,1} -> * + (1-x-y)*P_{n-2,n-2}^{0,0,1}
    elseif K == J-1 && ξ == J && κ == J-1
        T(J-2)/s # y*(1-x-y)*P_{n-2,n-2}^{0,1,1} -> * + (1-x-y)*P_{n-2,n-2}^{0,0,1}
    elseif K == J && ξ == J && κ == J-1
        -T(J-2)/s # y*(1-x-y)*P_{n-2,n-2}^{0,1,1} -> x*(1-x-y)*P_{n-2,n-2}^{1,0,1}
    elseif K==J && κ == ξ && (2 < ξ < J)
        T((J+ξ-2)*(ξ-2))/(s*(2ξ-3)) # Lowering{2} for interior term
    elseif K==J && κ == ξ-1 && (2 < ξ < J)
        -T((J-ξ)*(ξ-2))/(s*(2ξ-3)) # Lowering{2} for interior term
    elseif K==J-1 && κ == ξ && (2 < ξ < J-1)
        -T((J-ξ)*(ξ-2))/(s*(2ξ-3)) # Lowering{2} for interior term
    elseif K==J-1 && κ == ξ-1 && (2 < ξ < J)
        T((J+ξ-4)*(ξ-2))/(s*(2ξ-3)) # Lowering{2} for interior term
    else
        zero(T)
    end
end

function getindex(R::ConcreteConversion{DirichletTriangle{1,1,1},DirichletTriangle{1,1,0},T},k::Integer,j::Integer)::T where {T}
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = 2J-3

    if K == J == 1
        one(T)
    elseif K == J == 2 && κ == ξ == 1
        -T(2)  # 1-2x = x + *
    elseif J == 2 && K == 1 && κ == ξ == 1
        one(T) # 1-2x
    elseif K == J == 2  && κ == ξ == 2
        -2one(T) #1-x-2y = 2(1-x-y) + *
    elseif K == J == 2  && ξ == 2 && κ == 1
        -one(T) #1-x-2y = * + x-1
    elseif J == 2 && K == 1 && ξ == 2 && κ == 1
        one(T) #1-x-2y = * + x-1
    elseif K == J && κ == ξ == 1
        -T(J-2)/s # x*(1-x)*P^{(1,1)} -> x*P^{(0,1)}
    elseif K == J-1 && κ == ξ == 1
        T(J-2)/s # x*(1-x)*P^{(1,1)} -> x*P^{(0,1)}
    elseif K == J && κ == ξ == 2
        -2T(J)/s # x*(1-x-2y)*P_{n-2}^{(1,0,0)} -> 2x*(1-x-y)*P_{n-2}^{(1,0,1)} + *
    elseif K == J-1 && κ == ξ == 2 && J > 3
        2T(J-2)/s # x*(1-x-2y)*P_{n-2}^{(1,0,0)} -> 2x*(1-x-y)*P_{n-2}^{(1,0,1)} + *
    elseif K == J && ξ == 2 && κ == 1
        -T(J-2)/s # x*(1-x-2y)*P_{n-2}^{(1,0,0)} -> * + x*P^{(0,1)}
    elseif K == J-1 && ξ == 2 && κ == 1
        T(J-2)/s # x*(1-x-2y)*P_{n-2}^{(1,0,0)} -> * + x*P^{(0,1)}
    elseif K == J && κ == ξ == J
        -T(J-2)/s # y*(1-x-y)*P_{n-2,n-2}^{0,1,1} -> * + (1-x-y)*P_{n-2,n-2}^{0,0,1}
    elseif K == J-1 && ξ == J && κ == J-1
        T(J-2)/s # y*(1-x-y)*P_{n-2,n-2}^{0,1,1} -> * + (1-x-y)*P_{n-2,n-2}^{0,0,1}
    elseif K == J && ξ == J && κ == J-1
        -T(J-2)/s # y*(1-x-y)*P_{n-2,n-2}^{0,1,1} -> x*(1-x-y)*P_{n-2,n-2}^{1,0,1}
    elseif K==J && κ == ξ && (2 < ξ < J)
        -T((J+ξ-2)*(ξ-2))/(s*(2ξ-3)) # Lowering{2} for interior term
    elseif K==J && κ == ξ-1 && (2 < ξ < J)
        -T((J-ξ)*(ξ-2))/(s*(2ξ-3)) # Lowering{2} for interior term
    elseif K==J-1 && κ == ξ && (2 < ξ < J-1)
        T((J-ξ)*(ξ-2))/(s*(2ξ-3)) # Lowering{2} for interior term
    elseif K==J-1 && κ == ξ-1 && (2 < ξ < J)
        T((J+ξ-4)*(ξ-2))/(s*(2ξ-3)) # Lowering{2} for interior term
    else
        zero(T)
    end
end




## Restriction Operators

function Conversion(a::DirichletTriangle{1,0,0}, b::Jacobi)
    @assert b == Legendre(domain(a).a .. domain(a).c)
    ConcreteConversion(a,b)
end

function Conversion(a::DirichletTriangle{0,1,0}, b::Jacobi)
    @assert b == Legendre(domain(a).a .. domain(a).b)
    ConcreteConversion(a,b)
end

function Conversion(a::DirichletTriangle{0,0,1}, b::Jacobi)
    @assert b == Legendre(domain(a).c .. domain(a).b)
    ConcreteConversion(a,b)
end


function Conversion(a::DirichletTriangle{1,1,0}, b::Jacobi)
    d = domain(a)
    if b == Legendre(d.a .. d.c)
        Conversion(a, DirichletTriangle{1,0,0}(d), b)
    elseif b == Legendre(d.a .. d.b)
        Conversion(a, DirichletTriangle{0,1,0}(d), b)
    else
        throw(ArgumentError())
    end
end


function Conversion(a::DirichletTriangle{1,0,1}, b::Jacobi)
    d = domain(a)
    if b == Legendre(d.a .. d.c)
        Conversion(a, DirichletTriangle{1,0,0}(d), b)
    elseif b == Legendre(d.c .. d.b)
        Conversion(a, DirichletTriangle{0,0,1}(d), b)
    else
        throw(ArgumentError())
    end
end

function Conversion(a::DirichletTriangle{0,1,1}, b::Jacobi)
    d = domain(a)
    if b == Legendre(d.a .. d.b)
        Conversion(a, DirichletTriangle{0,1,0}(d), b)
    elseif b == Legendre(d.c .. d.b)
        Conversion(a, DirichletTriangle{0,0,1}(d), b)
    else
        throw(ArgumentError())
    end
end


function Conversion(a::DirichletTriangle{1,1,1}, b::Jacobi)
    d = domain(a)
    if b == Legendre(d.a .. d.c)
        Conversion(a, DirichletTriangle{1,0,1}(d), b)
    elseif b == Legendre(d.a .. d.b)
        Conversion(a, DirichletTriangle{0,1,1}(d), b)
    elseif b == Legendre(d.c .. d.b)
        Conversion(a, DirichletTriangle{0,1,1}(d), b)
    else
        throw(ArgumentError())
    end
end



isblockbanded(::ConcreteConversion{<:DirichletTriangle,<:Jacobi}) = true

blockbandinds(::ConcreteConversion{<:DirichletTriangle,<:Jacobi}) = (0,0)
function getindex(R::ConcreteConversion{DirichletTriangle{1,0,0},<:Jacobi},k::Integer,j::Integer)
    T=eltype(R)
    J = Int(block(domainspace(R),j))
    ξ=j-blockstart(domainspace(R),J)+1

    k==J==ξ ? one(T) : zero(T)
end

function getindex(R::ConcreteConversion{DirichletTriangle{0,1,0},<:Jacobi},k::Integer,j::Integer)
    T=eltype(R)
    J = Int(block(domainspace(R),j))
    ξ=j-blockstart(domainspace(R),J)+1

    k==J &&ξ==1 ? one(T) : zero(T)
end

function getindex(R::ConcreteConversion{DirichletTriangle{0,0,1},<:Jacobi},k::Integer,j::Integer)
    T=eltype(R)
    J = Int(block(domainspace(R),j))
    ξ=j-blockstart(domainspace(R),J)+1

    k==J &&ξ==1 ? one(T) : zero(T)
end



function Dirichlet(D::DirichletTriangle{1,1,1}, k::Int)
    @assert k==0
    d = domain(D)
    Rx=Conversion(DirichletTriangle{1,1,1}(d),DirichletTriangle{1,1,0}(d),DirichletTriangle{1,0,0}(d),Legendre(d.a .. d.c))
    Ry=Conversion(DirichletTriangle{1,1,1}(d),DirichletTriangle{1,1,0}(d),DirichletTriangle{0,1,0}(d),Legendre(d.a .. d.b))
    Rz=Conversion(DirichletTriangle{1,1,1}(d),DirichletTriangle{0,1,1}(d),DirichletTriangle{0,0,1}(d),Legendre(d.c .. d.b))

    DirichletWrapper(InterlaceOperator(Operator{Float64}[Rx;Ry;Rz],DirichletTriangle{1,1,1}(d),PiecewiseSpace((rangespace(Rx),rangespace(Ry),rangespace(Rz)))))
end


Dirichlet(d::Triangle) = Dirichlet(DirichletTriangle{1,1,1}(d))


Base.sum(f::Fun{<:DirichletTriangle}) = sum(Fun(f,KoornwinderTriangle(0,0,0,domain(f))))




#### Derivatives


function Derivative(A::DirichletTriangle{1,0,1}, order)
    d = domain(A)
    if order == [1,0]
        if d == Triangle()
            ConcreteDerivative(A,order)
        else
            S = KoornwinderTriangle(0,0,0,d)
            DerivativeWrapper(Derivative(S,order)*Conversion(A,S),order)
        end
    elseif order == [0,1]
        S = KoornwinderTriangle(0,0,0,d)
        DerivativeWrapper(Derivative(S,order)*Conversion(A,S),order)
    elseif order[1] ≥ 1
        D = Derivative(A,[1,0])
        DerivativeWrapper(TimesOperator(Derivative(rangespace(D),[order[1]-1,order[2]]),D),order)
    else
        @assert order[2] > 1
        D=Derivative(A,[0,1])
        DerivativeWrapper(TimesOperator(Derivative(rangespace(D),[order[1],order[2]-1]),D),order)
    end
end

function Derivative(A::DirichletTriangle{0,1,1}, order)
    d = domain(A)
    if order == [0,1]
        if d == Triangle()
            ConcreteDerivative(A,order)
        else
            S = KoornwinderTriangle(0,0,0,d)
            DerivativeWrapper(Derivative(S,order)*Conversion(A,S),order)
        end
    elseif order == [1,0]
        S = KoornwinderTriangle(0,0,0,d)
        DerivativeWrapper(Derivative(S,order)*Conversion(A,S),order)
    elseif order[1] > 1
        D = Derivative(A,[1,0])
        DerivativeWrapper(TimesOperator(Derivative(rangespace(D),[order[1]-1,order[2]]),D),order)
    else
        @assert order[2] ≥ 1
        D=Derivative(A,[0,1])
        DerivativeWrapper(TimesOperator(Derivative(rangespace(D),[order[1],order[2]-1]),D),order)
    end
end

function Derivative(A::DirichletTriangle{1,1,1}, order)
    d = domain(A)
    if order == [1,0] && d == Triangle()
        S = DirichletTriangle{1,0,1}()
        DerivativeWrapper(Derivative(S,order)*Conversion(A,S),order)
    elseif order == [0,1] && d == Triangle()
        S = DirichletTriangle{0,1,1}()
        DerivativeWrapper(Derivative(S,order)*Conversion(A,S),order)
    elseif order == [1,0] # d ≠ Triangle()
        M_x,M_y = tocanonicalD(d)[:,1]
        A_c = setcanonicaldomain(A)
        D_x,D_y = Derivative(A_c,[1,0]),Derivative(A_c,[0,1])
        L = M_x*D_x + M_y*D_y
        DerivativeWrapper(SpaceOperator(L,A,setdomain(rangespace(L), d)),order)
    elseif order == [0,1] # d ≠ Triangle()
        M_x,M_y = tocanonicalD(d)[:,2]
        A_c = setcanonicaldomain(A)
        D_x,D_y = Derivative(A_c,[1,0]),Derivative(A_c,[0,1])
        L = M_x*D_x + M_y*D_y
        DerivativeWrapper(SpaceOperator(L,A,setdomain(rangespace(L), d)),order)
    elseif order[1] > 1
        D = Derivative(A,[1,0])
        DerivativeWrapper(TimesOperator(Derivative(rangespace(D),[order[1]-1,order[2]]),D),order)
    else
        @assert order[2] > 1
        D=Derivative(A,[0,1])
        DerivativeWrapper(TimesOperator(Derivative(rangespace(D),[order[1],order[2]-1]),D),order)
    end
end


rangespace(D::ConcreteDerivative{DirichletTriangle{1,0,1}}) = KoornwinderTriangle(0,0,0)
rangespace(D::ConcreteDerivative{DirichletTriangle{0,1,1}}) = KoornwinderTriangle(0,0,0)

isbandedblockbanded(::ConcreteDerivative{DirichletTriangle{1,0,1}}) = true
isbandedblockbanded(::ConcreteDerivative{DirichletTriangle{0,1,1}}) = true


blockbandinds(::ConcreteDerivative{DirichletTriangle{1,0,1}}) = (-1,1)
blockbandinds(::ConcreteDerivative{DirichletTriangle{0,1,1}}) = (-1,1)

subblockbandinds(::ConcreteDerivative{DirichletTriangle{1,0,1}}) = (0,1)
subblockbandinds(::ConcreteDerivative{DirichletTriangle{1,0,1}}, k::Integer) = k == 1 ? 0 : 1

subblockbandinds(::ConcreteDerivative{DirichletTriangle{0,1,1}}) = (-1,1)
subblockbandinds(::ConcreteDerivative{DirichletTriangle{0,1,1}}, k::Integer) = k == 1 ? -1 : 1


function getindex(R::ConcreteDerivative{DirichletTriangle{1,0,1}}, k::Integer, j::Integer)
    T=eltype(R)
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    if K == J-1
        if κ == ξ == 1
            T(J-1)
        elseif ξ == J && κ == K
            T(1-J)
        elseif ξ == κ || κ == ξ-1
            T(ξ-J)/2
        else
            zero(T)
        end
    else
        zero(T)
    end
end

function getindex(R::ConcreteDerivative{DirichletTriangle{0,1,1}}, k::Integer, j::Integer)
    T=eltype(R)
    K = Int(block(rangespace(R),k))
    J = Int(block(domainspace(R),j))
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    if K == J-1
        if κ == 1 && ξ == 2
            -2*one(T)
        elseif κ == ξ-1
            (2one(T)-ξ)
        else
            zero(T)
        end
    else
        zero(T)
    end
end
