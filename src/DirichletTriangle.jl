# this is TriangleWeight(a,b,c,KoornwinderTriangle(a,b,c)) with some extra columns to span
# all the polynomials
immutable DirichletTriangle{a,b,c} <: Space{RealBasis,Triangle,2}  end

spacescompatible{a,b,c}(::DirichletTriangle{a,b,c},::DirichletTriangle{a,b,c}) = true

domain(::DirichletTriangle) = Triangle()

# TODO: @tensorspace
tensorizer(K::DirichletTriangle) = Tensorizer((ApproxFun.repeated(true),ApproxFun.repeated(true)))

# we have each polynomial
blocklengths(K::DirichletTriangle) = 1:∞

for OP in (:block,:blockstart,:blockstop)
    @eval begin
        $OP(s::DirichletTriangle,M::Block) = $OP(tensorizer(s),M)
        $OP(s::DirichletTriangle,M) = $OP(tensorizer(s),M)
    end
end


maxspace_rule(A::DirichletTriangle,B::KoornwinderTriangle) = B
conversion_rule(A::DirichletTriangle,B::KoornwinderTriangle) = A


Conversion(A::DirichletTriangle,B::KoornwinderTriangle) = ConcreteConversion(A,B)
Conversion(A::DirichletTriangle,B::DirichletTriangle) = ConcreteConversion(A,B)

isbandedblockbanded{a,b,c}(::ConcreteConversion{DirichletTriangle{a,b,c},KoornwinderTriangle}) = true

blockbandinds(::ConcreteConversion{DirichletTriangle{1,0,0},KoornwinderTriangle}) = (0,1)
blockbandinds(::ConcreteConversion{DirichletTriangle{0,1,0},KoornwinderTriangle}) = (0,1)
blockbandinds(::ConcreteConversion{DirichletTriangle{0,0,1},KoornwinderTriangle}) = (0,1)




subblockbandinds(::ConcreteConversion{DirichletTriangle{1,0,0},KoornwinderTriangle}) = (0,0)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,0,0},KoornwinderTriangle},k::Integer) = 0

subblockbandinds(::ConcreteConversion{DirichletTriangle{0,1,0},KoornwinderTriangle}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{0,1,0},KoornwinderTriangle},k::Integer) = k==1? 0 : 1


subblockbandinds(::ConcreteConversion{DirichletTriangle{0,0,1},KoornwinderTriangle}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{0,0,1},KoornwinderTriangle},k::Integer) = k==1? 0 : 1

function getindex(R::ConcreteConversion{DirichletTriangle{1,0,0},KoornwinderTriangle},k::Integer,j::Integer)
    T=eltype(R)
    K=block(rangespace(R),k).K
    J=block(domainspace(R),j).K
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
    K=block(rangespace(R),k).K
    J=block(domainspace(R),j).K
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = (2(ξ-1))*(2J-1)

    if K==J && κ==ξ==1
        T(K)/(2K-1)
    elseif K==J-1 && ξ==1
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
    K=block(rangespace(R),k).K
    J=block(domainspace(R),j).K
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = (2(ξ-1))*(2J-1)

    if K==J && κ==ξ==1
        T(K)/(2K-1)
    elseif K==J-1 && ξ==1
        -T(K)/(2K+1)
    elseif K==J-1 && κ == ξ-1
        T((ξ-1)*(J+ξ-2)/s)
    elseif K==J-1 && κ == ξ
        T((ξ-1)*(J-ξ)/s)
    elseif K==J && κ == ξ-1
        T(-(ξ-1)*(J-ξ+1)/s)
    elseif K==J && κ == ξ
        T(-(ξ-1)*(J+ξ-1)/s)
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
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,1,0},DirichletTriangle{1,0,0}},k::Integer) = k==1? 0 :1

blockbandinds(::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{1,0,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{1,0,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{1,0,0}},k::Integer) = k==1? 0 :1



blockbandinds(::ConcreteConversion{DirichletTriangle{0,1,1},DirichletTriangle{0,1,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{0,1,1},DirichletTriangle{0,1,0}}) = (0,1)
subblockbandinds(::ConcreteConversion{DirichletTriangle{0,1,1},DirichletTriangle{0,1,0}},k::Integer) = k==1? 0 :1



function getindex(R::ConcreteConversion{DirichletTriangle{1,1,0},DirichletTriangle{0,1,0}},k::Integer,j::Integer)
    T=eltype(R)
    K=block(rangespace(R),k).K
    J=block(domainspace(R),j).K
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


function getindex{T}(R::ConcreteConversion{DirichletTriangle{1,1,0},DirichletTriangle{1,0,0},T},k::Integer,j::Integer)::T
    K=block(rangespace(R),k).K
    J=block(domainspace(R),j).K
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = (2(ξ-1))*(2J-2)


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
        (ξ-1)*(J+ξ-3)/s # Lowering{2}
    elseif K==J-1 && κ == ξ && 1 < ξ < J-1
        -(ξ-1)*(J-ξ)/s # Lowering{2}
    elseif K==J && κ == ξ-1
        -(ξ-1)*(J-ξ)/s # Lowering{2}
    elseif K==J && κ == ξ && 1 < ξ < J
        (ξ-1)*(J+ξ-1)/s # Lowering{2}
    else
        zero(T)
    end
end


function getindex(R::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{0,0,1}},k::Integer,j::Integer)
    T=eltype(R)
    K=block(rangespace(R),k).K
    J=block(domainspace(R),j).K
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


function getindex{T}(R::ConcreteConversion{DirichletTriangle{1,0,1},DirichletTriangle{1,0,0},T},k::Integer,j::Integer)::T
    K=block(rangespace(R),k).K
    J=block(domainspace(R),j).K
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    # J->J-2 and ξ->ξ-1
    s = (2(ξ-1))*(2J-2)


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
        (ξ-1)*(J+ξ-3)/s # Lowering{3}
    elseif K==J-1 && κ == ξ && 1 < ξ < J-1
        (ξ-1)*(J-ξ)/s # Lowering{3}
    elseif K==J && κ == ξ-1
        -(ξ-1)*(J-ξ)/s # Lowering{3}
    elseif K==J && κ == ξ && 1 < ξ < J
        -(ξ-1)*(J+ξ-1)/s # Lowering{3}
    else
        zero(T)
    end
end
