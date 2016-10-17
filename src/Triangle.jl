export Triangle,KoornwinderTriangle


## Triangle Def
# currently right trianglel
immutable Triangle <: BivariateDomain{Float64} end


#canonical is rectangle [0,1]^2
# with the map (x,y)=(s,(1-s)*t)
fromcanonical(::Triangle,s,t)=s,(1-s)*t
tocanonical(::Triangle,x,y)=x,y/(1-x)
checkpoints(d::Triangle)=[fromcanonical(d,(.1,.2243));fromcanonical(d,(-.212423,-.3))]

∂(d::Triangle) = PiecewiseInterval([(0.,0.),(1.,0.),(0.,1.),(0.,0.)])

# expansion in OPs orthogonal to
# x^α*y^β*(1-x-y)^γ
# defined as
# P_{n-k}^{2k+β+γ+1,α}(2x-1)*(1-x)^k*P_k^{γ,β}(2y/(1-x)-1)


immutable ProductTriangle <: AbstractProductSpace{Tuple{WeightedJacobi{Float64,Interval{Float64}},
                                                        Jacobi{Float64,Interval{Float64}}},Float64,2}
    α::Float64
    β::Float64
    γ::Float64
    domain::Triangle
end

immutable KoornwinderTriangle <: Space{RealBasis,Triangle,2}
    α::Float64
    β::Float64
    γ::Float64
    domain::Triangle
end

typealias TriangleSpace Union{ProductTriangle,KoornwinderTriangle}

ProductTriangle(K::KoornwinderTriangle) = ProductTriangle(K.α,K.β,K.γ,K.domain)


for TYP in (:ProductTriangle,:KoornwinderTriangle)
    @eval begin
        $TYP(α,β,γ) = $TYP(α,β,γ,Triangle())
        $TYP(T::Triangle) = $TYP(0.,0.,0.,T)
        spacescompatible(K1::$TYP,K2::$TYP) =
            K1.α==K2.α && K1.β==K2.β && K1.γ==K2.γ
    end
end



Space(T::Triangle) = KoornwinderTriangle(T)


# TODO: @tensorspace
tensorizer(K::TriangleSpace) = TensorIterator((∞,∞))

# we have each polynomial
blocklengths(K::TriangleSpace) = 1:∞

for OP in (:block,:blockstart,:blockstop)
    @eval $OP(s::TriangleSpace,M) = $OP(tensorizer(s),M)
end


# support for ProductFun constructor

function space(T::ProductTriangle,k::Integer)
    @assert k==2
    Jacobi(T.γ,T.β,Interval(0.,1.))
end

columnspace(T::ProductTriangle,k::Integer) =
    JacobiWeight(0.,k-1.,Jacobi(2k-1+T.β+T.γ,T.α,Interval(0.,1.)))


# convert coefficients

Fun(f::Function,S::KoornwinderTriangle) =
    Fun(Fun(ProductFun(f,ProductTriangle(S))),S)

function coefficients(f::AbstractVector,K::KoornwinderTriangle,P::ProductTriangle)
    C=totensor(K,f)
    D=Float64[2.0^(-k) for k=0:size(C,1)-1]
    fromtensor(K,(C.')*diagm(D))
end

function coefficients(f::AbstractVector,K::ProductTriangle,P::KoornwinderTriangle)
    C=totensor(K,f)
    D=Float64[2.0^(k) for k=0:size(C,1)-1]
    fromtensor(P,(C*diagm(D)).')
end


import ApproxFun:viewblock
function clenshaw2D(Jx,Jy,cfs,x,y)
    N=length(cfs)
    bk1=zeros(N+1)
    bk2=zeros(N+2)

    A=[sparse(viewblock(Jx,N+2,N+1)) sparse(viewblock(Jy,N+2,N+1))]

    Abk1x=zeros(N+1)
    Abk1y=zeros(N+1)


    for K=N:-1:1
        A,Ap=[sparse(viewblock(Jx,K+1,K)) sparse(viewblock(Jy,K+1,K))],A
        Bx,By=viewblock(Jx,K,K),viewblock(Jy,K,K)
        Cx,Cy=viewblock(Jx,K,K+1),viewblock(Jy,K,K+1)

        Abk1=A\bk1

        Abk1x,Abk2x=Abk1[1:K],Abk1x
        Abk1y,Abk2y=Abk1[K+1:end],Abk1y

        bk1,bk2=cfs[K] + x*Abk1x + y*Abk1y - Bx*Abk1x - By*Abk1y -
            Cx*Abk2x - Cy*Abk2y,bk1
    end
    bk1[1]
end

# convert to vector of coefficients
# TODO: replace with RaggedMatrix
function totree(S,f::Vector)
    N=block(S,length(f))
    ret = Array(Vector{eltype(f)},N)
    for K=1:N-1
        ret[K]=f[blockrange(S,K)]
    end
    ret[N]=pad(f[blockstart(S,N):end],N)
    ret
end

function clenshaw(f::AbstractVector,S::KoornwinderTriangle,x,y)
    N = block(S,length(f))
    n = blockstop(S,N+2)
    m = blockstop(S,N+1)
    clenshaw2D((Recurrence(1,S))[1:n,1:m],
                (Recurrence(2,S)↦S)[1:n,1:m],
                    totree(S,f),x,y)
end

clenshaw(f::Fun{KoornwinderTriangle},x,y) = clenshaw(f.coefficients,f.space,x,y)


# evaluate(f::AbstractVector,K::KoornwinderTriangle,x...) =
#     evaluate(coefficients(f,K,ProductTriangle(K)),ProductTriangle(K),x...)

evaluate(f::AbstractVector,K::KoornwinderTriangle,x,y) =
    clenshaw(f,K,x,y)

evaluate(f::AbstractVector,K::KoornwinderTriangle,x) =
    clenshaw(f,K,x...)

# Operators



function Derivative(K::KoornwinderTriangle,order::Vector{Int})
    @assert length(order)==2
    if order==[1,0] || order==[0,1]
        ConcreteDerivative(K,order)
    elseif order[1]>1
        D=Derivative(K,[1,0])
        DerivativeWrapper(TimesOperator(Derivative(rangespace(D),[order[1]-1,order[2]]),D),order)
    else
        @assert order[2]≥1
        D=Derivative(K,[0,1])
        DerivativeWrapper(TimesOperator(Derivative(rangespace(D),[order[1],order[2]-1]),D),order)
    end
end


rangespace(D::ConcreteDerivative{KoornwinderTriangle}) =
    KoornwinderTriangle(D.space.α+D.order[1],
                        D.space.β+D.order[2],
                        D.space.γ+sum(D.order),
                        domain(D))
bandinds(D::ConcreteDerivative{KoornwinderTriangle}) = 0,sum(D.order)
blockbandinds(D::ConcreteDerivative{KoornwinderTriangle},k::Integer) = k==0 ? 0 : sum(D.order)

function getindex(D::ConcreteDerivative{KoornwinderTriangle},n::Integer,j::Integer)
    K=domainspace(D)
    α,β,γ = K.α,K.β,K.γ
    if D.order==[0,1]
        if j==n+1
            ret=bzeros(n,j,0,1)

            for k=1:size(ret,1)
                ret[k,k+1]+=(1+k+β+γ)
            end
        else
            ret=bzeros(n,j,0,0)
        end
    elseif D.order==[1,0]
        if j==n+1
            ret=bzeros(n,j,0,1)

            for k=1:size(ret,1)
                ret[k,k]+=(1+k+n+α+β+γ)*(k+γ+β)/(2k+γ+β-1)
                ret[k,k+1]+=(k+β)*(k+n+γ+β+1)/(2k+γ+β+1)
            end
        else
            ret=bzeros(n,j,0,0)
        end
    else
        error("Not implemented")
    end
    ret
end



## Conversion

conversion_rule(K1::KoornwinderTriangle,K2::KoornwinderTriangle) =
    KoornwinderTriangle(min(K1.α,K2.α),min(K1.β,K2.β),min(K1.γ,K2.γ))
maxspace_rule(K1::KoornwinderTriangle,K2::KoornwinderTriangle) =
    KoornwinderTriangle(max(K1.α,K2.α),max(K1.β,K2.β),max(K1.γ,K2.γ))

function Conversion(K1::KoornwinderTriangle,K2::KoornwinderTriangle)
    @assert K1.α≤K2.α && K1.β≤K2.β && K1.γ≤K2.γ &&
        isapproxinteger(K1.α-K2.α) && isapproxinteger(K1.β-K2.β) &&
        isapproxinteger(K1.γ-K2.γ)

    if (K1.α+1==K2.α && K1.β==K2.β && K1.γ==K2.γ) ||
            (K1.α==K2.α && K1.β+1==K2.β && K1.γ==K2.γ) ||
            (K1.α==K2.α && K1.β==K2.β && K1.γ+1==K2.γ)
        ConcreteConversion(K1,K2)
    elseif K1.α+1<K2.α || (K1.α+1==K2.α && (K1.β+1≥K2.β || K1.γ+1≥K2.γ))
        # increment α if we have e.g. (α+2,β,γ) or  (α+1,β+1,γ)
        Conversion(K1,KoornwinderTriangle(K1.α+1,K1.β,K1.γ),K2)
    elseif K1.β+1<K2.β || (K1.β+1==K2.β && K1.γ+1≥K2.γ)
        # increment β
        Conversion(K1,KoornwinderTriangle(K1.α,K1.β+1,K1.γ),K2)
    elseif K1.γ+1<K2.γ
        # increment γ
        Conversion(K1,KoornwinderTriangle(K1.α,K1.β,K1.γ+1),K2)
    else
        error("There is a bug")
    end
end


isbandedblockbanded(::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle}) = true


blockbandinds(C::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle}) = (0,1)

subblockbandinds(C::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle}) = (0,1)
subblockbandinds(C::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle},k::Integer) = k==1?0:1


domaintensorizer(R::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle}) = tensorizer(domainspace(R))
rangetensorizer(R::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle}) = tensorizer(rangespace(R))



function getindex{T}(C::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle,T},k::Integer,j::Integer)
    K1=domainspace(C);K2=rangespace(C)
    α,β,γ = K1.α,K1.β,K1.γ
    K=block(K2,k)
    J=block(K1,j)
    κ=k-blockstart(K2,K)+1
    ξ=j-blockstart(K1,J)+1

    if K2.α == α+1 && K2.β == β && K2.γ == γ
        if     K == J    && κ == ξ
            T((K+κ+α+β+γ)/(2K+α+β+γ))
        elseif J == K+1  && κ == ξ
            T((K+κ+β+γ)/(2K+α+β+γ+2))
        else
            zero(T)
        end
    elseif K2.α==α && K2.β==β+1 && K2.γ==γ
        if     J == K   && κ == ξ
            T((K+κ+α+β+γ)/(2K+α+β+γ)*(κ+β+γ)/(2κ+β+γ-1))
        elseif J == K   && κ+1 == ξ
            T(-(κ+γ)/(2κ+β+γ+1)*(K-κ)/(2K+α+β+γ))
        elseif J == K+1 && κ == ξ
            T(-(K-κ+α+1)/(2K+α+β+γ+2)*(κ+β+γ)/(2κ+β+γ-1))
        elseif J == K+1 && κ+1 == ξ
            T((κ+γ)/(2κ+β+γ+1)*(K+κ+β+γ+1)/(2K+α+β+γ+2))
        else
            zero(T)
        end
    elseif K2.α==α && K2.β==β && K2.γ==γ+1
        if K == J && κ == ξ
            T((K+κ+α+β+γ)/(2K+α+β+γ)*(κ+β+γ)/(2κ+β+γ-1))
        elseif K == J && κ+1 == ξ
            T((κ+β)/(2κ+β+γ+1)*(K-κ)/(2K+α+β+γ))
        elseif J == K+1 && κ == ξ
            T(-(K-κ+α+1)/(2K+α+β+γ+2)*(κ+β+γ)/(2κ+β+γ-1))
        elseif J == K+1 && κ+1 == ξ
            T(-(κ+β)/(2κ+β+γ+1)*(K+κ+β+γ+1)/(2K+α+β+γ+2))
        else
            zero(T)
        end
    else
        error("Not implemented")
    end
end



## Jacobi Operators

# x is 1, 2, ... for x, y, z,...
immutable Recurrence{x,S,T} <: Operator{T}
    space::S
end

Recurrence(k::Integer,sp) = Recurrence{k,typeof(sp),promote_type(eltype(sp),eltype(domain(sp)))}(sp)
Base.convert{x,T,S}(::Type{Operator{T}},J::Recurrence{x,S}) = Recurrence{x,S,T}(J.space)


domainspace(R::Recurrence) = R.space

isbandedblockbanded(::Recurrence{1,KoornwinderTriangle}) = true
isbandedblockbanded(::Recurrence{2,KoornwinderTriangle}) = true

blockbandinds(::Recurrence{1,KoornwinderTriangle}) = (-1,1)
blockbandinds(::Recurrence{2,KoornwinderTriangle}) = (-1,0)

subblockbandinds(::Recurrence{1,KoornwinderTriangle}) = (0,0)
subblockbandinds(::Recurrence{2,KoornwinderTriangle}) = (-1,0)

subblockbandinds(::Recurrence{1,KoornwinderTriangle},k::Integer) = 0
subblockbandinds(::Recurrence{2,KoornwinderTriangle},k::Integer) = k==1? -1 : 0


domaintensorizer(R::Recurrence{1,KoornwinderTriangle}) = tensorizer(domainspace(R))
rangetensorizer(R::Recurrence{1,KoornwinderTriangle}) = tensorizer(rangespace(R))

domaintensorizer(R::Recurrence{2,KoornwinderTriangle}) = tensorizer(domainspace(R))
rangetensorizer(R::Recurrence{2,KoornwinderTriangle}) = tensorizer(rangespace(R))


rangespace(R::Recurrence{1,KoornwinderTriangle}) = R.space
rangespace(R::Recurrence{2,KoornwinderTriangle}) =
    KoornwinderTriangle(R.space.α,R.space.β-1,R.space.γ)

function getindex{T}(R::Recurrence{1,KoornwinderTriangle,T},k::Integer,j::Integer)
    α,β,γ=R.space.α,R.space.β,R.space.γ
    K=block(rangespace(R),k)
    J=block(domainspace(R),j)
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    if K == J && κ == ξ
        T((-2κ^2 + 2K^2 - 2κ*(-1 + β + γ) + (1 + α)*(-1 + α + β + γ) + 2K*(α + β + γ))/
                    ((-1 + 2K + α + β + γ)*(1 + 2K + α + β + γ)))
    elseif K==J-1 && κ == ξ
        T(((1 - κ + K + α)*(κ + K + β + γ))/((1 + 2K + α + β + γ)*(2 + 2K + α + β + γ)))
    elseif K==J+1 && κ == ξ
        T(((1 + J - κ)*(J + κ + α + β + γ))/((2 + 2*(-1 + J) + α + β + γ)*(3 + 2*(-1 + J) + α + β + γ)))
    else
        zero(T)
    end
end

function getindex{T}(R::Recurrence{2,KoornwinderTriangle,T},k::Integer,j::Integer)
    α,β,γ=R.space.α,R.space.β,R.space.γ
    K=block(rangespace(R),k)
    J=block(domainspace(R),j)
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    if K==J && κ == ξ
        T((κ-1+β)*(K+κ+β+γ-1)/((2κ-1+β+γ)*(2K+α+β+γ)))
    elseif K==J && κ == ξ+1
        T(-ξ*(K-ξ+α)/((2ξ-1+β+γ)*(2K+α+β+γ)))
    elseif K==J+1 && κ == ξ
        T(-(κ-1+β)*(J-κ+1)/((2κ-1+β+γ)*(2J+α+β+γ)))
    elseif K==J+1 && κ == ξ+1
        T(ξ*(J+ξ+α+β+γ)/((2ξ-1+β+γ)*(2J+α+β+γ)))
    else
        zero(T)
    end
end
