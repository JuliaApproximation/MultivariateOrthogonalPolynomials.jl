export Triangle, KoornwinderTriangle, ProductTriangle, TriangleWeight, WeightedTriangle


## Triangle Def
# currently right trianglel
immutable Triangle <: BivariateDomain{Vec{2,Float64}} end


#canonical is rectangle [0,1]^2
# with the map (x,y)=(s,(1-s)*t)
canonicaldomain(::Triangle) = Segment(0,1)^2
fromcanonical(::Triangle,st::Vec) = Vec(st[1],(1-st[1])*st[2])
tocanonical(::Triangle,xy::Vec) = Vec(xy[1],xy[1]==1 ? zero(eltype(xy)) : xy[2]/(1-xy[1]))
checkpoints(d::Triangle) = [fromcanonical(d,Vec(.1,.2243)),fromcanonical(d,Vec(-.212423,-.3))]

∂(d::Triangle) = PiecewiseSegment([Vec(0.,0.),Vec(1.,0.),Vec(0.,1.),Vec(0.,0.)])

# expansion in OPs orthogonal to
# x^α*y^β*(1-x-y)^γ
# defined as
# P_{n-k}^{2k+β+γ+1,α}(2x-1)*(1-x)^k*P_k^{γ,β}(2y/(1-x)-1)


immutable ProductTriangle <: AbstractProductSpace{Tuple{WeightedJacobi{Float64,Segment{Float64}},
                                                        Jacobi{Float64,Segment{Float64}}},Float64,Triangle,2}
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

points(K::KoornwinderTriangle,n::Integer) =
    map(Vec,map(vec,points(ProductTriangle(K),round(Int,sqrt(n)),round(Int,sqrt(n))))...)

points(K::Triangle,n::Integer) = points(KoornwinderTriangle(0,0,0),n)

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
tensorizer(K::TriangleSpace) = Tensorizer((ApproxFun.repeated(true),ApproxFun.repeated(true)))

# we have each polynomial
blocklengths(K::TriangleSpace) = 1:∞

for OP in (:block,:blockstart,:blockstop)
    @eval begin
        $OP(s::TriangleSpace,M::Block) = $OP(tensorizer(s),M)
        $OP(s::TriangleSpace,M) = $OP(tensorizer(s),M)
    end
end


# support for ProductFun constructor

function space(T::ProductTriangle,k::Integer)
    @assert k==2
    Jacobi(T.β,T.γ,Segment(0.,1.))
end

columnspace(T::ProductTriangle,k::Integer) =
    JacobiWeight(0.,k-1.,Jacobi(T.α,2k-1+T.β+T.γ,Segment(0.,1.)))

Base.sum{KT<:KoornwinderTriangle}(f::Fun{KT}) =
    Fun(f,KoornwinderTriangle(0,0,0)).coefficients[1]/2

# convert coefficients

Fun(f::F,S::KoornwinderTriangle;kwds...) =
    Fun(Fun(ProductFun(f,ProductTriangle(S))),S)



immutable KoornwinderTriangleITransformPlan{PE,PT}
    plan::PE
    points::PT
end

*(P::KoornwinderTriangleITransformPlan,cfs::Vector) = P.plan.(P.points)

plan_itransform(S::KoornwinderTriangle,cfs) =
    KoornwinderTriangleITransformPlan(plan_evaluate(Fun(S,cfs)),points(S,length(cfs)))

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


function clenshaw2D{T}(Jx,Jy,cfs::Vector{Vector{T}},x,y)
    N=length(cfs)
    bk1=zeros(T,N+1)
    bk2=zeros(T,N+2)

    Abk1x=zeros(T,N+1)
    Abk1y=zeros(T,N+1)
    Abk2x=zeros(T,N+1)
    Abk2y=zeros(T,N+1)


    @inbounds for K=Block(N):-1:Block(2)
        Bx,By=view(Jx,K,K),view(Jy,K,K)
        Cx,Cy=view(Jx,K,K+1),view(Jy,K,K+1)
        JxK=view(Jx,K+1,K)
        JyK=view(Jy,K+1,K)
        @inbounds for k=1:K.K
            bk1[k] /= inbands_getindex(JxK,k,k)
        end

        bk1[K.K-1] -= JyK[K.K-1,end]/(JxK[K.K-1,K.K-1]*JyK[end,end])*bk1[K.K+1]
        bk1[K.K]   -= JyK[K.K,end]/(JxK[K.K,K.K]*JyK[end,end])*bk1[K.K+1]
        bk1[K.K+1] /= JyK[K.K+1,end]

        resize!(Abk2x,K.K)
        BLAS.blascopy!(K.K,bk1,1,Abk2x,1)
        resize!(Abk2y,K.K)
        Abk2y[1:K.K-1]=0
        Abk2y[end]=bk1[K.K+1]

        Abk1x,Abk2x=Abk2x,Abk1x
        Abk1y,Abk2y=Abk2y,Abk1y

        bk2 = bk1  ::Vector{T}

        bk1 = (x*Abk1x) ::Vector{T}
        Base.axpy!(y,Abk1y,bk1)
        αA_mul_B_plus_βC!(-one(T),Bx,Abk1x,one(T),bk1)
        αA_mul_B_plus_βC!(-one(T),By,Abk1y,one(T),bk1)
        αA_mul_B_plus_βC!(-one(T),Cx,Abk2x,one(T),bk1)
        αA_mul_B_plus_βC!(-one(T),Cy,Abk2y,one(T),bk1)
        Base.axpy!(one(T),cfs[K.K],bk1)
    end

    K = Block(1)
    Bx,By=view(Jx,K,K),view(Jy,K,K)
    Cx,Cy=view(Jx,K,K+1),view(Jy,K,K+1)
    JxK=view(Jx,K+1,K)
    JyK=view(Jy,K+1,K)

    bk1[1] /= JxK[1,1]
    bk1[1] -= JyK[1,end]/(JxK[1,1]*JyK[2,end])*bk1[2]
    bk1[2] /= JyK[2,end]

    Abk1x,Abk2x=bk1[1:1],Abk1x
    Abk1y,Abk2y=[bk1[2]],Abk1y
    (cfs[1] + x*Abk1x + y*Abk1y - Bx*Abk1x - By*Abk1y - Cx*Abk2x - Cy*Abk2y)[1]::T
end

# convert to vector of coefficients
# TODO: replace with RaggedMatrix
function totree(S,f::Fun)
    N=block(S,ncoefficients(f))
    ret = Array(Vector{eltype(f)},N.K)
    for K=Block(1):N
        ret[K.K]=coefficient(f,K)
    end

    ret
end

immutable TriangleEvaluatePlan{S,RX,RY,T}
    space::S
    coefficients::Vector{Vector{T}}
    Jx::RX
    Jy::RY
end

function plan_evaluate(f::Fun{KoornwinderTriangle},x...)
    N = nblocks(f)
    S = space(f)
    TriangleEvaluatePlan(S,
                totree(S,f),
                (Lowering{1}(S)→S)[Block(1):Block(N+2),Block(1):Block(N+1)],
                (Lowering{2}(S)→S)[Block(1):Block(N+2),Block(1):Block(N+1)])
end

(P::TriangleEvaluatePlan)(x,y) = clenshaw2D(P.Jx,P.Jy,P.coefficients,x,y)

(P::TriangleEvaluatePlan)(pt::Vec) = P(pt...)

# evaluate(f::AbstractVector,K::KoornwinderTriangle,x...) =
#     evaluate(coefficients(f,K,ProductTriangle(K)),ProductTriangle(K),x...)

evaluate(f::AbstractVector,K::KoornwinderTriangle,x...) = plan_evaluate(Fun(K,f))(x...)

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


isbandedblockbanded(::ConcreteDerivative{KoornwinderTriangle}) = true


blockbandinds(D::ConcreteDerivative{KoornwinderTriangle}) = 0,sum(D.order)
subblockbandinds(D::ConcreteDerivative{KoornwinderTriangle}) = (0,sum(D.order))
subblockbandinds(D::ConcreteDerivative{KoornwinderTriangle},k::Integer) = k==1 ? 0 : sum(D.order)

function getindex(D::ConcreteDerivative{KoornwinderTriangle},k::Integer,j::Integer)
    T=eltype(D)
    S=domainspace(D)
    α,β,γ = S.α,S.β,S.γ
    K=block(rangespace(D),k).K
    J=block(S,j).K
    κ=k-blockstart(rangespace(D),K)+1
    ξ=j-blockstart(S,J)+1

    if D.order==[0,1]
        if J==K+1 && κ+1 == ξ
            T(1+κ+β+γ)
        else
            zero(T)
        end
    elseif D.order==[1,0]
        if J == K+1 && κ == ξ
            T((1+κ+K+α+β+γ)*(κ+γ+β)/(2κ+γ+β-1))
        elseif J == K+1 && κ+1 == ξ
            T((κ+β)*(κ+K+γ+β+1)/(2κ+γ+β+1))
        else
            zero(T)
        end
    else
        error("Not implemented")
    end
end

function Base.convert{T}(::Type{BandedBlockBandedMatrix},S::SubOperator{T,ConcreteDerivative{KoornwinderTriangle,Vector{Int},T},
                                                                        Tuple{UnitRange{Block},UnitRange{Block}}})
    ret = bbbzeros(S)
    D = parent(S)
    sp=domainspace(D)
    α,β,γ = sp.α,sp.β,sp.γ
    K_sh = first(parentindexes(S)[1])-1
    J_sh = first(parentindexes(S)[2])-1
    N,M=blocksize(ret)::Tuple{Int,Int}

    if D.order == [1,0]
        for K=Block.(1:N)
            J = K+K_sh-J_sh+1
            if 1 ≤ J ≤ M
                bl = view(ret,K,J)
                KK = size(bl,1)
                @inbounds for κ=1:KK
                    bl[κ,κ] = (1+κ+KK+α+β+γ)*(κ+γ+β)/(2κ+γ+β-1)
                    bl[κ,κ+1] = (κ+β)*(κ+KK+γ+β+1)/(2κ+γ+β+1)
                end
            end
        end
    elseif D.order == [0,1]
        for K=Block.(1:N)
            J = K+K_sh-J_sh+1
            if 1 ≤ J ≤ M
                bl = view(ret,K,J)
                KK = size(bl,1)
                @inbounds for κ=1:KK
                    bl[κ,κ+1] = 1+κ+β+γ
                end
            end
        end
    end
    ret
end



## Conversion

union_rule(K1::KoornwinderTriangle,K2::KoornwinderTriangle) =
    KoornwinderTriangle(min(K1.α,K2.α),min(K1.β,K2.β),min(K1.γ,K2.γ))
conversion_rule(K1::KoornwinderTriangle,K2::KoornwinderTriangle) =
    KoornwinderTriangle(min(K1.α,K2.α),min(K1.β,K2.β),min(K1.γ,K2.γ))
maxspace_rule(K1::KoornwinderTriangle,K2::KoornwinderTriangle) =
    KoornwinderTriangle(max(K1.α,K2.α),max(K1.β,K2.β),max(K1.γ,K2.γ))

function Conversion(K1::KoornwinderTriangle,K2::KoornwinderTriangle)
    @assert K1.α≤K2.α && K1.β≤K2.β && K1.γ≤K2.γ &&
        isapproxinteger(K1.α-K2.α) && isapproxinteger(K1.β-K2.β) &&
        isapproxinteger(K1.γ-K2.γ)

    if (K1.α==K2.α && K1.β==K2.β && K1.γ==K2.γ)
        ConversionWrapper(eye(K1))
    elseif (K1.α+1==K2.α && K1.β==K2.β && K1.γ==K2.γ) ||
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
        error("There is a bug: cannot convert $K1 to $K2")
    end
end


isbandedblockbanded(::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle}) = true


blockbandinds(C::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle}) = (0,1)

subblockbandinds(C::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle}) = (0,1)
subblockbandinds(C::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle},k::Integer) = k==1?0:1



function getindex{T}(C::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle,T},k::Integer,j::Integer)
    K1=domainspace(C);K2=rangespace(C)
    α,β,γ = K1.α,K1.β,K1.γ
    K=block(K2,k).K
    J=block(K1,j).K
    κ=k-blockstart(K2,K)+1
    ξ=j-blockstart(K1,J)+1

    if K2.α == α+1 && K2.β == β && K2.γ == γ
        if     K == J    && κ == ξ
            T((J+ξ+α+β+γ)/(2J+α+β+γ))
        elseif J == K+1  && κ == ξ
            T((J+ξ+β+γ-1)/(2J+α+β+γ))
        else
            zero(T)
        end
    elseif K2.α==α && K2.β==β+1 && K2.γ==γ
        if     J == K   && κ == ξ
            (β+γ == -1) && return T((K+κ+α+β+γ)/(2*(2K+α+β+γ)))
            T((K+κ+α+β+γ)/(2K+α+β+γ)*(κ+β+γ)/(2κ+β+γ-1))
        elseif J == K   && κ+1 == ξ
            T(-(κ+γ)/(2κ+β+γ+1)*(K-κ)/(2K+α+β+γ))
        elseif J == K+1 && κ == ξ
            (β+γ == -1) && return T(-(K-κ+α+1)/(2*(2K+α+β+γ+2)))
            T(-(K-κ+α+1)/(2K+α+β+γ+2)*(κ+β+γ)/(2κ+β+γ-1))
        elseif J == K+1 && κ+1 == ξ
            T((κ+γ)/(2κ+β+γ+1)*(K+κ+β+γ+1)/(2K+α+β+γ+2))
        else
            zero(T)
        end
    elseif K2.α==α && K2.β==β && K2.γ==γ+1
        if K == J && κ == ξ
            (β+γ == -1) && return T((K+κ+α+β+γ)/(2*(2K+α+β+γ)))
            T((K+κ+α+β+γ)/(2K+α+β+γ)*(κ+β+γ)/(2κ+β+γ-1))
        elseif K == J && κ+1 == ξ
            T((κ+β)/(2κ+β+γ+1)*(K-κ)/(2K+α+β+γ))
        elseif J == K+1 && κ == ξ
            (β+γ == -1) && return T(-(K-κ+α+1)/(2*(2K+α+β+γ+2)))
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


function Base.convert{T}(::Type{BandedBlockBandedMatrix},S::SubOperator{T,ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle,T},
                                                                        Tuple{UnitRange{Block},UnitRange{Block}}})
    ret = bbbzeros(S)
    K1=domainspace(parent(S))
    K2=rangespace(parent(S))
    α,β,γ = K1.α,K1.β,K1.γ
    K_sh = first(parentindexes(S)[1])-1
    J_sh = first(parentindexes(S)[2])-1
    N,M=blocksize(ret)::Tuple{Int,Int}

    if K2.α == α+1 && K2.β == β && K2.γ == γ
        for KK=Block.(1:N)
            JJ = KK+K_sh-J_sh  # diagonal
            if 1 ≤ JJ ≤ M
                bl = view(ret,KK,JJ)
                J = size(bl,2)
                @inbounds for ξ=1:J
                    bl[ξ,ξ] = (J+ξ+α+β+γ)/(2J+α+β+γ)
                end
            end
            JJ = KK+K_sh-J_sh+1  # super-diagonal
            if 1 ≤ JJ ≤ M
                bl = view(ret,KK,JJ)
                J = size(bl,2)
                @inbounds for ξ=1:J-1
                    bl[ξ,ξ] = (J+ξ+β+γ-1)/(2J+α+β+γ)
                end
            end
        end
    elseif K2.α==α && K2.β==β+1 && K2.γ==γ && β+γ==-1
        for KK=Block.(1:N)
            J = KK+K_sh-J_sh  # diagonal
            if 1 ≤ J ≤ M
                bl = view(ret,KK,J)
                K = size(bl,1)
                s=2K+α+β+γ
                @inbounds for κ=1:K
                    bl[κ,κ] = (K+κ+α+β+γ)/(2s)
                end
                @inbounds for κ=1:K-1
                    bl[κ,κ+1] = -(κ+γ)/(2κ+β+γ+1)*(K-κ)/s
                end
            end
            J = KK+K_sh-J_sh+1  # super-diagonal
            if 1 ≤ J ≤ M
                bl = view(ret,KK,J)
                K = size(bl,1)
                s=T(2K+α+β+γ+2)

                @inbounds for κ=1:K
                    bl[κ,κ] = -(K-κ+α+1)/(2s)
                    bl[κ,κ+1] = (κ+γ)/(2κ+β+γ+1)*(K+κ+β+γ+1)/s
                end
            end
        end
    elseif K2.α==α && K2.β==β+1 && K2.γ==γ
        for KK=Block.(1:N)
            J = KK+K_sh-J_sh  # diagonal
            if 1 ≤ J ≤ M
                bl = view(ret,KK,J)
                K = size(bl,1)
                @inbounds for κ=1:K
                    bl[κ,κ] = (K+κ+α+β+γ)/(2K+α+β+γ)*(κ+β+γ)/(2κ+β+γ-1)
                end
                @inbounds for κ=1:K-1
                    bl[κ,κ+1] = -(κ+γ)/(2κ+β+γ+1)*(K-κ)/(2K+α+β+γ)
                end
            end
            J = KK+K_sh-J_sh+1  # super-diagonal
            if 1 ≤ J ≤ M
                bl = view(ret,KK,J)
                K = size(bl,1)
                @inbounds for κ=1:K
                    bl[κ,κ] = -(K-κ+α+1)/(2K+α+β+γ+2)*(κ+β+γ)/(2κ+β+γ-1)
                    bl[κ,κ+1] = (κ+γ)/(2κ+β+γ+1)*(K+κ+β+γ+1)/(2K+α+β+γ+2)
                end
            end
        end
    elseif K2.α==α && K2.β==β && K2.γ==γ+1  && β+γ==-1
        for KK=Block.(1:N)
            J = KK+K_sh-J_sh  # diagonal
            if 1 ≤ J ≤ M
                bl = view(ret,KK,J)
                K = size(bl,1)
                s=2K+α+β+γ
                @inbounds for κ=1:K
                    bl[κ,κ] = (K+κ+α+β+γ)/(2s)
                end
                @inbounds for κ=1:K-1
                   bl[κ,κ+1] = (κ+β)/(2κ+β+γ+1)*(K-κ)/s
               end
            end
            J = KK+K_sh-J_sh+1  # super-diagonal
            if 1 ≤ J ≤ M
                bl = view(ret,KK,J)
                K = size(bl,1)
                s=2K+α+β+γ+2
                @inbounds for κ=1:K
                    bl[κ,κ] = -(K-κ+α+1)/(2s)
                    bl[κ,κ+1] = -(κ+β)/(2κ+β+γ+1)*(K+κ+β+γ+1)/s
                end
            end
        end
    elseif K2.α==α && K2.β==β && K2.γ==γ+1
        for KK=Block.(1:N)
            J = KK+K_sh-J_sh  # diagonal
            if 1 ≤ J ≤ M
                bl = view(ret,KK,J)
                K = size(bl,1)
                @inbounds for κ=1:K
                    bl[κ,κ] = (K+κ+α+β+γ)/(2K+α+β+γ)*(κ+β+γ)/(2κ+β+γ-1)
                end
                @inbounds for κ=1:K-1
                   bl[κ,κ+1] = (κ+β)/(2κ+β+γ+1)*(K-κ)/(2K+α+β+γ)
               end
            end
            J = KK+K_sh-J_sh+1  # super-diagonal
            if 1 ≤ J ≤ M
                bl = view(ret,KK,J)
                K = size(bl,1)
                @inbounds for κ=1:K
                    bl[κ,κ] = -(K-κ+α+1)/(2K+α+β+γ+2)*(κ+β+γ)/(2κ+β+γ-1)
                    bl[κ,κ+1] = -(κ+β)/(2κ+β+γ+1)*(K+κ+β+γ+1)/(2K+α+β+γ+2)
                end
            end
        end
    end
    ret
end



## Jacobi Operators

# k is 1, 2, ... for x, y, z,...
immutable Lowering{k,S,T} <: Operator{T}
    space::S
end

Base.convert{k}(::Type{Lowering{k}},sp) = Lowering{k,typeof(sp),promote_type(eltype(sp),eltype(eltype(domain(sp))))}(sp)
Base.convert{x,T,S}(::Type{Operator{T}},J::Lowering{x,S}) = Lowering{x,S,T}(J.space)


domainspace(R::Lowering) = R.space

isbandedblockbanded(::Lowering) = true

blockbandinds(::Lowering) = (-1,0)

subblockbandinds(::Lowering{1,KoornwinderTriangle}) = (0,0)
subblockbandinds(::Lowering{2,KoornwinderTriangle}) = (-1,0)
subblockbandinds(::Lowering{3,KoornwinderTriangle}) = (-1,0)

subblockbandinds(::Lowering{1,KoornwinderTriangle},k::Integer) = 0
subblockbandinds(::Lowering{2,KoornwinderTriangle},k::Integer) = k==1? -1 : 0
subblockbandinds(::Lowering{3,KoornwinderTriangle},k::Integer) = k==1? -1 : 0


rangespace(R::Lowering{1,KoornwinderTriangle}) =
    KoornwinderTriangle(R.space.α-1,R.space.β,R.space.γ)
rangespace(R::Lowering{2,KoornwinderTriangle}) =
    KoornwinderTriangle(R.space.α,R.space.β-1,R.space.γ)
rangespace(R::Lowering{3,KoornwinderTriangle}) =
    KoornwinderTriangle(R.space.α,R.space.β,R.space.γ-1)


function getindex{T}(R::Lowering{1,KoornwinderTriangle,T},k::Integer,j::Integer)
    α,β,γ=R.space.α,R.space.β,R.space.γ
    K=block(rangespace(R),k).K
    J=block(domainspace(R),j).K
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = 2J+α+β+γ

    if K==J && κ == ξ
        T((J-ξ+α)/s)
    elseif K==J+1 && κ == ξ
        T((J-ξ+1)/s)
    else
        zero(T)
    end
end


function getindex{T}(R::Lowering{2,KoornwinderTriangle,T},k::Integer,j::Integer)
    α,β,γ=R.space.α,R.space.β,R.space.γ
    K=block(rangespace(R),k).K
    J=block(domainspace(R),j).K
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = (2ξ-1+β+γ)*(2J+α+β+γ)

    if K==J && κ == ξ
        T((ξ-1+β)*(J+ξ+β+γ-1)/s)
    elseif K==J && κ == ξ+1
        T(-ξ*(J-ξ+α)/s)
    elseif K==J+1 && κ == ξ
        T(-(ξ-1+β)*(J-ξ+1)/s)
    elseif K==J+1 && κ == ξ+1
        T(ξ*(J+ξ+α+β+γ)/s)
    else
        zero(T)
    end
end

function getindex{T}(R::Lowering{3,KoornwinderTriangle,T},k::Integer,j::Integer)
    α,β,γ=R.space.α,R.space.β,R.space.γ
    K=block(rangespace(R),k).K
    J=block(domainspace(R),j).K
    κ=k-blockstart(rangespace(R),K)+1
    ξ=j-blockstart(domainspace(R),J)+1

    s = (2ξ+β+γ-1)*(2J+α+β+γ)

    if K==J && κ == ξ
        T((ξ-1+γ)*(J+ξ+β+γ-1)/s)
    elseif K==J && κ == ξ+1
        T(ξ*(J-ξ+α)/s)
    elseif K==J+1 && κ == ξ
        T(-(ξ-1+γ)*(J-ξ+1)/s)
    elseif K==J+1 && κ == ξ+1
        T(-ξ*(J+ξ+α+β+γ)/s)
    else
        zero(T)
    end
end


function Base.convert{T}(::Type{BandedBlockBandedMatrix},S::SubOperator{T,Lowering{1,KoornwinderTriangle,T},
                                                                        Tuple{UnitRange{Block},UnitRange{Block}}})
    ret = bbbzeros(S)
    R = parent(S)
    α,β,γ=R.space.α,R.space.β,R.space.γ
    K_sh = first(parentindexes(S)[1])-1
    J_sh = first(parentindexes(S)[2])-1
    N,M=blocksize(ret)::Tuple{Int,Int}

    for KK=Block.(1:N)
        JJ = KK+K_sh-J_sh-1  # super-diagonal
        if 1 ≤ JJ ≤ M
            bl = view(ret,KK,JJ)
            J = size(bl,2)
            s = 2J+α+β+γ
            @inbounds for ξ=1:J
                bl[ξ,ξ] = (J-ξ+1)/s
            end
        end
        JJ = KK+K_sh-J_sh  # diagonal
        if 1 ≤ JJ ≤ M
            bl = view(ret,KK,JJ)
            J = size(bl,2)
            s = 2J+α+β+γ
            @inbounds for ξ=1:J
                bl[ξ,ξ] = (J-ξ+α)/s
            end
        end
    end
    ret
end


function Base.convert{T}(::Type{BandedBlockBandedMatrix},S::SubOperator{T,Lowering{2,KoornwinderTriangle,T},
                                                                        Tuple{UnitRange{Block},UnitRange{Block}}})
    ret = bbbzeros(S)
    R = parent(S)
    α,β,γ=R.space.α,R.space.β,R.space.γ
    K_sh = first(parentindexes(S)[1])-1
    J_sh = first(parentindexes(S)[2])-1
    N,M=blocksize(ret)::Tuple{Int,Int}


    for KK=Block.(1:N)
        JJ = KK+K_sh-J_sh-1  # super-diagonal
        if 1 ≤ JJ ≤ M
            bl = view(ret,KK,JJ)
            J = size(bl,2)

            @inbounds for ξ=1:J
                s = (2ξ-1+β+γ)*(2J+α+β+γ)
                bl[ξ,ξ] = -(ξ-1+β)*(J-ξ+1)/s
                bl[ξ+1,ξ] = ξ*(J+ξ+α+β+γ)/s
            end
        end
        JJ = KK+K_sh-J_sh  # diagonal
        if 1 ≤ JJ ≤ M
            bl = view(ret,KK,JJ)
            J = size(bl,2)
            @inbounds for ξ=1:J
                s = (2ξ-1+β+γ)*(2J+α+β+γ)
                bl[ξ,ξ] = (ξ-1+β)*(J+ξ+β+γ-1)/s
                bl[ξ+1,ξ] = -ξ*(J-ξ+α)/s
            end
        end
    end
    ret
end

function Base.convert{T}(::Type{BandedBlockBandedMatrix},S::SubOperator{T,Lowering{3,KoornwinderTriangle,T},
                                                                        Tuple{UnitRange{Block},UnitRange{Block}}})
    ret = bbbzeros(S)
    R = parent(S)
    α,β,γ=R.space.α,R.space.β,R.space.γ
    K_sh = first(parentindexes(S)[1])-1
    J_sh = first(parentindexes(S)[2])-1
    N,M=blocksize(ret)::Tuple{Int,Int}

    for KK=Block.(1:N)
        JJ = KK+K_sh-J_sh-1  # super-diagonal
        if 1 ≤ JJ ≤ M
            bl = view(ret,KK,JJ)
            J = size(bl,2)

            @inbounds for ξ=1:J
                s = (2ξ-1+β+γ)*(2J+α+β+γ)
                bl[ξ,ξ] = -(ξ-1+γ)*(J-ξ+1)/s
                bl[ξ+1,ξ] = -ξ*(J+ξ+α+β+γ)/s
            end
        end
        JJ = KK+K_sh-J_sh  # diagonal
        if 1 ≤ JJ ≤ M
            bl = view(ret,KK,JJ)
            J = size(bl,2)
            @inbounds for ξ=1:J
                s = (2ξ-1+β+γ)*(2J+α+β+γ)
                bl[ξ,ξ] = (ξ-1+γ)*(J+ξ+β+γ-1)/s
                bl[ξ+1,ξ] = ξ*(J-ξ+α)/s
            end
        end
    end
    ret
end


### Weighted

immutable TriangleWeight{S} <: WeightSpace{S,RealBasis,Triangle,2}
    α::Float64
    β::Float64
    γ::Float64
    space::S

    TriangleWeight(α::Number,β::Number,γ::Number,sp::S) =
        new(Float64(α),Float64(β),Float64(γ),sp)
end

TriangleWeight(α::Number,β::Number,γ::Number,sp::Space) =
    TriangleWeight{typeof(sp)}(α,β,γ,sp)

WeightedTriangle(α::Number,β::Number,γ::Number) =
    TriangleWeight(α,β,γ,KoornwinderTriangle(α,β,γ))

weight(S::TriangleWeight,x,y) = x.^S.α.*y.^S.β.*(1-x-y).^S.γ
weight(S::TriangleWeight,xy::Vec) = weight(S,xy...)

immutable TriangleWeightEvaluatePlan{S,PP}
    space::S
    plan::PP
end

plan_evaluate{SS}(f::Fun{TriangleWeight{SS}},xy...) =
    TriangleWeightEvaluatePlan(space(f),
        plan_evaluate(Fun(space(f).space,coefficients(f)),xy...))

(P::TriangleWeightEvaluatePlan)(xy...) =
    weight(P.space,xy...)*P.plan(xy...)


itransform(S::TriangleWeight,cfs::Vector) =
    plan_evaluate(Fun(S,cfs)).(points(S,length(cfs)))

#TODO: Move to Singulariaties.jl
for func in (:blocklengths,:tensorizer)
    @eval $func(S::TriangleWeight) = $func(S.space)
end

for OP in (:block,:blockstart,:blockstop)
    @eval $OP(s::TriangleWeight,M::Integer) = $OP(s.space,M)
end


spacescompatible(A::TriangleWeight,B::TriangleWeight) = A.α ≈ B.α && A.β ≈ B.β && A.γ ≈ B.γ &&
                                                        spacescompatible(A.space,B.space)




function conversion_rule(A::TriangleWeight,B::TriangleWeight)
    if isapproxinteger(A.α-B.α) && isapproxinteger(A.β-B.β) && isapproxinteger(A.γ-B.γ)
        ct=conversion_type(A.space,B.space)
        ct==NoSpace()?NoSpace():TriangleWeight(max(A.α,B.α),max(A.β,B.β),max(A.γ,B.γ),ct)
    else
        NoSpace()
    end
end

function maxspace_rule(A::TriangleWeight,B::TriangleWeight)
    if isapproxinteger(A.α-B.α) && isapproxinteger(A.β-B.β) && isapproxinteger(A.γ-B.γ)
        ms=maxspace(A.space,B.space)
        if min(A.α,B.α)==0.0 && min(A.β,B.β) == 0.0 && min(A.γ,B.γ) == 0.0
            return ms
        else
            return TriangleWeight(min(A.α,B.α),min(A.β,B.β),min(A.γ,B.γ),ms)
        end
    end
    NoSpace()
end

maxspace_rule(A::TriangleWeight,B::KoornwinderTriangle) = maxspace(A,TriangleWeight(0.,0.,0.,B))

conversion_rule(A::TriangleWeight,B::KoornwinderTriangle) = conversion_type(A,TriangleWeight(0.,0.,0.,B))

function Conversion(A::TriangleWeight,B::TriangleWeight)
    @assert isapproxinteger(A.α-B.α) && isapproxinteger(A.β-B.β) && isapproxinteger(A.γ-B.γ)
    @assert A.α≥B.α && A.β≥B.β && A.γ≥B.γ

    if A.α ≈ B.α && A.β ≈ B.β && A.γ ≈ B.γ
        ConversionWrapper(SpaceOperator(Conversion(A.space,B.space),A,B))
    elseif A.α ≈ B.α+1 && A.β ≈ B.β && A.γ ≈ B.γ &&
           A.space.α ≈ B.space.α+1 && A.space.β ≈ B.space.β && A.space.γ ≈ B.space.γ
        ConversionWrapper(SpaceOperator(Lowering{1}(A.space),A,B))
    elseif A.α ≈ B.α && A.β ≈ B.β+1 && A.γ ≈ B.γ &&
           A.space.α ≈ B.space.α && A.space.β ≈ B.space.β+1 && A.space.γ ≈ B.space.γ
        ConversionWrapper(SpaceOperator(Lowering{2}(A.space),A,B))
    elseif A.α ≈ B.α && A.β ≈ B.β && A.γ ≈ B.γ+1 &&
           A.space.α ≈ B.space.α && A.space.β ≈ B.space.β && A.space.γ ≈ B.space.γ+1
        ConversionWrapper(SpaceOperator(Lowering{3}(A.space),A,B))
    elseif A.α ≈ B.α+1 && A.β ≈ B.β && A.γ ≈ B.γ
        Jx = Lowering{1}(A.space)
        C = Conversion(rangespace(Jx),B.space)
        ConversionWrapper(SpaceOperator(C*Jx,A,B))
    elseif A.α ≈ B.α && A.β ≈ B.β+1 && A.γ ≈ B.γ
        Jy = Lowering{2}(A.space)
        C = Conversion(rangespace(Jy),B.space)
        ConversionWrapper(SpaceOperator(C*Jy,A,B))
    elseif A.α ≈ B.α && A.β ≈ B.β && A.γ ≈ B.γ+1
        Jz = Lowering{3}(A.space)
        C = Conversion(rangespace(Jz),B.space)
        ConversionWrapper(SpaceOperator(C*Jz,A,B))
    elseif A.α ≥ B.α+1
        Conversion(A,TriangleWeight(A.α-1,A.β,A.γ,
                                    KoornwinderTriangle(A.space.α-1,A.space.β,A.space.γ)),B)
    elseif A.β ≥ B.β+1
        Conversion(A,TriangleWeight(A.α,A.β-1,A.γ,
                                        KoornwinderTriangle(A.space.α,A.space.β-1,A.space.γ)),B)
    elseif A.γ ≥ B.γ+1
        Conversion(A,TriangleWeight(A.α,A.β,A.γ-1,
                                        KoornwinderTriangle(A.space.α,A.space.β,A.space.γ-1)),B)
    else
        error("Somethings gone wrong!")
    end
end

Conversion(A::TriangleWeight,B::KoornwinderTriangle) =
    ConversionWrapper(SpaceOperator(
        Conversion(A,TriangleWeight(0.,0.,0.,B)),
        A,B))


function triangleweight_Derivative(S::TriangleWeight,order)
    if S.α == S.β == S.γ == 0
        D=Derivative(S.space,order)
        SpaceOperator(D,S,rangespace(D))
    elseif S.α == S.β == S.γ == S.space.α == S.space.β == S.space.γ == 1
        C=Conversion(S,KoornwinderTriangle(0,0,0))
        D = Derivative(rangespace(C),order)
        SpaceOperator(D*C,S,rangespace(D))
    elseif order[2] == 0 && S.α == 0 && S.γ == 0
        Dx = Derivative(S.space,order)
        DerivativeWrapper(
            SpaceOperator(Dx,S,TriangleWeight(S.α,S.β,S.γ,rangespace(Dx))),
            order)
    elseif order[1] == 0 && S.β == 0 && S.γ == 0
        Dy = Derivative(S.space,order)
        DerivativeWrapper(
            SpaceOperator(Dy,S,TriangleWeight(S.α,S.β,S.γ,rangespace(Dy))),
            order)
    elseif order == [1,0] && S.α == 0
        Dx = Derivative(S.space,order)
        Jx = Lowering{1}(rangespace(Dx))
        Jy = Lowering{2}(rangespace(Dx))
        A = -S.γ*I + (I-Jx-Jy)*Dx
        DerivativeWrapper(
            SpaceOperator(A,S,TriangleWeight(S.α,S.β,S.γ-1,rangespace(A))),
            order)
    elseif order == [1,0] && S.γ == 0
        Dx = Derivative(S.space,order)
        Jx = Lowering{1}(rangespace(Dx))
        A = S.α*I + Jx*Dx
        DerivativeWrapper(
            SpaceOperator(A,S,TriangleWeight(S.α-1,S.β,S.γ,rangespace(A))),
            order)
    elseif order == [1,0]
        Dx = Derivative(S.space,order)
        Jx = Lowering{1}(rangespace(Dx))
        Jy = Lowering{2}(rangespace(Dx))
        A = S.α*(I-Jx-Jy) - S.γ*Jx + Jx*(I-Jx-Jy)*Dx
        DerivativeWrapper(
            SpaceOperator(A,S,TriangleWeight(S.α-1,S.β,S.γ-1,rangespace(A))),
            order)
    elseif order == [0,1] && S.β == 0
        Dy = Derivative(S.space,order)
        Jx = Lowering{1}(rangespace(Dy))
        Jy = Lowering{2}(rangespace(Dy))
        A = -S.γ*I + (I-Jx-Jy)*Dy
        DerivativeWrapper(
            SpaceOperator(A,S,TriangleWeight(S.α,S.β,S.γ-1,rangespace(A))),
            order)
    elseif order == [0,1] && S.γ == 0
        Dy = Derivative(S.space,order)
        Jy = Lowering{2}(rangespace(Dy))
        A = S.β*I + Jy*Dy
        DerivativeWrapper(
            SpaceOperator(A,S,TriangleWeight(S.α,S.β-1,S.γ,rangespace(A))),
            order)
    elseif order == [0,1]
        Dy = Derivative(S.space,order)
        Jx = Lowering{1}(rangespace(Dy))
        Jy = Lowering{2}(rangespace(Dy))
        A = S.β*(I-Jx-Jy) - S.γ*Jy + Jy*(I-Jx-Jy)*Dy
        DerivativeWrapper(
            SpaceOperator(A,S,TriangleWeight(S.α,S.β-1,S.γ-1,rangespace(A))),
            order)
    elseif order[1] > 1
        D=Derivative(S,[1,0])
        DerivativeWrapper(TimesOperator(Derivative(rangespace(D),[order[1]-1,order[2]]),D),order)
    else
        @assert order[2] > 1
        D=Derivative(S,[0,1])
        DerivativeWrapper(TimesOperator(Derivative(rangespace(D),[order[1],order[2]-1]),D),order)
    end
end

function Derivative(S::TriangleWeight{KoornwinderTriangle},order)
    if S.α == 0 && S.β == 0 && S.γ == 0
        D = Derivative(S.space,order)
        DerivativeWrapper(SpaceOperator(D,S,rangespace(D)),order)
    elseif (order[1] ≥ 1 &&  (S.α == 0 || S.γ == 0) ) ||
            (order[2] ≥ 1 &&  (S.β == 0 || S.γ == 0))
        C = Conversion(S,KoornwinderTriangle(0,0,0))
        DerivativeWrapper(Derivative(rangespace(C),order)*C,order)
    elseif S.α == S.space.α && S.β == S.space.β && S.γ == S.space.γ
        if order == [1,0] || order == [0,1]
            ConcreteDerivative(S,order)
        elseif order[1] ≥ 1
            D1 = Derivative(S,[1,0])
            DerivativeWrapper(TimesOperator(Derivative(rangespace(D1),[order[1]-1,order[2]]),D1),order)
        else #order[2] ≥ 1
            D2 = Derivative(S,[0,1])
            DerivativeWrapper(TimesOperator(Derivative(rangespace(D2),[order[1],order[2]-1]),D2),order)
        end
    else
        triangleweight_Derivative(S,order)
    end
end
Derivative(S::TriangleWeight,order) = triangleweight_Derivative(S,order)


rangespace(D::ConcreteDerivative{TriangleWeight{KoornwinderTriangle}}) =
    WeightedTriangle(D.space.α-D.order[1],D.space.β-D.order[2],D.space.γ-1)

isbandedblockbanded(::ConcreteDerivative{TriangleWeight{KoornwinderTriangle}}) = true

blockbandinds(::ConcreteDerivative{TriangleWeight{KoornwinderTriangle}}) = (-1,0)
subblockbandinds(D::ConcreteDerivative{TriangleWeight{KoornwinderTriangle}}) =
    (-1,0)  #TODO: subblockbandinds (-1,-1) for Dy
subblockbandinds(D::ConcreteDerivative{TriangleWeight{KoornwinderTriangle}},k::Integer) =
    k==1?-1:0  #TODO: subblockbandinds (-1,-1) for Dy


function getindex{OT,T}(D::ConcreteDerivative{TriangleWeight{KoornwinderTriangle},OT,T},k::Integer,j::Integer)::T
    α,β,γ=D.space.α,D.space.β,D.space.γ
    K=block(rangespace(D),k).K
    J=block(domainspace(D),j).K
    κ=k-blockstart(rangespace(D),K)+1
    ξ=j-blockstart(domainspace(D),J)+1

    if D.order[1] == 1
        K == J+1 && ξ == κ && return -T((ξ+γ-1)*(J-ξ+1))/(2ξ+β+γ-1)
        K == J+1 && ξ+1 == κ && return -T(ξ*(J-ξ+α))/(2ξ+β+γ-1)
    else
        @assert D.order[2] == 1
        K == J+1 && ξ+1 == κ && return T(-ξ)
    end

    zero(T)
end






## Multiplication Operators
function operator_clenshaw2D{T}(Jx,Jy,cfs::Vector{Vector{T}},x,y)
    N=length(cfs)
    S = domainspace(x)
    Z=ZeroOperator(S,S)
    bk1=Array(Operator{T},N+1);bk1[:]=Z
    bk2=Array(Operator{T},N+2);bk2[:]=Z

    Abk1x=Array(Operator{T},N+1);Abk1x[:]=Z
    Abk1y=Array(Operator{T},N+1);Abk1y[:]=Z
    Abk2x=Array(Operator{T},N+1);Abk2x[:]=Z
    Abk2y=Array(Operator{T},N+1);Abk2y[:]=Z

    for K=Block(N):-1:Block(2)
        Bx,By=view(Jx,K,K),view(Jy,K,K)
        Cx,Cy=view(Jx,K,K+1),view(Jy,K,K+1)
        JxK=view(Jx,K+1,K)
        JyK=view(Jy,K+1,K)
        @inbounds for k=1:K.K
            bk1[k] /= JxK[k,k]
        end

        bk1[K.K-1] -= JyK[K.K-1,end]/(JxK[K.K-1,K.K-1]*JyK[end,end])*bk1[K.K+1]
        bk1[K.K]   -= JyK[K.K,end]/(JxK[K.K,K.K]*JyK[end,end])*bk1[K.K+1]
        bk1[K.K+1] /= JyK[K.K+1,end]

        resize!(Abk2x,K.K)
        Abk2x[:]=bk1[1:K.K]
        resize!(Abk2y,K.K)
        Abk2y[1:K.K-1]=Z
        Abk2y[end]=bk1[K.K+1]

        Abk1x,Abk2x=Abk2x,Abk1x
        Abk1y,Abk2y=Abk2y,Abk1y


        bk1,bk2 = bk2,bk1
        resize!(bk1,K.K)
        bk1[:]=map((opx,opy)->x*opx+y*opy,Abk1x,Abk1y)
        bk1[:]-=Matrix(Bx)*Abk1x+Matrix(By)*Abk1y
        bk1[:]-=Matrix(Cx)*Abk2x+Matrix(Cy)*Abk2y
        for k=1:length(bk1)
            bk1[k]+=cfs[K.K][k]*I
        end
    end


    K =Block(1)
    Bx,By=view(Jx,K,K),view(Jy,K,K)
    Cx,Cy=view(Jx,K,K+1),view(Jy,K,K+1)
    JxK=view(Jx,K+1,K)
    JyK=view(Jy,K+1,K)

    bk1[1] /= JxK[1,1]
    bk1[1]   -= JyK[1,end]/(JxK[1,1]*JyK[2,end])*bk1[2]
    bk1[2] /= JyK[2,end]

    Abk1x,Abk2x=bk1[1:1],Abk1x
    Abk1y,Abk2y=[bk1[2]],Abk1y
    cfs[1][1]*I + x*Abk1x[1] + y*Abk1y[1] -
        Bx[1,1]*Abk1x[1] - By[1,1]*Abk1y[1] -
        (Matrix(Cx)*Abk2x)[1] - (Matrix(Cy)*Abk2y)[1]
end


function Multiplication(f::Fun{KoornwinderTriangle},S::KoornwinderTriangle)
    S1=space(f)
    op=operator_clenshaw2D(Lowering{1}(S1)→S1,Lowering{2}(S1)→S1,plan_evaluate(f).coefficients,Lowering{1}(S)→S,Lowering{2}(S)→S)
    MultiplicationWrapper(f,op)
end
