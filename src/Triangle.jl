export Triangle,KoornwinderTriangle



## Triangle Def
# currently right trianglel
immutable Triangle <: BivariateDomain{Float64} end

# expansion in OPs orthogonal to
# x^α*y^β*(1-x-y)^γ
# defined as
# P_{n-k}^{2k+β+γ+1,α}(2x-1)*(1-x)^k*P_k^{γ,β}(2y/(1-x)-1)


immutable KoornwinderTriangle <: Space{RealBasis,Triangle,2}
    α::Float64
    β::Float64
    γ::Float64
    domain::Triangle
end

KoornwinderTriangle(α,β,γ)=KoornwinderTriangle(α,β,γ,Triangle())
KoornwinderTriangle(T::Triangle)=KoornwinderTriangle(0.,0.,0.,T)
Space(T::Triangle)=KoornwinderTriangle(T)

spacescompatible(K1::KoornwinderTriangle,K2::KoornwinderTriangle)=K1.α==K2.α && K1.β==K2.β && K1.γ==K2.γ

function Derivative(K::KoornwinderTriangle,order::Vector{Int})
    @assert length(order)==2
    if order==[1,0] || order==[0,1]
        ConcreteDerivative(K,order)
    elseif order[1]>1
        D=Derivative(K,[1,0])
        DerivativeWrapper(TimesOperator(Derivative(rangespace(D),[order[1]-1,order[2]]),D),order)
    else
        @assert order[2]>1
        D=Derivative(K,[0,1])
        DerivativeWrapper(TimesOperator(Derivative(rangespace(D),[order[1],order[2]-1]),D),order)
    end
end


rangespace(D::ConcreteDerivative{KoornwinderTriangle})=KoornwinderTriangle(D.space.α+D.order[1],D.space.β+D.order[2],D.space.γ+sum(D.order),domain(D))
bandinds(D::ConcreteDerivative{KoornwinderTriangle})=0,sum(D.order)
blockbandinds(D::ConcreteDerivative{KoornwinderTriangle},k::Integer)=k==0?0:sum(D.order)

function addentries!(D::ConcreteDerivative{KoornwinderTriangle},A,kr::Range,::Colon)
    K=domainspace(D)
    α,β,γ = K.α,K.β,K.γ
    if D.order==[0,1]
        for n=kr
            Mkj=A[n,n+1]

            for k=1:size(Mkj,1)
                Mkj[k,k+1]+=(1+k+β+γ)
            end
        end
    elseif D.order==[1,0]
        for n=kr
            Mkj=A[n,n+1]

            for k=1:size(Mkj,1)
                Mkj[k,k]+=(1+k+n+α+β+γ)*(k+γ+β)/(2k+γ+β-1)
                Mkj[k,k+1]+=(k+β)*(k+n+γ+β+1)/(2k+γ+β+1)
            end
        end
    else
        error("Not implemented")
    end
    A
end



## Conversion

conversion_rule(K1::KoornwinderTriangle,K2::KoornwinderTriangle)=
    KoornwinderTriangle(min(K1.α,K2.α),min(K1.β,K2.β),min(K1.γ,K2.γ))
maxspace_rule(K1::KoornwinderTriangle,K2::KoornwinderTriangle)=
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
bandinds(C::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle})=(0,1)
blockbandinds(C::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle},k::Integer)=k==1?0:1


function addentries!(C::ConcreteConversion{KoornwinderTriangle,KoornwinderTriangle},A,kr::Range,::Colon)
    K1=domainspace(C);K2=rangespace(C)
    α,β,γ = K1.α,K1.β,K1.γ
    if K2.α==α+1 && K2.β==β && K2.γ==γ
        for n=kr
            B=A[n,n]

            for k=1:size(B,1)
                B[k,k]+=(n+k+α+β+γ)/(2n+α+β+γ)
            end

            B=A[n,n+1]
            for k=1:size(B,1)
                B[k,k]+=(n+k+β+γ)/(2n+α+β+γ+2)
            end
        end
    elseif K2.α==α && K2.β==β+1 && K2.γ==γ
        for n=kr
            B=A[n,n]

            for k=1:size(B,1)
                B[k,k]  += (n+k+α+β+γ)/(2n+α+β+γ)*(k+β+γ)/(2k+β+γ-1)
                if k+1 ≤ size(B,2)
                    B[k,k+1]-= (k+γ)/(2k+β+γ+1)*(n-k)/(2n+α+β+γ)
                end
            end

            B=A[n,n+1]
            for k=1:size(B,1)
                B[k,k]  -= (n-k+α+1)/(2n+α+β+γ+2)*(k+β+γ)/(2k+β+γ-1)
                if k+1 ≤ size(B,2)
                    B[k,k+1]+= (k+γ)/(2k+β+γ+1)*(n+k+β+γ+1)/(2n+α+β+γ+2)
                end
            end
        end
    elseif K2.α==α && K2.β==β && K2.γ==γ+1
        for n=kr
            B=A[n,n]

            for k=1:size(B,1)
                B[k,k]  += (n+k+α+β+γ)/(2n+α+β+γ)*(k+β+γ)/(2k+β+γ-1)
                if k+1 ≤ size(B,2)
                    B[k,k+1]-= (k+β)/(2k+β+γ+1)*(n-k)/(2n+α+β+γ)
                end
            end

            B=A[n,n+1]
            for k=1:size(B,1)
                B[k,k]  -= (n-k+α+1)/(2n+α+β+γ+2)*(k+β+γ)/(2k+β+γ-1)
                if k+1 ≤ size(B,2)
                    B[k,k+1]+= (k+β)/(2k+β+γ+1)*(n+k+β+γ+1)/(2n+α+β+γ+2)
                end
            end
        end
    else
        error("Not implemented")
    end

    A
end

