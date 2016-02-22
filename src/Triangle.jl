export Triangle


using ApproxFun
    import ApproxFun: BivariateDomain, RealBasis, Derivative, domain, ConcreteDerivative,
                rangespace, bandinds, blockbandinds,spacescompatible,addentries!,conversion_rule,maxspace_rule,
                ConcreteConversion, isapproxinteger, Conversion


## Triangle Def
# currently right trianglel
    immutable Triangle <: BivariateDomain{Float64} end

    #Triangle(::AnyDomain)=Triangle(NaN,(NaN,NaN))e
    #isambiguous(d::Triangle)=isnan(d.radius) && all(isnan,d.center)


    #canonical is rectangle [r,0]x[-π,π]
    # we assume radius and centre are zero for now
    # fromcanonical{T<:Real}(D::Disk{T},x,t)=D.radius*x*cos(t)+D.center[1],D.radius*x*sin(t)+D.center[2]
    # tocanonical{T<:Real}(D::Disk{T},x,y)=sqrt((x-D.center[1])^2+(y-D.center[2]^2))/D.radius,atan2(y-D.center[2],x-D.center[1])
    # checkpoints(d::Disk)=[fromcanonical(d,(.1,.2243));fromcanonical(d,(-.212423,-.3))]

    # function points(d::Disk,n,m,k)
    #     ptsx=0.5*(1-gaussjacobi(n,1.,0.)[1])
    #     ptst=points(PeriodicInterval(),m)
    #
    #     Float64[fromcanonical(d,x,t)[k] for x in ptsx, t in ptst]
    # end


    # ∂(d::Disk)=Circle(Complex(d.center...),d.radius)


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
    spacescompatible(K1::KoornwinderTriangle,K2::KoornwinderTriangle)=K1.α==K2.α && K1.β==K2.β && K1.γ==K2.γ


    rangespace(D::ConcreteDerivative{KoornwinderTriangle})=KoornwinderTriangle(D.space.α+D.order[1],D.space.β+D.order[2],D.space.γ+sum(D.order),domain(D))
    bandinds(D::ConcreteDerivative{KoornwinderTriangle})=0,sum(D.order)
    blockbandinds(D::ConcreteDerivative{KoornwinderTriangle})=0,sum(D.order)

    function addentries!(D::ConcreteDerivative{KoornwinderTriangle},A,kr::Range,::Colon)
        @assert D.order==(0,1)
        α,β,γ = K.α,K.β,K.γ
        for n=kr
            Mkj=A[n,n+1]

            for k=1:size(Mkj,1)
                Mkj[k,k+1]+=(1+k+β+γ)
            end
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
    elseif K1.γ+1>K2.γ
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



C=Conversion(KoornwinderTriangle(0,0,0),
             KoornwinderTriangle(1,1,1))
full(C[1:10,1:10])
bandinds(C)

P=C.op
n=10
A=ApproxFun.bazeros(P,n,:);kr=1:10

if length(kr)==0
    return A
end

st=step(kr)

krl=Array(Int,length(P.ops),2)

krl[1,1],krl[1,2]=kr[1],kr[end]

for m=1:length(P.ops)-1
    br=bandinds(P.ops[m])
    krl[m+1,1]=max(st-mod(kr[1],st),br[1] + krl[m,1])  # no negative
    krl[m+1,2]=br[end] + krl[m,2]
end

krl

# The following returns a banded Matrix with all rows
# for large k its upper triangular
BA=slice(P.ops[end],krl[end,1]:st:krl[end,2],:)
m=(length(P.ops)-1)
C=slice(P.ops[m],krl[m,1]:st:krl[m,2],:)

ApproxFun.blockbandzeros(

n,m=size(A,1),size(B,2)
ApproxFun.bamultiply!(ApproxFun.bazeros(Matrix{Float64},n,m,A.l+B.l,A.u+B.u),A,B)
A,B=C,BA
A*B
A
ApproxFun.BandedMatrix(C,10)

(C.op.ops[2][1:10,1:10],C.op.ops[3][1:10,1:10])



C=Conversion(KoornwinderTriangle(1,0,0),
             KoornwinderTriangle(1,1,1))*Conversion(KoornwinderTriangle(0,0,0),
             KoornwinderTriangle(1,0,0))
    C1=Matrix{Float64}(C.op.ops[1][1:10,1:10])
    C2=Matrix{Float64}(C.op.ops[2][1:10,1:10])
    C3=Matrix{Float64}(C.op.ops[3][1:10,1:10])
    CC=C1*C2*C3


C=Conversion(KoornwinderTriangle(0,1,0),
             KoornwinderTriangle(1,1,1))*Conversion(KoornwinderTriangle(0,0,0),
             KoornwinderTriangle(0,1,0))
    C1=Matrix{Float64}(C.op.ops[1][1:10,1:10])
    C2=Matrix{Float64}(C.op.ops[2][1:10,1:10])
    C3=Matrix{Float64}(C.op.ops[3][1:10,1:10])
    C1*C2*C3-CC|>norm


C=Conversion(KoornwinderTriangle(0,0,1),
             KoornwinderTriangle(1,1,1))*Conversion(KoornwinderTriangle(0,0,0),
             KoornwinderTriangle(0,0,1))
    C1=Matrix{Float64}(C.op.ops[1][1:10,1:10])
    C2=Matrix{Float64}(C.op.ops[2][1:10,1:10])
    C3=Matrix{Float64}(C.op.ops[3][1:10,1:10])
    C1*C2*C3-CC|>norm


C1


C1


C1*C2*C3

C1*C2*C3|>chopm

|>full
C[1:10,1:10]|>full

K=KoornwinderTriangle(0,0,0)
ApproxFun.op_eltype(K)
@which Derivative(K,(0,1))

Dy[1:10,1:10]

Dx=Derivative(K,(1,0))
maxspace(rangespace(Dx),
    rangespace(Dy))


L=Dx+Dy

L[1,1]
domainspace(Dy),rangespace(Dy)
full(Dy[1:10,1:10])

Dx=Derivative(Chebyshev()^2,(1,0))

full(Dx[1:10,1:10])

ApproxFun.op_eltype(K)
using ApproxFun
    import ApproxFun: DerivativeWrapper

Derivative(Chebyshev()^2,(1,0))|>blockbandinds
S,order=Chebyshev()^2,(1,0)
Dx=

L=lap(S)
L[1,1]

typeof(Dx)|>super  |>super         |>super

ApproxFun.blockbandinds(Dx)
Dx[1,2]

L=lap(S)s

eltype(L)

typeof(D)|>super|>super|>super
ApproxFun.bazeros(ApproxFun.BandedMatrix{Float64},5,5,1,2)
Dx2=Derivative(S,(2,0))
Dy2=Derivative(S,(0,2))
full(Dx[1:3,1:3])
full((Dx2+Dy2)[1:5,1:5])

Dx^2

