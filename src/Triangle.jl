export Triangle


using ApproxFun
    import ApproxFun: BivariateDomain, RealBasis, Derivative, domain, ConcreteDerivative,
                rangespace, bandinds, blockbandinds,spacescompatible,addentries!,conversion_rule,maxspace_rule

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

    conversion_rule(K1::KoornwinderTriangle,K2::KoornwinderTriangle)=
        KoornwinderTriangle(min(K1.α,K2.α),min(K1.β,K2.β),min(K1.γ,K2.γ))
    maxspace_rule(K1::KoornwinderTriangle,K2::KoornwinderTriangle)=
        KoornwinderTriangle(max(K1.α,K2.α),max(K1.β,K2.β),max(K1.γ,K2.γ))


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

K=KoornwinderTriangle(0,0,0)

Dy=Derivative(K,(0,1))
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

