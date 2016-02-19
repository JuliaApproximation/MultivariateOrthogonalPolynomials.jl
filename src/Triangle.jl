export Triangle

# currently right triangle
immutable Triangle <: BivariateDomain{Float64}

end

#Triangle(::AnyDomain)=Triangle(NaN,(NaN,NaN))
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
end