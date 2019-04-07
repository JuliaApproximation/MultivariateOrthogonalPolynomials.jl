using ApproxFun
f = (x,y) -> x*y+cos(y-0.1)+sin(x)+1; ff = Fun((r,θ) -> f(r*cos(θ),r*sin(θ)), (-1..1) × PeriodicSegment());
ApproxFunBase.coefficientmatrix(ff)

struct ChebyshevDisk{V,T} <: Space{Disk{V},T}
    domain::Disk{V}
end

ChebyshevDisk(d::Disk{V}) where V = ChebyshevDisk{V,eltype(V)}(d)
ChebyshevDisk() = ChebyshevDisk(Disk())

rectspace(_) = Chebyshev() * Laurent()


function points(S::ChebyshevDisk, N)
    pts = points(rectspace(S), N)
    fromcanonical.(Ref(S.domain), polar.(pts))
end

plan_transform(S::DuffyTriangle, n::AbstractVector) = 
    TransformPlan(S, plan_transform(rectspace(S),n), Val{false})
plan_itransform(S::DuffyTriangle, n::AbstractVector) = 
    ITransformPlan(S, plan_itransform(rectspace(S),n), Val{false})


*(P::TransformPlan{<:Any,<:DuffyTriangle}, v::AbstractArray) = checkerboard(P.plan*v)
*(P::ITransformPlan{<:Any,<:DuffyTriangle}, v::AbstractArray) = P.plan*icheckerboard(v)

evaluate(cfs::AbstractVector, S::DuffyTriangle, x) = evaluate(cfs, rectspace(S), ipolar(tocanonical(S.domain,x)))

