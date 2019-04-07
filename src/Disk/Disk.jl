export ChebyshevDisk

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

plan_transform(S::ChebyshevDisk, n::AbstractVector) = 
    TransformPlan(S, plan_transform(rectspace(S),n), Val{false})
plan_itransform(S::ChebyshevDisk, n::AbstractVector) = 
    ITransformPlan(S, plan_itransform(rectspace(S),n), Val{false})


*(P::TransformPlan{<:Any,<:ChebyshevDisk}, v::AbstractArray) = checkerboard(P.plan*v)
*(P::ITransformPlan{<:Any,<:ChebyshevDisk}, v::AbstractArray) = P.plan*icheckerboard(v)

evaluate(cfs::AbstractVector, S::ChebyshevDisk, x) = evaluate(cfs, rectspace(S), ipolar(tocanonical(S.domain,x)))

