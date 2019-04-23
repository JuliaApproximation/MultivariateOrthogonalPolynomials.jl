export Cone, DuffyCone, LegendreCone

struct Cone <: Domain{Vec{3,Float64}} end
struct DuffyCone <: Space{Cone,Float64} end
struct LegendreCone <: Space{Cone,Float64} end

@containsconstants DuffyCone
@containsconstants LegendreCone

spacescompatible(::DuffyCone, ::DuffyCone) = true
spacescompatible(::LegendreCone, ::LegendreCone) = true

domain(::DuffyCone) = Cone()
domain(::LegendreCone) = Cone()

tensorizer(K::DuffyCone) = DiskTensorizer()
tensorizer(K::LegendreCone) = DiskTensorizer()

rectspace(::DuffyCone) = NormalizedJacobi(0,1,Segment(1,0))*Fourier()

conepolar(t,θ) = Vec(t, t*cos(θ), t*sin(θ))
conepolar(tθ) = ((t,θ) = tθ; conepolar(t,θ))

points(::Cone, n) = conepolar.(points(rectspace(DuffyCone()), n))
checkpoints(::Cone) = conepolar.(checkpoints(rectspace(DuffyCone())))


plan_transform(S::DuffyCone, n::AbstractVector) = 
    TransformPlan(S, plan_transform(rectspace(S),n), Val{false})
plan_itransform(S::DuffyCone, n::AbstractVector) = 
    ITransformPlan(S, plan_itransform(rectspace(S),n), Val{false})

*(P::TransformPlan{<:Any,<:DuffyCone}, v::AbstractArray) = P.plan*v
*(P::ITransformPlan{<:Any,<:DuffyCone}, v::AbstractArray) = P.plan*v

evaluate(cfs::AbstractVector, S::DuffyCone, txy) = evaluate(cfs, rectspace(S), ipolar(txy[Vec(2,3)]))
evaluate(cfs::AbstractVector, S::LegendreCone, txy) = evaluate(coefficients(cfs, S, DuffyCone()), DuffyCone(), txy)


# function transform(::DuffyCone, v)
#     n = (1 + isqrt(1+8*length(v))) ÷ 4
#     F = copy(reshape(v,n,2n-1))
#     Pt = plan_transform(NormalizedJacobi(0,1,Segment(1,0)), n)
#     Pθ = plan_transform(Fourier(), 2n-1)
#     for j = 1:size(F,2)
#         F[:,j] = Pt*F[:,j]
#     end
#     for k = 1:size(F,1)
#         F[k,:] = Pθ * F[k,:]
#     end
#     fromtensor(rectspace(DuffyCone()), F)
# end

function evaluate(cfs::AbstractVector, S::DuffyCone, txy::Vec{3})
    t,x,y = txy
    @assert t ≈ sqrt(x^2+y^2)
    θ = atan(y,x)
    Fun(rectspace(S), cfs)(t,θ)
end


function _coefficients(triangleplan, v::AbstractVector{T}, ::DuffyCone, ::LegendreCone) where T
    F = totensor(rectspace(DuffyCone()), v)
    F = pad(F, :, 2size(F,1)-1)
    Fc = F[:,1:2:end]
    c_execute_tri_lo2hi(triangleplan, Fc)
    F[:,1:2:end] .= Fc
    # ignore first column
    Fc[:,2:end] .= F[:,2:2:end]
    c_execute_tri_lo2hi(triangleplan, Fc)
    F[:,2:2:end] .= Fc[:,2:end]
    fromtensor(LegendreCone(), F)
end

function _coefficients(triangleplan, v::AbstractVector{T}, ::LegendreCone,  ::DuffyCone) where T
    F = totensor(LegendreCone(), v)
    F = pad(F, :, 2size(F,1)-1)
    Fc = F[:,1:2:end]
    c_execute_tri_hi2lo(triangleplan, Fc)
    F[:,1:2:end] .= Fc
    # ignore first column
    Fc[:,2:end] .= F[:,2:2:end]
    c_execute_tri_hi2lo(triangleplan, Fc)
    F[:,2:2:end] .= Fc[:,2:end]
    fromtensor(DuffyCone(), F)
end

function coefficients(cfs::AbstractVector{T}, CD::DuffyCone, ZD::LegendreCone) where T
    c = totensor(rectspace(DuffyCone()), cfs)            # TODO: wasteful
    n = size(c,1)
    _coefficients(c_plan_rottriangle(n, zero(T), zero(T), zero(T)), cfs, CD, ZD)
end

function coefficients(cfs::AbstractVector{T}, ZD::LegendreCone, CD::DuffyCone) where T
    c = tridevec(cfs)            # TODO: wasteful
    n = size(c,1)
    _coefficients(c_plan_rottriangle(n, zero(T), zero(T), zero(T)), cfs, ZD, CD)
end

struct LegendreConeTransformPlan{DUF,CHEB}
    duffyplan::DUF
    triangleplan::CHEB
end

function LegendreConeTransformPlan(S::LegendreCone, v::AbstractVector{T}) where T 
    D = plan_transform(DuffyCone(),v)
    F = totensor(rectspace(DuffyCone()), D*v) # wasteful, just use to figure out `size(F,1)`
    P = c_plan_rottriangle(size(F,1), zero(T), zero(T), zero(T))
    LegendreConeTransformPlan(D,P)
end


*(P::LegendreConeTransformPlan, v::AbstractVector) = _coefficients(P.triangleplan, P.duffyplan*v, DuffyCone(), LegendreCone())

plan_transform(K::LegendreCone, v::AbstractVector) = LegendreConeTransformPlan(K, v)

struct LegendreConeITransformPlan{DUF,CHEB}
    iduffyplan::DUF
    triangleplan::CHEB
end

function LegendreConeITransformPlan(S::LegendreCone, v::AbstractVector{T}) where T 
    F = fromtensor(LegendreCone(), v) # wasteful, just use to figure out `size(F,1)`
    P = c_plan_rottriangle(size(F,1), zero(T), zero(T), zero(T))
    D = plan_itransform(DuffyCone(), _coefficients(P, v, S, DuffyCone()))
    LegendreConeITransformPlan(D,P)
end


*(P::LegendreConeITransformPlan, v::AbstractVector) = P.iduffyplan*_coefficients(P.triangleplan, v, LegendreCone(), DuffyCone())


plan_itransform(K::LegendreCone, v::AbstractVector) = LegendreConeITransformPlan(K, v)

