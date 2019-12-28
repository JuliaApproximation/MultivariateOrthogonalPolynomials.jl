export Cone, DuffyCone, LegendreCone, Conic, DuffyConic, LegendreConic

### 
#  Conic
###

struct Conic <: Domain{Vec{3,Float64}} end
struct DuffyConic <: Space{Conic,Float64} end
struct LegendreConic <: Space{Conic,Float64} end

@containsconstants DuffyConic
@containsconstants LegendreConic

spacescompatible(::DuffyConic, ::DuffyConic) = true
spacescompatible(::LegendreConic, ::LegendreConic) = true

domain(::DuffyConic) = Conic()
domain(::LegendreConic) = Conic()


# groups by Fourier
struct ConicTensorizer end



tensorizer(K::DuffyConic) = DiskTensorizer()
tensorizer(K::LegendreConic) = ConicTensorizer()

function isqrt_roundup(n)
    N = isqrt(n)
    N^2 == n ? N : N+1
end

function fromtensor(::ConicTensorizer, A::AbstractMatrix{T}) where T
    N = size(A,1)
    M = 2N-1
    @assert size(A,2) == M
    B = PseudoBlockArray(A, Ones{Int}(N), [1; Fill(2,N-1)])
    a = Vector{T}()
    for N = 1:blocksize(B,2), K=1:N
        append!(a, vec(view(B,Block(K,N-K+1))))
    end
    a
end

function totensor(::ConicTensorizer, a::AbstractVector{T}) where T
    N = isqrt_roundup(length(a))
    M = 2N-1
    Ã = zeros(eltype(a), N, M)
    B = PseudoBlockArray(Ã, Ones{Int}(N), [1; Fill(2,N-1)])
    k = 1
    for N = 1:blocksize(B,2), K=1:N
        V = view(B, Block(K,N-K+1))
        for j = 1:length(V)
            V[j] = a[k]
            k += 1
            k > length(a) && return Ã
        end
    end
    Ã
end

rectspace(::DuffyConic) = NormalizedJacobi(0,1,Segment(1,0))*Fourier()

conemap(t,x,y) = Vec(t, t*x, t*y)
conemap(txy) = ((t,x,y) = txy; conemap(t,x,y))
conicpolar(t,θ) = conemap(t, cos(θ), sin(θ))
conicpolar(tθ) = ((t,θ) = tθ; conicpolar(t,θ))
conepolar(t,r,θ) = conemap(t, r*cos(θ), r*sin(θ))
conepolar(v::Vec) = conepolar(v...)


# M = 2N-1
# N*M == 2N^2 -N == n
# N = (1 + sqrt(1 + 8n)/4)
function pointsize(::Conic, n) 
    N = (1 + isqrt(1+8n)) ÷ 4
    N,2N-1
end

pointsize(s::Space, n) = pointsize(domain(s), n)

function points(d::Conic, n)
    a,b = rectspace(DuffyConic()).spaces
    pts=Array{float(eltype(d))}(undef,0)
    N,M = pointsize(d,n)

    for y in points(b,M),
        x in points(a,N)
        push!(pts,conicpolar(Vec(x...,y...)))
    end
    pts
end
checkpoints(::Conic) = conicpolar.(checkpoints(rectspace(DuffyConic())))


function plan_transform(S::DuffyConic, v::AbstractVector{T}) where T
    n = length(v)
    N,M = pointsize(S,length(v))
    D = plan_transform!(rectspace(S), Array{T}(undef,N,M))
    TransformPlan(S, D, Val{false})
end
plan_itransform(S::DuffyConic, n::AbstractVector) = 
    ITransformPlan(S, plan_itransform(rectspace(S),n), Val{false})

*(P::TransformPlan{<:Any,<:DuffyConic}, v::AbstractArray) = P.plan*v
*(P::ITransformPlan{<:Any,<:DuffyConic}, v::AbstractArray) = P.plan*v

evaluate(cfs::AbstractVector, S::DuffyConic, txy) = evaluate(cfs, rectspace(S), ipolar(txy[Vec(2,3)]))
evaluate(cfs::AbstractVector, S::LegendreConic, txy) = evaluate(coefficients(cfs, S, DuffyConic()), DuffyConic(), txy)


# function transform(::DuffyConic, v)
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
#     fromtensor(rectspace(DuffyConic()), F)
# end

function evaluate(cfs::AbstractVector, S::DuffyConic, txy::Vec{3})
    t,x,y = txy
    @assert t ≈ sqrt(x^2+y^2)
    θ = atan(y,x)
    Fun(rectspace(S), cfs)(t,θ)
end

function duffy2legendreconic!(triangleplan, F::AbstractMatrix)
    Fc = F[:,1:2:end]
    c_execute_tri_lo2hi(triangleplan, Fc)
    F[:,1:2:end] .= Fc
    # ignore first column
    Fc[:,2:end] .= F[:,2:2:end]
    c_execute_tri_lo2hi(triangleplan, Fc)
    F[:,2:2:end] .= Fc[:,2:end]
    F
end

function legendre2duffyconic!(triangleplan, F::AbstractMatrix)
    Fc = F[:,1:2:end]
    c_execute_tri_hi2lo(triangleplan, Fc)
    F[:,1:2:end] .= Fc
    # ignore first column
    Fc[:,2:end] .= F[:,2:2:end]
    c_execute_tri_hi2lo(triangleplan, Fc)
    F[:,2:2:end] .= Fc[:,2:end]
    F
end


function _coefficients(triangleplan, v::AbstractVector{T}, ::DuffyConic, ::LegendreConic) where T
    F = totensor(rectspace(DuffyConic()), v)
    F = pad(F, :, 2size(F,1)-1)
    duffy2legendreconic!(triangleplan, F)
    fromtensor(LegendreConic(), F)
end

function _coefficients(triangleplan, v::AbstractVector{T}, ::LegendreConic, ::DuffyConic) where T
    F = totensor(LegendreConic(), v)
    F = pad(F, :, 2size(F,1)-1)
    legendre2duffyconic!(triangleplan, F)
    fromtensor(rectspace(DuffyConic()), F)
end

function coefficients(cfs::AbstractVector{T}, CD::DuffyConic, ZD::LegendreConic) where T
    c = totensor(rectspace(DuffyConic()), cfs)            # TODO: wasteful
    n = size(c,1)
    _coefficients(c_plan_rottriangle(n, zero(T), zero(T), zero(T)), cfs, CD, ZD)
end

function coefficients(cfs::AbstractVector{T}, ZD::LegendreConic, CD::DuffyConic) where T
    n = isqrt_roundup(length(cfs))
    _coefficients(c_plan_rottriangle(n, zero(T), zero(T), zero(T)), cfs, ZD, CD)
end

struct LegendreConicTransformPlan{DUF,CHEB}
    duffyplan::DUF
    triangleplan::CHEB
end

function LegendreConicTransformPlan(S::LegendreConic, v::AbstractVector{T}) where T 
    n = length(v)
    N,M = pointsize(Conic(), n)
    D = plan_transform!(rectspace(DuffyConic()), Array{T}(undef,N,M))
    P = c_plan_rottriangle(N, zero(T), zero(T), zero(T))
    LegendreConicTransformPlan(D,P)
end


function *(P::LegendreConicTransformPlan, v::AbstractVector) 
    N,M = P.duffyplan.plan[1][2],P.duffyplan.plan[2][2] 
    V=reshape(v,N,M)
    fromtensor(LegendreConic(),
               duffy2legendreconic!(P.triangleplan,P.duffyplan*copy(V)))
end

plan_transform(K::LegendreConic, v::AbstractVector) = LegendreConicTransformPlan(K, v)

struct LegendreConicITransformPlan{DUF,CHEB}
    iduffyplan::DUF
    triangleplan::CHEB
end

function LegendreConicITransformPlan(S::LegendreConic, v::AbstractVector{T}) where T 
    F = fromtensor(LegendreConic(), v) # wasteful, just use to figure out `size(F,1)`
    P = c_plan_rottriangle(size(F,1), zero(T), zero(T), zero(T))
    D = plan_itransform(DuffyConic(), _coefficients(P, v, S, DuffyConic()))
    LegendreConicITransformPlan(D,P)
end


*(P::LegendreConicITransformPlan, v::AbstractVector) = P.iduffyplan*_coefficients(P.triangleplan, v, LegendreConic(), DuffyConic())


plan_itransform(K::LegendreConic, v::AbstractVector) = LegendreConicITransformPlan(K, v)



### 
#  Cone
###

struct Cone <: Domain{Vec{3,Float64}} end

in(x::Vec{3}, d::Cone) = 0 ≤ x[1] ≤ 1 && x[2]^2 + x[3]^2 ≤ x[1]^2 

struct DuffyCone <: Space{Cone,Float64} end
struct LegendreCone <: Space{Cone,Float64} end

@containsconstants DuffyCone
@containsconstants LegendreCone

spacescompatible(::DuffyCone, ::DuffyCone) = true
spacescompatible(::LegendreCone, ::LegendreCone) = true

domain(::DuffyCone) = Cone()
domain(::LegendreCone) = Cone()

ConeTensorizer() = Tensorizer((Ones{Int}(∞),Base.OneTo(∞)))

tensorizer(K::DuffyCone) = ConeTensorizer()
tensorizer(K::LegendreCone) = ConeTensorizer()

rectspace(::DuffyCone) = NormalizedJacobi(0,1,Segment(1,0))*ZernikeDisk()

blocklengths(d::DuffyCone) = blocklengths(rectspace(d))

# M = N*(2N-1)
# N*N*(2N-1) == N^2*(2N-1) == n
# N =1/6*(1 + 1/(1 + 54n + 6sqrt(3)sqrt(n + 27n^2))^(1/3) + (1 + 54n + 6sqrt(3)sqrt(n + 27n^2))^(1/3))
function pointsize(::Cone, n) 
    N = round(Int, 1/6*(1 + 1/(1 + 54n + 6sqrt(3)sqrt(n + 27n^2))^(1/3) + (1 + 54n + 6sqrt(3)sqrt(n + 27n^2))^(1/3)), RoundUp)
    N, N, 2N-1
end

function points(d::Cone, n)
    a,_ = rectspace(DuffyCone()).spaces
    b,c = rectspace(ZernikeDisk()).spaces

    M,_,N = pointsize(d, n)
    p_a = points(a,M)
    p_b = points(b,M)
    p_c = points(c,N)
    
    conepolar.(Vec.(p_a,reshape(p_b,1,M),reshape(p_c,1,1,N)))
end



# function points(d::Cone,n)
#     N,M = pointsize(d,n)
#     pts = Array{ApproxFunBase.float(eltype(d))}(undef,0)
#     a,b = rectspace(DuffyCone()).spaces
#     for y in points(b,M), x in points(a,N)
#         push!(pts,conemap(Vec(x...,y...)))
#     end
#     pts
# end


checkpoints(::Cone) = conemap.(checkpoints(rectspace(DuffyCone())))


function plan_transform(sp::DuffyCone, V::AbstractArray{T,3})  where T
    M,M̃,N = size(V)
    @assert M == M̃
    a,b = rectspace(sp).spaces
    TransformPlan(sp, (plan_transform(a, M), plan_transform(b, Array{T}(undef,M,N))), Val{false})
end

plan_itransform(S::DuffyCone, n::AbstractVector) = 
    ITransformPlan(S, plan_itransform(rectspace(S),n), Val{false})

function tensortransform(P::TransformPlan{<:Any,<:DuffyCone,false}, V::AbstractArray{<:Any,3})
    M,_,N = size(V)
    R = Array{Float64}(undef,M,sum(1:M))
    for k = 1:M
        R[k,:] = P.plan[2]*V[k,:,:]
    end
    for j = 1:size(R,2)
        R[:,j] = P.plan[1]*R[:,j]
    end
    R
end

*(P::TransformPlan{<:Any,<:DuffyCone,false}, V::AbstractArray{<:Any,3}) =
    fromtensor(P.space,tensortransform(P, V))

*(P::ITransformPlan{<:Any,<:DuffyCone}, v::AbstractArray) = P.plan*v

# function evaluate(cfs::AbstractVector, S::DuffyCone, txy) 
#     t,x,y = txy
#     evaluate(cfs, rectspace(S), Vec(t, 
# end
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
    @assert x^2+y^2 ≤ t^2
    a,b = rectspace(S).spaces
    mat = totensor(S, cfs)
    for j = 1:size(mat,2)
        # in place since totensor makes copy
        mat[1,j] = Fun(a, view(mat,:,j))(t)
    end
    Fun(b, view(mat,1,:))(x/t,y/t)
end

function duffy2legendrecone_column_J!(P, Fc, F, Jin)
    J = sum(1:Jin)
    for j = Jin:size(Fc,2)
        if 1 ≤ J ≤ size(F,2)
            Fc[:,j] .= view(F,:,J)
        else
            Fc[:,j] .= 0
        end
        J += j
    end
    c_execute_tri_lo2hi(P, Fc)
    J = sum(1:Jin)
    for j = Jin:size(Fc,2)
        J > size(F,2) && break
        F[:,J] .= view(Fc,:,j)
        J += j
    end
    F
end

function legendre2duffycone_column_J!(P, Fc, F, Jin)
    J = sum(1:Jin)
    for j = Jin:size(Fc,2)
        if 1 ≤ J ≤ size(F,2)
            Fc[:,j] .= view(F,:,J)
        else
            Fc[:,j] .= 0
        end
        J += j
    end
    c_execute_tri_hi2lo(P, Fc)
    J = sum(1:Jin)
    for j = Jin:size(Fc,2)
        J > size(F,2) && break
        F[:,J] .= view(Fc,:,j)
        J += j
    end
    F
end

function duffy2legendrecone!(triangleplan, F::AbstractMatrix)
    Fc = Matrix{eltype(F)}(undef,size(F,1),size(F,1))
    for J = 1:(isqrt(1 + 8size(F,2))-1)÷ 2 # actually div 
        duffy2legendrecone_column_J!(triangleplan, Fc, F, J)
    end
    F
end

function legendre2duffycone!(triangleplan, F::AbstractMatrix)
    Fc = Matrix{eltype(F)}(undef,size(F,1),size(F,1))
    for J = 1:(isqrt(1 + 8size(F,2))-1)÷ 2 # actually div 
        legendre2duffycone_column_J!(triangleplan, Fc, F, J)
    end
    F
end

function _coefficients!(triangleplan, F::AbstractMatrix{T}, ::DuffyCone, ::LegendreCone) where T
    for j = 1:size(F,2)
        F[:,j] = coefficients(F[:,j], NormalizedJacobi(0,1,Segment(1,0)), NormalizedJacobi(0,2,Segment(1,0)))
    end
    duffy2legendrecone!(triangleplan, F)
end

function _coefficients(triangleplan, v::AbstractVector{T}, D::DuffyCone, L::LegendreCone) where T
    F = totensor(rectspace(DuffyCone()), v)
    fromtensor(LegendreCone(), _coefficients!(triangleplan, F, D, L))
end

function _coefficients(triangleplan, v::AbstractVector{T}, ::LegendreCone, ::DuffyCone) where T
    F = totensor(LegendreCone(), v)
    legendre2duffycone!(triangleplan, F)
    for j = 1:size(F,2)
        F[:,j] = coefficients(F[:,j], NormalizedJacobi(0,2,Segment(1,0)), NormalizedJacobi(0,1,Segment(1,0)))
    end
    fromtensor(rectspace(DuffyCone()), F)
end

function coefficients(cfs::AbstractVector{T}, CD::DuffyCone, ZD::LegendreCone) where T
    c = totensor(rectspace(DuffyCone()), cfs)            # TODO: wasteful
    n = size(c,1)
    _coefficients(c_plan_rottriangle(n, zero(T), zero(T), one(T)), cfs, CD, ZD)
end

function coefficients(cfs::AbstractVector{T}, ZD::LegendreCone, CD::DuffyCone) where T
    c = totensor(LegendreCone(), cfs)            # TODO: wasteful
    n = size(c,1)
    _coefficients(c_plan_rottriangle(n, zero(T), zero(T), one(T)), cfs, ZD, CD)
end

struct LegendreConeTransformPlan{DUF,CHEB}
    duffyplan::DUF
    triangleplan::CHEB
end

function LegendreConeTransformPlan(S::LegendreCone, V::AbstractArray{T,3}) where T 
    M,M̃,N = size(V)
    @assert M̃ == M
    D = plan_transform(DuffyCone(), V)
    P = c_plan_rottriangle(M, zero(T), zero(T), one(T))
    LegendreConeTransformPlan(D,P)
end


function tensortransform(P::LegendreConeTransformPlan, V::AbstractArray{<:Any,3}) 
    C = tensortransform(P.duffyplan,V)
    _coefficients!(P.triangleplan,C, DuffyCone(), LegendreCone())
end

*(P::LegendreConeTransformPlan, V::AbstractArray{<:Any,3}) =
    fromtensor(LegendreCone(), tensortransform(P,V))

plan_transform(K::LegendreCone, V::AbstractArray{<:Any,3}) = LegendreConeTransformPlan(K, V)

struct LegendreConeITransformPlan{DUF,CHEB}
    iduffyplan::DUF
    triangleplan::CHEB
end

function LegendreConeITransformPlan(S::LegendreCone, v::AbstractVector{T}) where T 
    F = fromtensor(LegendreCone(), v) # wasteful, just use to figure out `size(F,1)`
    P = c_plan_rottriangle(size(F,1), zero(T), zero(T), one(T))
    D = plan_itransform(DuffyCone(), _coefficients(P, v, S, DuffyCone()))
    LegendreConeITransformPlan(D,P)
end


*(P::LegendreConeITransformPlan, v::AbstractVector) = P.iduffyplan*_coefficients(P.triangleplan, v, LegendreCone(), DuffyCone())


plan_itransform(K::LegendreCone, v::AbstractVector) = LegendreConeITransformPlan(K, v)

