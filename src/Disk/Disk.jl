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

# drop every other entry
function checkerboard(A::AbstractMatrix{T}) where T
    m,n = size(A)
    C = zeros(T, (m+1)÷2, n)
    z = @view(A[1:2:end,1])
    e1 = @view(A[1:2:end,4:4:end])
    e2 = @view(A[1:2:end,5:4:end])
    o1 = @view(A[2:2:end,2:4:end])
    o2 = @view(A[2:2:end,3:4:end])
    C[1:size(z,1),1] .= z
    C[1:size(e1,1),4:4:n] .= e1
    C[1:size(e2,1),5:4:n] .= e2
    C[1:size(o1,1),2:4:n] .= o1
    C[1:size(o2,1),3:4:n] .= o2
    C
end

function icheckerboard(C::AbstractMatrix{T}) where T
    m,n = size(C)
    A = zeros(T, 2*m, n)
    A[1:2:end,1:2:end] .= C[:,1:2:end]
    A[2:2:end,2:2:end] .= C[:,2:2:end]
    A
end

*(P::TransformPlan{<:Any,<:ChebyshevDisk}, v::AbstractArray) = checkerboard(P.plan*v)
*(P::ITransformPlan{<:Any,<:ChebyshevDisk}, v::AbstractArray) = P.plan*icheckerboard(v)

evaluate(cfs::AbstractVector, S::ChebyshevDisk, x) = evaluate(cfs, rectspace(S), ipolar(tocanonical(S.domain,x)))

