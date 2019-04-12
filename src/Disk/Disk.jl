export ChebyshevDisk, ZernikeDisk

struct ZernikeDisk{V,T} <: Space{Disk{V},T}
    domain::Disk{V}
end

ZernikeDisk(d::Disk{V}) where V = ZernikeDisk{V,complex(eltype(V))}(d)
ZernikeDisk() = ZernikeDisk(Disk())

spacescompatible(::ZernikeDisk, ::ZernikeDisk) = true

@containsconstants ZernikeDisk

points(K::ZernikeDisk, n::Integer) =
    fromcanonical.(Ref(K), points(ChebyshevDisk(), n))

evaluate(cfs::AbstractVector, Z::ZernikeDisk, xy) = 
    Fun(Fun(Z, cfs), ChebyshevDisk())(xy)

struct ChebyshevDisk{V,T} <: Space{Disk{V},T}
    domain::Disk{V}
end

@containsconstants ChebyshevDisk

ChebyshevDisk(d::Disk{V}) where V = ChebyshevDisk{V,eltype(V)}(d)
ChebyshevDisk() = ChebyshevDisk(Disk())

rectspace(_) = Chebyshev() * Fourier()

spacescompatible(K1::ChebyshevDisk, K2::ChebyshevDisk) = domainscompatible(K1, K2)

function points(S::ChebyshevDisk, N)
    pts = points(rectspace(S), N)
    polar.(pts)
end

tensorizer(K::ChebyshevDisk) = Tensorizer((Ones{Int}(∞),Ones{Int}(∞)))

# we have each polynomial
blocklengths(K::ChebyshevDisk) = 1:∞

for OP in (:block,:blockstart,:blockstop)
    @eval begin
        $OP(s::ChebyshevDisk,M::Block) = $OP(tensorizer(s),M)
        $OP(s::ChebyshevDisk,M) = $OP(tensorizer(s),M)
    end
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
    e1,e2 = @view(A[1:2:end,4:4:end]),@view(A[1:2:end,5:4:end])
    o1,o2 = @view(A[2:2:end,2:4:end]),@view(A[2:2:end,3:4:end])
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
    A[1:2:end,1] .= C[:,1]
    A[1:2:end,4:4:end] .= C[:,4:4:end]
    A[1:2:end,5:4:end] .= C[:,5:4:end]
    A[2:2:end,2:4:end] .= C[:,2:4:end]
    A[2:2:end,3:4:end] .= C[:,3:4:end]
    A
end

torectcfs(S, v) = fromtensor(rectspace(S), icheckerboard(totensor(S, v)))
fromrectcfs(S, v) = fromtensor(S, checkerboard(totensor(rectspace(S),v)))

*(P::TransformPlan{<:Any,<:ChebyshevDisk}, v::AbstractArray) = fromrectcfs(P.space, P.plan*v)
*(P::ITransformPlan{<:Any,<:ChebyshevDisk}, v::AbstractArray) = P.plan*torectcfs(P.space, v)

evaluate(cfs::AbstractVector, S::ChebyshevDisk, x) = evaluate(torectcfs(S,cfs), rectspace(S), ipolar(x))


function _coefficients(disk2cxf, v̂::AbstractVector, ::ChebyshevDisk, ::ZernikeDisk)
    F = totensor(ChebyshevDisk(), v̂)
    F *= sqrt(convert(T,π))
    n = size(F,1)
    F̌ = disk2cxf \ pad(F,n,4n-3)
    trivec(F̌)
end

function _coefficients(disk2cxf, v::AbstractVector, ::ZernikeDisk,  ::ChebyshevDisk)
    F̌ = tridevec(v)
    F̌ /= sqrt(convert(T,π))
    n = size(F,1)
    F = disk2cxf \ pad(F̌,n,4n-3)
    fromtensor(ChebyshevDisk(), F)
end

function coefficients(cfs::AbstractVector, CD::ChebyshevDisk, ZD::ZernikeDisk)
    c = totensor(ChebyshevDisk(), cfs)            # TODO: wasteful
    n = size(c,1)
    _coefficients(CDisk2CxfPlan(n), cfs, CD, ZD)
end

function coefficients(cfs::AbstractVector, ZD::ZernikeDisk, CD::ChebyshevDisk)
    c = tridevec(cfs)            # TODO: wasteful
    n = size(c,1)
    _coefficients(CDisk2CxfPlan(n), cfs, CD, ZD)
end


struct FastZernikeDiskTransformPlan{DUF,CHEB}
    cxfplan::DUF
    disk2cxf::CHEB
end

function FastZernikeDiskTransformPlan(S::ZernikeDisk, v::AbstractVector{T}) where T
    # n = floor(Integer,sqrt(2length(v)) + 1/2)
    # v = Array{T}(undef, sum(1:n))
    cxfplan = plan_transform(ChebyshevDisk(),v)
    c = totensor(ChebyshevDisk(), cxfplan*v)            # TODO: wasteful
    n = size(c,1)
    FastZernikeDiskTransformPlan(cxfplan, CDisk2CxfPlan(n))
end

*(P::FastZernikeDiskTransformPlan, v::AbstractVector) = 
    _coefficients(P.disk2cxf, P.cxfplan*v, ChebyshevDisk(), ZernikeDisk())

plan_transform(K::ZernikeDisk, v::AbstractVector) = FastZernikeDiskTransformPlan(K, v)

struct FastZernikeDiskITransformPlan{DUF,CHEB}
    icxfplan::DUF
    disk2cxf::CHEB
end

function FastZernikeDiskITransformPlan(S::ZernikeDisk, v::AbstractVector{T}) where T
    # n = floor(Integer,sqrt(2length(v)) + 1/2)
    # v = Array{T}(undef, sum(1:n))
    FastZernikeDiskITransformPlan(plan_itransform(ChebyshevDisk(), v), CDisk2CxfPlan(n))
end

function *(P::FastZernikeDiskITransformPlan, v)
    # n = floor(Integer,sqrt(2length(v)) + 1/2)
    # v = pad(v, sum(1:n))
    F̌ = trinormalize!(tridevec(v))
    F = P.disk2cxf * F̌
    v̂ = trivec(transpose(F))
    P.icxfplan*v̂
end


plan_itransform(K::ZernikeDisk, v::AbstractVector) = FastZernikeDiskITransformPlan(K, v)
Base.sum(f::Fun{<:ZernikeDisk}) = Fun(f,ZernikeDisk()).coefficients[1]/π