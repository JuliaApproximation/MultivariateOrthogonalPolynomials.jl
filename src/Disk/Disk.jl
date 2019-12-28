export ChebyshevDisk, ZernikeDisk

struct ZernikeDisk{V,T} <: Space{Disk{V},T}
    domain::Disk{V}
end

ZernikeDisk(d::Disk{V}) where V = ZernikeDisk{V,complex(eltype(V))}(d)
ZernikeDisk() = ZernikeDisk(Disk())

spacescompatible(::ZernikeDisk, ::ZernikeDisk) = true

@containsconstants ZernikeDisk

function pointsize(::ZernikeDisk, n)
    N = (1 + isqrt(1+8n)) ÷ 4
    N,2N-1
end

# M = 2N-1
# N*M == 2N^2 -N == n
# N = (1 + sqrt(1 + 8n)/4)
function points(d::ZernikeDisk, n)
    a,b = rectspace(ZernikeDisk()).spaces
    M,N = pointsize(d, n)
    p_b = points(b,N)
    p_a = points(a,M)
    polar.(Vec.(p_a, p_b'))
end

function evaluate(cfs::AbstractVector, Z::ZernikeDisk, xy) 
    C = totensor(ZernikeDisk(), cfs)
    F = _coefficients(CDisk2CxfPlan(size(C,1)), C, ZernikeDisk(), ChebyshevDisk())
    ProductFun(icheckerboard(F), rectspace(ChebyshevDisk()))(ipolar(xy...)...)
end

Space(d::Disk) = ZernikeDisk(d)    

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

struct DiskTensorizer end

function fromtensor(::DiskTensorizer, A::AbstractMatrix{T}) where T
    N = size(A,1)
    M = 4(N-1)+1
    @assert size(A,2) == M
    B = PseudoBlockArray(A, Ones{Int}(N), [1; Fill(2,2*(N-1))])
    a = Vector{T}()
    for N = 1:blocksize(B,2), K=1:(N+1)÷2
        append!(a, vec(view(B,Block(K,N-2K+2))))
    end
    a
end

# N + 4 * sum(1:N-1)
# N + 4 * N (N-1)/2 == n
function totensor(::DiskTensorizer, a::AbstractVector{T}) where T
    n = length(a)
    N = round(Int,1/4*(1 + sqrt(1 + 8n)),RoundUp)
    M = 4*(N-1)+1
    Ã = zeros(eltype(a), N, M)
    B = PseudoBlockArray(Ã, Ones{Int}(N), [1; Fill(2,2*(N-1))])
    k = 1
    for N = 1:blocksize(B,2), K=1:(N+1)÷2
        V = view(B, Block(K,N-2K+2))
        for j = 1:length(V)
            V[j] = a[k]
            k += 1
            k > length(a) && return Ã
        end
    end
    Ã
end

tensorizer(K::ZernikeDisk) = DiskTensorizer()
tensorizer(K::ChebyshevDisk) = Tensorizer((Ones{Int}(∞),Ones{Int}(∞)))

# we have each polynomial
blocklengths(K::ChebyshevDisk) = Base.OneTo(∞)
blocklengths(K::ZernikeDisk) = Base.OneTo(∞)

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
    #4(N-1) == n-1
    N = (n-1)÷4+1
    4(N-1)+1 == n || (N += 1) # round up
    M = 4(N-1)+1
    C = zeros(T, N, M)
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


function _coefficients(disk2cxf, v̂::AbstractVector{T}, ::ChebyshevDisk, ::ZernikeDisk) where T
    F = totensor(ChebyshevDisk(), v̂)
    n = disk2cxf.n
    F̌ = disk2cxf \ pad(F,n,4n-3)
    fromtensor(ZernikeDisk(), F̌)
end

function _coefficients(disk2cxf, F̌::AbstractMatrix{T}, ::ZernikeDisk,  ::ChebyshevDisk) where T
    n = disk2cxf.n
    disk2cxf * pad(F̌,n,4n-3)
end

_coefficients(disk2cxf, v::AbstractVector{T}, ::ZernikeDisk,  ::ChebyshevDisk) where T =
    fromtensor(ChebyshevDisk(), _coefficients(disk2cxf, totensor(ZernikeDisk(), v), ZernikeDisk(), ChebyshevDisk()))

function coefficients(cfs::AbstractVector, CD::ChebyshevDisk, ZD::ZernikeDisk)
    c = totensor(ChebyshevDisk(), cfs)            # TODO: wasteful
    n = size(c,1)
    _coefficients(CDisk2CxfPlan(n), cfs, CD, ZD)
end

function coefficients(cfs::AbstractVector, ZD::ZernikeDisk, CD::ChebyshevDisk)
    c = totensor(ZernikeDisk(), cfs)            # TODO: wasteful
    n = size(c,1)
    _coefficients(CDisk2CxfPlan(n), cfs, ZD, CD)
end


struct ZernikeDiskTransformPlan{DUF,CHEB}
    cxfplan::DUF
    disk2cxf::CHEB
end

function ZernikeDiskTransformPlan(S::ZernikeDisk, V::AbstractMatrix{T}) where T
    N,M = size(V)
    @assert M == 2N-1
    D = plan_transform!(rectspace(ChebyshevDisk()), Array{T}(undef,N,M))
    N = (M-1)÷4+1
    4(N-1)+1 == M || (N += 1) # round up
    ZernikeDiskTransformPlan(D, CDisk2CxfPlan(N))
end

function *(P::ZernikeDiskTransformPlan, V::AbstractMatrix) 
    V = P.cxfplan*copy(V)
    C = checkerboard(V)
    fromtensor(ZernikeDisk(),  P.disk2cxf\C)[1:sum(1:size(V,1))]
end

plan_transform(K::ZernikeDisk, v::AbstractMatrix) = ZernikeDiskTransformPlan(K, v)

struct ZernikeDiskITransformPlan{DUF,CHEB}
    icxfplan::DUF
    disk2cxf::CHEB
end

function ZernikeDiskITransformPlan(S::ZernikeDisk, v::AbstractVector{T}) where T
    # n = floor(Integer,sqrt(2length(v)) + 1/2)
    # v = Array{T}(undef, sum(1:n))
    ZernikeDiskITransformPlan(plan_itransform(ChebyshevDisk(), v), CDisk2CxfPlan(n))
end

function *(P::ZernikeDiskITransformPlan, v)
    # n = floor(Integer,sqrt(2length(v)) + 1/2)
    # v = pad(v, sum(1:n))
    F̌ = trinormalize!(tridevec(v))
    F = P.disk2cxf * F̌
    v̂ = trivec(transpose(F))
    P.icxfplan*v̂
end


plan_itransform(K::ZernikeDisk, v::AbstractVector) = ZernikeDiskITransformPlan(K, v)
Base.sum(f::Fun{<:ZernikeDisk}) = Fun(f,ZernikeDisk()).coefficients[1]/π