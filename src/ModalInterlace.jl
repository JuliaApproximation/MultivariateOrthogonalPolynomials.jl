"""
    ModalTrav(A::AbstractMatrix)

    takes coefficients as provided by the Zernike polynomial layout of FastTransforms.jl and
    makes them accessible sorted such that in each block the m=0 entries are always in first place,
    followed by alternating sin and cos terms of increasing |m|.
"""
struct ModalTrav{T, AA<:AbstractMatrix{T}} <: AbstractBlockVector{T}
    matrix::AA
    function ModalTrav{T, AA}(matrix::AA) where {T,AA<:AbstractMatrix{T}}
        m,n = size(matrix)
        if isfinite(m)
            isfinite(n) && isodd(n) && m == n ÷ 4 + 1 || throw(ArgumentError("size must match"))
        end
        new{T,AA}(matrix)
    end
end

ModalTrav{T}(matrix::AbstractMatrix{T}) where T = ModalTrav{T,typeof(matrix)}(matrix)
ModalTrav(matrix::AbstractMatrix{T}) where T = ModalTrav{T}(matrix)

function ModalTrav(v::AbstractVector{T}) where T
    N = blocksize(v,1)
    m = N ÷ 2 + 1
    n = 4(m-1) + 1
    mat = zeros(T, m, n)
    for K in blockaxes(v,1)
        K̃ = Int(K)
        w = v[K]
        if isodd(K̃)
            mat[K̃÷2 + 1,1] = w[1]
            for j = 2:2:K̃-1
                mat[K̃÷2-j÷2+1,2(j-1)+2] = w[j]
                mat[K̃÷2-j÷2+1,2(j-1)+3] = w[j+1]
            end
        else
            for j = 1:2:K̃
                mat[K̃÷2-j÷2,2(j-1)+2] = w[j]
                mat[K̃÷2-j÷2,2(j-1)+3] = w[j+1]
            end
        end
    end
    ModalTrav(mat)
end

copy(A::ModalTrav) = ModalTrav(copy(A.matrix))

_diviffinite(n) = div(n,2,RoundUp)
_diviffinite(n::InfiniteCardinal) = n

axes(A::ModalTrav) = (blockedrange(oneto(_diviffinite(size(A.matrix,2)))),)

getindex(A::ModalTrav, K::Block{1}) = _modaltravgetindex(A.matrix, K)

_modaltravgetindex(mat, K) = _modaltravgetindex(MemoryLayout(mat), mat, K)
function _modaltravgetindex(_, mat, K::Block{1})
    k = Int(K)
    m = (k+1)÷2
    _modaltravgetindex(Matrix(mat[1:m, 1:4m+1]), K)
end

function _modaltravgetindex(::AbstractStridedLayout, mat, K::Block{1})
    k = Int(K)
    k == 1 && return mat[1:1]
    k == 2 && return mat[1,2:3]
    st = stride(mat,2)
    if isodd(k)
        # nonnegative terms
        p = mat[range(k÷2+1, step=4*st-1, length=k÷2+1)]
        # negative terms
        n = mat[range(k÷2+3*st, step=4*st-1, length=k÷2)]
        interlace(p,n)
    else
        # negative terms
        n = mat[range(st+k÷2, step=4*st-1, length=k÷2)]
        # positive terms
        p = mat[range(2st+k÷2, step=4*st-1, length=k÷2)]
        interlace(n,p)
    end
end

getindex(A::ModalTrav, k::Int) = A[findblockindex(axes(A,1), k)]


similar(A::ModalTrav, ::Type{T}) where T = ModalTrav(similar(A.matrix, T))
function fill!(A::ModalTrav, x)
    fill!(A.matrix, x)
    A
end

struct ModalTravStyle <: AbstractBlockStyle{1} end

ModalTravStyle(::Val{1}) = ModalTravStyle()

BroadcastStyle(::Type{<:ModalTrav}) = ModalTravStyle()

function similar(bc::Broadcasted{ModalTravStyle}, ::Type{T}) where T
    N = blocklength(axes(bc,1))
    n = 2N-1
    m = n ÷ 4 + 1
    ModalTrav(Matrix{T}(undef,m,n))
end

_modal2matrix(a::ModalTrav) = a.matrix
_modal2matrix(a::Broadcasted) = broadcasted(a.f, map(_modal2matrix, a.args)...)
_modal2matrix(a) = a

function copyto!(dest::ModalTrav, bc::Broadcasted{ModalTravStyle})
    broadcast!(bc.f, dest.matrix, map(_modal2matrix, bc.args)...)
    dest
end



"""
    ModalInterlace(ops, (M,N), (l,u))

interlaces the entries of a vector of banded matrices
acting on the different Fourier modes. That is, a ModalInterlace
multiplying a DiagTrav is the same as the operators multiplying the matrix
that the DiagTrav wraps. We assume the same operator acts on the Sin and Cos.
"""
struct ModalInterlace{T, MMNN<:Tuple} <: AbstractBandedBlockBandedMatrix{T}
    ops
    MN::MMNN
    bandwidths::NTuple{2,Int}
end

ModalInterlace{T}(ops, MN::NTuple{2,Integer}, bandwidths::NTuple{2,Int}) where T = ModalInterlace{T,typeof(MN)}(ops, MN, bandwidths)
ModalInterlace(ops::AbstractVector{<:AbstractMatrix}, MN::NTuple{2,Integer}, bandwidths::NTuple{2,Int}) = ModalInterlace{eltype(eltype(ops))}(ops, MN, bandwidths)

axes(Z::ModalInterlace) = blockedrange.(oneto.(Z.MN))

blockbandwidths(R::ModalInterlace) = R.bandwidths
subblockbandwidths(::ModalInterlace) = (0,0)
copy(M::ModalInterlace) = M


function Base.view(R::ModalInterlace{T}, KJ::Block{2}) where T
    K,J = KJ.n
    dat = Matrix{T}(undef,1,J)
    l,u = blockbandwidths(R)
    if iseven(J-K) && -l ≤ J - K ≤ u
        sh = (J-K)÷2
        if isodd(K)
            k = K÷2+1
            dat[1,1] = R.ops[1][k,k+sh]
        end
        for m in range(2-iseven(K); step=2, length=J÷2-max(0,sh))
            k = K÷2-m÷2+isodd(K)
            dat[1,m] = dat[1,m+1] = R.ops[m+1][k,k+sh]
        end
    else
        fill!(dat, zero(T))
    end
    _BandedMatrix(dat, K, 0, 0)
end

getindex(R::ModalInterlace, k::Integer, j::Integer) = R[findblockindex.(axes(R),(k,j))...]

struct ModalInterlaceLayout <: AbstractBandedBlockBandedLayout end
struct LazyModalInterlaceLayout <: AbstractLazyBandedBlockBandedLayout end

MemoryLayout(::Type{<:ModalInterlace}) = ModalInterlaceLayout()
MemoryLayout(::Type{<:ModalInterlace{<:Any,NTuple{2,InfiniteCardinal{0}}}}) = LazyModalInterlaceLayout()
sublayout(::Union{ModalInterlaceLayout,LazyModalInterlaceLayout}, ::Type{<:NTuple{2,BlockSlice{<:BlockOneTo}}}) = ModalInterlaceLayout()


function sub_materialize(::ModalInterlaceLayout, V::AbstractMatrix{T}) where T
    kr,jr = parentindices(V)
    KR,JR = kr.block,jr.block
    M,N = Int(last(KR)), Int(last(JR))
    R = parent(V)
    ModalInterlace{T}([layout_getindex(R.ops[m],1:(M-m+2)÷2,1:(N-m+2)÷2) for m=1:min(N,M)], (M,N), R.bandwidths)
end

# act like lazy array
Base.BroadcastStyle(::Type{<:ModalInterlace{<:Any,NTuple{2,InfiniteCardinal{0}}}}) = LazyArrayStyle{2}()


# TODO: overload muladd!
function *(A::ModalInterlace, b::ModalTrav)
    M = b.matrix
    ret = similar(M)
    mul!(view(ret,:,1), A.ops[1], M[:,1])
    for j = 1:size(M,2)÷4
        mul!(@view(ret[1:end-j,4j-2]), A.ops[2j], @view(M[1:end-j,4j-2]))
        mul!(@view(ret[1:end-j,4j-1]), A.ops[2j], @view(M[1:end-j,4j-1]))
        mul!(@view(ret[1:end-j,4j]), A.ops[2j+1], @view(M[1:end-j,4j]))
        mul!(@view(ret[1:end-j,4j+1]), A.ops[2j+1], @view(M[1:end-j,4j+1]))
    end
    ModalTrav(ret)
end


function \(A::ModalInterlace, b::ModalTrav)
    M = b.matrix
    ret = similar(M)
    ldiv!(view(ret,:,1), A.ops[1], M[:,1])
    for j = 1:size(M,2)÷4
        ldiv!(@view(ret[1:end-j,4j-2]), A.ops[2j], @view(M[1:end-j,4j-2]))
        ldiv!(@view(ret[1:end-j,4j-1]), A.ops[2j], @view(M[1:end-j,4j-1]))
        ldiv!(@view(ret[1:end-j,4j]), A.ops[2j+1], @view(M[1:end-j,4j]))
        ldiv!(@view(ret[1:end-j,4j+1]), A.ops[2j+1], @view(M[1:end-j,4j+1]))
    end
    ModalTrav(ret)
end
