"""
    DiskTrav(A::AbstractMatrix)

    takes coefficients as provided by the Zernike polynomial layout of FastTransforms.jl and
    makes them accessible sorted such that in each block the m=0 entries are always in first place, 
    followed by alternating sin and cos terms of increasing |m|.
"""
struct DiskTrav{T, AA<:AbstractMatrix{T}} <: AbstractBlockVector{T}
    matrix::AA
    function DiskTrav{T, AA}(matrix::AA) where {T,AA<:AbstractMatrix{T}}
        n,m = size(matrix)
        isodd(m) && n == m ÷ 4 + 1 || throw(ArgumentError("size must match"))
        new{T,AA}(matrix)
    end
end

DiskTrav{T}(matrix::AbstractMatrix{T}) where T = DiskTrav{T,typeof(matrix)}(matrix)
DiskTrav(matrix::AbstractMatrix{T}) where T = DiskTrav{T}(matrix)

axes(A::DiskTrav) = (blockedrange(oneto(div(size(A.matrix,2),2,RoundUp))),)

function getindex(A::DiskTrav, K::Block{1})
    k = Int(K)
    k == 1 && return A.matrix[1:1]
    k == 2 && return A.matrix[1,2:3]
    st = stride(A.matrix,2)
    if isodd(k)
        # nonnegative terms
        p = A.matrix[range(k÷2+1, step=4*st-1, length=k÷2+1)]
        # negative terms
        n = A.matrix[range(k÷2+3*st, step=4*st-1, length=k÷2)]
        interlace(p,n)
    else
        # negative terms
        n = A.matrix[range(st+k÷2, step=4*st-1, length=k÷2)]
        # positive terms
        p = A.matrix[range(2st+k÷2, step=4*st-1, length=k÷2)]
        interlace(n,p)
    end
end

getindex(A::DiskTrav, k::Int) = A[findblockindex(axes(A,1), k)]

ClassicalOrthogonalPolynomials.checkpoints(d::UnitDisk{T}) where T = [SVector{2,T}(0.1,0.2), SVector{2,T}(0.2,0.3)]

"""
    ZernikeWeight(a, b)

is a quasi-vector representing `r^(2a) * (1-r^2)^b`
"""
struct ZernikeWeight{T} <: Weight{T}
    a::T
    b::T
end

"""
    ZernikeWeight(b)

is a quasi-vector representing `(1-r^2)^b`
"""

ZernikeWeight(b) = ZernikeWeight(zero(b), b)

axes(::ZernikeWeight{T}) where T = (Inclusion(UnitDisk{T}()),)

==(w::ZernikeWeight, v::ZernikeWeight) = w.a == v.a && w.b == v.b

function getindex(w::ZernikeWeight, xy::StaticVector{2})
    r = norm(xy)
    r^(2w.a) * (1-r^2)^w.b
end


"""
    Zernike(a, b)

is a quasi-matrix orthogonal `r^(2a) * (1-r^2)^b`
"""
struct Zernike{T} <: BivariateOrthogonalPolynomial{T}
    a::T
    b::T
    Zernike{T}(a::T, b::T) where T = new{T}(a, b)
end
Zernike{T}(a, b) where T = Zernike{T}(convert(T,a), convert(T,b))
Zernike(a::T, b::V) where {T,V} = Zernike{float(promote_type(T,V))}(a, b)
Zernike{T}(b) where T = Zernike{T}(zero(b), b)
Zernike{T}() where T = Zernike{T}(zero(T))

"""
    Zernike(b)

is a quasi-matrix orthogonal `(1-r^2)^b`
"""
Zernike(b) = Zernike(zero(b), b)
Zernike() = Zernike{Float64}()

axes(P::Zernike{T}) where T = (Inclusion(UnitDisk{T}()),blockedrange(oneto(∞)))

==(w::Zernike, v::Zernike) = w.a == v.a && w.b == v.b

copy(A::Zernike) = A

orthogonalityweight(Z::Zernike) = ZernikeWeight(Z.a, Z.b)

zerniker(ℓ, m, a, b, r::T) where T = sqrt(convert(T,2)^(m+a+b+2-iszero(m))/π) * r^m * normalizedjacobip((ℓ-m) ÷ 2, b, m+a, 2r^2-1)
zerniker(ℓ, m, b, r) = zerniker(ℓ, m, zero(b), b, r)
zerniker(ℓ, m, r) = zerniker(ℓ, m, zero(r), r)

function zernikez(ℓ, ms, a, b, rθ::RadialCoordinate{T}) where T
    r,θ = rθ.r,rθ.θ
    m = abs(ms)
    zerniker(ℓ, m, a, b, r) * (signbit(ms) ? sin(m*θ) : cos(m*θ))
end

zernikez(ℓ, ms, a, b, xy::StaticVector{2}) = zernikez(ℓ, ms, a, b, RadialCoordinate(xy))
zernikez(ℓ, ms, b, xy::StaticVector{2}) = zernikez(ℓ, ms, zero(b), b, xy)
zernikez(ℓ, ms, xy::StaticVector{2,T}) where T = zernikez(ℓ, ms, zero(T), xy)

function getindex(Z::Zernike{T}, rθ::RadialCoordinate, B::BlockIndex{1}) where T
    ℓ = Int(block(B))-1
    k = blockindex(B)
    m = iseven(ℓ) ? k-isodd(k) : k-iseven(k)
    zernikez(ℓ, (isodd(k+ℓ) ? 1 : -1) * m, Z.a, Z.b, rθ)
end


getindex(Z::Zernike, xy::StaticVector{2}, B::BlockIndex{1}) = Z[RadialCoordinate(xy), B]
getindex(Z::Zernike, xy::StaticVector{2}, B::Block{1}) = [Z[xy, B[j]] for j=1:Int(B)]
getindex(Z::Zernike, xy::StaticVector{2}, JR::BlockOneTo) = mortar([Z[xy,Block(J)] for J = 1:Int(JR[end])])




###
# Transforms
###

const FiniteZernike{T} = SubQuasiArray{T,2,Zernike{T},<:Tuple{<:Inclusion,<:BlockSlice{BlockRange1{OneTo{Int}}}}}

function grid(S::FiniteZernike{T}) where T
    N = blocksize(S,2) ÷ 2 + 1 # polynomial degree
    M = 4N-3

    r = sinpi.((N .-(0:N-1) .- one(T)/2) ./ (2N))

    # The angular grid:
    θ = (0:M-1)*convert(T,2)/M
    RadialCoordinate.(r, π*θ')
end

struct ZernikeTransform{T} <: Plan{T}
    N::Int
    disk2cxf::FastTransforms.FTPlan{T,2,FastTransforms.DISK}
    analysis::FastTransforms.FTPlan{T,2,FastTransforms.DISKANALYSIS}
end

function ZernikeTransform{T}(N::Int, a::Number, b::Number) where T<:Real
    Ñ = N ÷ 2 + 1
    ZernikeTransform{T}(N, plan_disk2cxf(T, Ñ, a, b), plan_disk_analysis(T, Ñ, 4Ñ-3))
end
*(P::ZernikeTransform{T}, f::Matrix{T}) where T = DiskTrav(P.disk2cxf \ (P.analysis * f))[Block.(1:P.N)]

factorize(S::FiniteZernike{T}) where T = TransformFactorization(grid(S), ZernikeTransform{T}(blocksize(S,2), parent(S).a, parent(S).b))

# gives the entries for the Laplacian times (1-r^2) * Zernike(1)
struct WeightedZernikeLaplacianDiag{T} <: AbstractBlockVector{T} end

axes(::WeightedZernikeLaplacianDiag) = (blockedrange(oneto(∞)),)

function Base.view(W::WeightedZernikeLaplacianDiag{T}, K::Block{1}) where T
    K̃ = Int(K)
    d = K̃÷2 + 1
    if isodd(K̃)
        v = (d:K̃) .* (d:(-1):1)
        convert(AbstractVector{T}, -4*interlace(v, v[2:end]))
    else
        v = (d:K̃) .* (d-1:(-1):1)
        convert(AbstractVector{T}, -4 * interlace(v, v))
    end
end

getindex(W::WeightedZernikeLaplacianDiag, k::Integer) = W[findblockindex(axes(W,1),k)]

@simplify function *(Δ::Laplacian, WZ::Weighted{<:Any,<:Zernike})
    @assert WZ.P.a == 0 && WZ.P.b == 1
    WZ.P * Diagonal(WeightedZernikeLaplacianDiag{eltype(eltype(WZ))}())
end

struct ZernikeConversion{T} <: AbstractBandedBlockBandedMatrix{T} end
ZernikeConversion() = ZernikeConversion{Float64}()

axes(Z::ZernikeConversion) = (blockedrange(oneto(∞)), blockedrange(oneto(∞)))

blockbandwidths(::ZernikeConversion) = (0,2)
subblockbandwidths(::ZernikeConversion) = (0,0)


function Base.view(W::ZernikeConversion{T}, KJ::Block{2}) where T
    K,J = KJ.n
    dat = Vector{T}()
    if J == K
        if iseven(K)
            R0 = Normalized(Jacobi(1,0)) \ Normalized(Jacobi(0,0))
            push!(dat, R0[K÷2-k+2,K÷2-k+2])
        end
        for m in enumerate(range(2+iseven(K); step=2, length=J))
            Rm = Normalized(Jacobi(1,m)) \ Normalized(Jacobi(0,m))
            dat[1,k] = Rm[K÷2-k+2,K÷2-k+2]
        end
    elseif J == K + 2
        for (k,m) in enumerate(range(Int(iseven(K)); step=2, length=K))
            Rm = Normalized(Jacobi(1,m)) \ Normalized(Jacobi(0,m))
            dat[1,k] = Rm[K÷2-k+2,K÷2-k+3]
        end
    else
        fill!(dat, zero(T))
    end
    _BandedMatrix(dat, K, 0, 0)
end

getindex(R::ZernikeConversion, k::Integer, j::Integer) = R[findblockindex.(axes(R),(k,j))...]