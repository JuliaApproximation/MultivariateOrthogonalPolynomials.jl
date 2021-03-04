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


struct Zernike{T} <: BivariateOrthogonalPolynomial{T}
    a::T
    b::T
    Zernike{T}(a::T, b::T) where T = new{T}(a, b)
end
Zernike{T}(a, b) where T = Zernike{T}(convert(T,a), convert(T,b))
Zernike(a::T, b::V) where {T,V} = Zernike{float(promote_type(T,V))}(a, b)
Zernike{T}() where T = Zernike{T}(zero(T), zero(T))
Zernike() = Zernike{Float64}()

axes(P::Zernike{T}) where T = (Inclusion(UnitDisk{T}()),blockedrange(oneto(∞)))

copy(A::Zernike) = A

zerniker(ℓ, m, a, b, r::T) where T = sqrt(convert(T,2)^(m+a+b+2-iszero(m))/π) * r^m * Normalized(Jacobi{T}(b, m+a))[2r^2-1,(ℓ-m) ÷ 2 + 1]

function zernikez(ℓ, ms, a, b, rθ::RadialCoordinate{T}) where T
    r,θ = rθ.r,rθ.θ
    m = abs(ms)
    zerniker(ℓ, m, a, b, r) * (signbit(ms) ? sin(m*θ) : cos(m*θ))
end

zernikez(ℓ, ms, a, b, xy::StaticVector{2}) = zernikez(ℓ, ms, a, b, RadialCoordinate(xy))

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
