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


struct Zernike{T} <: BivariateOrthogonalPolynomial{T} end

Zernike() = Zernike{Float64}()

axes(P::Zernike{T}) where T = (Inclusion(UnitDisk{T}()),blockedrange(oneto(∞)))

copy(A::Zernike) = A

function getindex(Z::Zernike{T}, rθ::RadialCoordinate, B::BlockIndex{1}) where T
    r,θ = rθ.r, rθ.θ
    ℓ = Int(block(B))-1
    k = blockindex(B)
    m = iseven(ℓ) ? k-isodd(k) : k-iseven(k)
    if iszero(m)
        sqrt(convert(T,2)/π) * Normalized(Legendre{T}())[2r^2-1, ℓ ÷ 2 + 1]
    else
        convert(T,2^((m+2)/2))/π * r^m * Normalized(Jacobi{T}(0, m))[2r^2-1,(ℓ-m) ÷ 2 + 1] * (isodd(k+ℓ) ? cos(m*θ) : sin(m*θ))
    end
end


getindex(Z::Zernike, xy::StaticVector{2}, B::BlockIndex{1}) = Z[RadialCoordinate(xy), B]
getindex(Z::Zernike, xy::StaticVector{2}, B::Block{1}) = [Z[xy, B[j]] for j=1:Int(B)]
getindex(Z::Zernike, xy::StaticVector{2}, JR::BlockOneTo) = mortar([Z[xy,Block(J)] for J = 1:Int(JR[end])])