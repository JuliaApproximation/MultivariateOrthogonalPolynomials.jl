"""
    DiskTrav(A::AbstractMatrix)

    takes coefficients as provided by the Zernike polynomial layout of FastTransforms.jl and
    makes them accessible sorted such that in each block the m=0 entries are always in first place, 
    followed by alternating sin and cos terms of increasing |m|.
"""
struct DiskTrav{T, AA<:AbstractMatrix{T}} <: AbstractBlockVector{T}
    matrix::AA
    function DiskTrav{T, AA}(matrix::AA) where {T,AA<:AbstractMatrix{T}}
        m,n = size(matrix)
        isodd(n) && m == n ÷ 4 + 1 || throw(ArgumentError("size must match"))
        new{T,AA}(matrix)
    end
end

DiskTrav{T}(matrix::AbstractMatrix{T}) where T = DiskTrav{T,typeof(matrix)}(matrix)
DiskTrav(matrix::AbstractMatrix{T}) where T = DiskTrav{T}(matrix)

function DiskTrav(v::AbstractVector{T}) where T
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
    DiskTrav(mat)
end

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
ZernikeWeight{T}(b) where T = ZernikeWeight{T}(zero(T), b)
ZernikeWeight{T}() where T = ZernikeWeight{T}(zero(T))
ZernikeWeight() = ZernikeWeight{Float64}()

copy(w::ZernikeWeight) = w

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

Base.summary(io::IO, P::Zernike) = print(io, "Zernike($(P.a), $(P.b))")

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

_angle(rθ::RadialCoordinate) = rθ.θ

function plotgrid(S::FiniteZernike{T}) where T
    N = blocksize(S,2) ÷ 2 + 1 # polynomial degree
    g = grid(parent(S)[:,Block.(OneTo(2N))]) # double sampling
    θ = [map(_angle,g[1,:]); 0]
    [permutedims(RadialCoordinate.(1,θ)); g g[:,1]; permutedims(RadialCoordinate.(0,θ))]
end

function plotgrid(S::SubQuasiArray{<:Any,2,<:Zernike})
    kr,jr = parentindices(S)
    Z = parent(S)
    plotgrid(Z[kr,Block.(OneTo(Int(findblock(axes(Z,2),maximum(jr)))))])
end

struct ZernikeTransform{T} <: Plan{T}
    N::Int
    disk2cxf::FastTransforms.FTPlan{T,2,FastTransforms.DISK}
    analysis::FastTransforms.FTPlan{T,2,FastTransforms.DISKANALYSIS}
end

struct ZernikeITransform{T} <: Plan{T}
    N::Int
    disk2cxf::FastTransforms.FTPlan{T,2,FastTransforms.DISK}
    synthesis::FastTransforms.FTPlan{T,2,FastTransforms.DISKSYNTHESIS}
end

function ZernikeTransform{T}(N::Int, a::Number, b::Number) where T<:Real
    Ñ = N ÷ 2 + 1
    ZernikeTransform{T}(N, plan_disk2cxf(T, Ñ, a, b), plan_disk_analysis(T, Ñ, 4Ñ-3))
end
function ZernikeITransform{T}(N::Int, a::Number, b::Number) where T<:Real
    Ñ = N ÷ 2 + 1
    ZernikeITransform{T}(N, plan_disk2cxf(T, Ñ, a, b), plan_disk_synthesis(T, Ñ, 4Ñ-3))
end

*(P::ZernikeTransform{T}, f::AbstractArray) where T = P * convert(Matrix{T}, f)
*(P::ZernikeTransform{T}, f::Matrix{T}) where T = DiskTrav(P.disk2cxf \ (P.analysis * f))[Block.(1:P.N)]
*(P::ZernikeITransform, f::AbstractVector) = P.synthesis * (P.disk2cxf * DiskTrav(f).matrix)

factorize(S::FiniteZernike{T}) where T = TransformFactorization(grid(S), ZernikeTransform{T}(blocksize(S,2), parent(S).a, parent(S).b))

# gives the entries for the Laplacian times (1-r^2) * Zernike(1)
struct WeightedZernikeLaplacianDiag{T} <: AbstractBlockVector{T} end

axes(::WeightedZernikeLaplacianDiag) = (blockedrange(oneto(∞)),)
copy(R::WeightedZernikeLaplacianDiag) = R

MemoryLayout(::Type{<:WeightedZernikeLaplacianDiag}) = LazyLayout()
Base.BroadcastStyle(::Type{<:Diagonal{<:Any,<:WeightedZernikeLaplacianDiag}}) = LazyArrayStyle{2}()

function Base.view(W::WeightedZernikeLaplacianDiag{T}, K::Block{1}) where T
    K̃ = Int(K)
    d = K̃÷2 + 1
    if isodd(K̃)
        v = (d:K̃) .* (d:(-1):1)
        convert(AbstractVector{T}, -4*interlace(v, v[2:end]))
    else
        v = (d:K̃) .* (d-1:(-1):1)
        convert(AbstractVector{T}, -4*interlace(v, v))
    end
end

getindex(W::WeightedZernikeLaplacianDiag, k::Integer) = W[findblockindex(axes(W,1),k)]

@simplify function *(Δ::Laplacian, WZ::Weighted{<:Any,<:Zernike})
    @assert WZ.P.a == 0 && WZ.P.b == 1
    WZ.P * Diagonal(WeightedZernikeLaplacianDiag{eltype(eltype(WZ))}())
end

@simplify function *(Δ::Laplacian, Z::Zernike)
    a,b = Z.a,Z.b
    @assert a == 0
    T = promote_type(eltype(eltype(Δ)),eltype(Z)) # TODO: remove extra eltype
    D = Derivative(Inclusion(ChebyshevInterval{T}())) 
    Δs = BroadcastVector{AbstractMatrix{T}}((C,B,A) -> 4(HalfWeighted{:b}(C)\(D*HalfWeighted{:b}(B)))*(B\(D*A)), Normalized.(Jacobi.(b+2,a:∞)), Normalized.(Jacobi.(b+1,(a+1):∞)), Normalized.(Jacobi.(b,a:∞)))
    Δ = ModalInterlace(Δs, (ℵ₀,ℵ₀), (-2,2))
    Zernike(a,b+2) * Δ
end

###
# Fractional Laplacian
###

function *(L::AbsLaplacianPower, WZ::Weighted{<:Any,<:Zernike{<:Any}})
    @assert axes(L,1) == axes(WZ,1) && WZ.P.a == 0 && WZ.P.b == L.α
    WZ.P * Diagonal(WeightedZernikeFractionalLaplacianDiag{typeof(L.α)}(L.α))
end

# gives the entries for the (negative!) fractional Laplacian (-Δ)^(α) times (1-r^2)^α * Zernike(α)
struct WeightedZernikeFractionalLaplacianDiag{T} <: AbstractBlockVector{T} 
    α::T
end

axes(::WeightedZernikeFractionalLaplacianDiag) = (blockedrange(oneto(∞)),)
copy(R::WeightedZernikeFractionalLaplacianDiag) = R

MemoryLayout(::Type{<:WeightedZernikeFractionalLaplacianDiag}) = LazyLayout()
Base.BroadcastStyle(::Type{<:Diagonal{<:Any,<:WeightedZernikeFractionalLaplacianDiag}}) = LazyArrayStyle{2}()

getindex(W::WeightedZernikeFractionalLaplacianDiag, k::Integer) = W[findblockindex(axes(W,1),k)]

function Base.view(W::WeightedZernikeFractionalLaplacianDiag{T}, K::Block{1}) where T
    l = Int(K)
    if isodd(l)
        m = Vcat(0,interlace(Array(2:2:l),Array(2:2:l)))
    else #if iseven(l)
        m = Vcat(interlace(Array(1:2:l),Array(1:2:l)))
    end
    return convert(AbstractVector{T}, 2^(2*W.α)*fractionalcfs2d.(l-1,m,W.α))
end

# generic d-dimensional ball fractional coefficients without the 2^(2*β) factor. m is assumed to be entered as abs(m)
function fractionalcfs(l::Integer, m::Integer, α::T, d::Integer) where T
    n = (l-m)÷2
    return exp(loggamma(α+n+1)+loggamma((2*α+2*n+d+2*m)/2)-loggamma(one(T)+n)-loggamma((2*one(T)*m+2*n+d)/2))
end
# 2 dimensional special case, again without the 2^(2*β) factor
fractionalcfs2d(l::Integer, m::Integer, β) = fractionalcfs(l,m,β,2)

function \(A::Zernike{T}, B::Zernike{V}) where {T,V}
    TV = promote_type(T,V)
    A.a == B.a && A.b == B.b && return Eye{TV}((axes(A,2),))
    st = Int(A.a - B.a + A.b - B.b)
    ModalInterlace{TV}((Normalized.(Jacobi{TV}.(A.b,A.a:∞)) .\ Normalized.(Jacobi{TV}.(B.b,B.a:∞))) .* convert(TV, 2)^(-st/2), (ℵ₀,ℵ₀), (0,2st))
end

function \(A::Zernike{T}, wB::Weighted{V,Zernike{V}}) where {T,V}
    TV = promote_type(T,V)
    B = wB.P
    A.a == B.a == A.b == B.b == 0 && return Eye{TV}((axes(A,2),))
    c = Int(B.a - A.a + B.b - A.b)
    @assert iszero(B.a)
    ModalInterlace{TV}((Normalized.(Jacobi{TV}.(A.b, A.a:∞)) .\ HalfWeighted{:a}.(Normalized.(Jacobi{TV}.(B.b, B.a:∞)))) .* convert(TV, 2)^(-c/2), (ℵ₀,ℵ₀), (2Int(B.b), 2Int(A.a+A.b)))
end