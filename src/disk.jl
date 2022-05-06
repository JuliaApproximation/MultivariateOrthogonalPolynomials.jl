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
# Jacobi matrices
###
function jacobimatrix(::Val{1}, Z::Zernike{T}) where T
    if iszero(Z.a)
        α = Z.b     # extract second basis parameter

    k = mortar(Base.OneTo.(oneto(∞)))     # k counts the the angular mode (+1)
    n = mortar(Fill.(oneto(∞),oneto(∞)))  # n counts the block number which corresponds to the order (+1)

    # repeatedly used for sorting
    keven = iseven.(k)
    kodd = isodd.(k)
    neven = iseven.(n)
    nodd = isodd.(n)

    # h1-h5 are helpers for our different sorting scheme

    ## Compute super diagonal of super diagonal blocks.
    dufirst = neven .* (k .== 2) .* n .* (n .+ 2*α) ./ 2

    ## Compute even-block entries in super diagonal of super diagonal blocks
    h1 = n .- (n .+ 2 .- k .- keven .* (n .> 2)) .÷ 2
    dueven = ((nodd .* k) .>= 2) .* h1 .* (h1 .+ α)

    ## Compute odd-block entries in super diagonal of super diagonal blocks
    h2 = (n .- 2 .+ k .+ kodd .* (n .> 2)) .÷ 2
    duodd = (((neven .* k) .>= 2)  .- (neven .* (k .== 2))) .* h2 .* (h2 .+ α)

    ## Compute even-block sub diagonal elements of super diagonal blocks
    h3 = n .- (k .+ 1 .+ kodd .+ n) .÷ 2
    dleven = ((k .<= (n .- 2)) .- ((k .< 2) .* nodd)) .* neven .* h3 .* (h3 .+ α)

    ## Compute odd-block sub diagonal elements of super diagonal blocks
    h4 = n .- (k .+ 1 .+ keven .+ n) .÷ 2
    dlodd = (k .> 1) .* (n .> 3) .* nodd .* h4 .* (h4 .+ α)

    ## Compute and add in special case odd-block sub diagonal elements of super diagonal blocks
    h5 = n .- 2 .+ k .+ keven
    dlspecial = nodd .* (k .== 1) .* h5 .* (h5 .+ 2*α) ./ 2

    # finalize bands with explicit formula
    quotient = 4 .* (n .+ (α-1)) .* (n .+ α)
    du = sqrt.( (dufirst .+ dueven .+ duodd ) ./ quotient)
    dl = sqrt.( (dleven .+ dlodd .+ dlspecial)  ./ quotient)

    return Symmetric(BlockBandedMatrices._BandedBlockBandedMatrix(BlockBroadcastArray(hcat, du, Zeros((axes(n,1),)), dl)', axes(n,1), (-1,1), (1,1)))
    else
        error("Implement for non-zero first basis parameter.")
    end
end

function jacobimatrix(::Val{2}, Z::Zernike{T}) where T
    if iszero(Z.a)
        α = Z.b     # extract second basis parameter

        k = mortar(Base.OneTo.(oneto(∞)))     # k counts the the angular mode (+1)
        n = mortar(Fill.(oneto(∞),oneto(∞)))  # n counts the block number which corresponds to the order (+1)
    
        # repeatedly used for sorting
        keven = iseven.(k)
        kodd = isodd.(k)
        neven = iseven.(n)
        nodd = isodd.(n)
            
        # h1-h4 are helpers for our different sorting scheme
    
        # first entries for all blocks
        h1 = (n .- nodd)
        l1 = (k .== 1) .* (h1 .* (h1 .+ 2*α) ./ 2)
    
        # Even blocks
        h0 = (kodd .* ((k .÷ 2) .+ 1) .- (keven .* ((k .÷ 2) .- 1)))
        h2 =  (k .>= 2) .* ((n .÷ 2 .- 1) .+ h0)
        l2 = neven .* (h2 .* (h2 .+ α))
    
        # Odd blocks
        h3 = (n .> k .>= 2) .* (((n .+ 1) .÷ 2) .- h0)
        l3 = nodd .* (h3 .* (h3 .+ α))
        # Combine for diagonal of super diagonal block
        d = sqrt.((l1 .+ l2 .+ l3) ./ (4 .* (n .+ (α-1)) .* (n .+ α)))
    
        # The off-diagonals of the super diagonal block are negative, shifted versions of the diagonal with some entries skipped
        dl = (-1) .* (nodd .* kodd .+ neven .* keven) .* Vcat(0 , d)
        du = (-1) .* (nodd .* keven .+ neven .* kodd) .* view(d,2:∞)
    
        # generate and return bands
        return Symmetric(BlockBandedMatrices._BandedBlockBandedMatrix(BlockBroadcastArray(hcat, dl, Zeros((axes(n,1),)), d, Zeros((axes(n,1),)), du)', axes(n,1), (-1,1), (2,2)))
    else
        error("Implement for non-zero first basis parameter.")
    end
end

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


function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{Zernike, AbstractVector}}, x) where T
    Z,c = u.args
    CS = blockcolsupport(c)
    N = Int(last(CS)) ÷ 2 + 1 # polynomial degree
    F = ZernikeITransform{T}(2N, Z.a, Z.b)
    C = F * c[Block.(OneTo(2N))] # transform to grid
    [permutedims(u[x[1,:]]); # evaluate on edge of disk
     C C[:,1];
     fill(u[x[end,1]], 1, size(x,2))] # evaluate at origin and repeat
end

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{Weighted{<:Any,<:Zernike}, AbstractVector}}, x) where T
    U = plotvalues(unweighted(u))
    w = weight(U.args[1])
    w[x] .* U
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
