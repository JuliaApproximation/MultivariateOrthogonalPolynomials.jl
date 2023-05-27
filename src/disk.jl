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

function grid(S::Zernike{T}, B::Block{1}) where T
    N = Int(B) ÷ 2 + 1 # matrix rows
    M = 4N-3 # matrix columns

    r = sinpi.((N .-(0:N-1) .- one(T)/2) ./ (2N))

    # The angular grid:
    θ = (0:M-1)*convert(T,2)/M
    RadialCoordinate.(r, π*θ')
end

_angle(rθ::RadialCoordinate) = rθ.θ

function plotgrid(S::Zernike{T}, B::Block{1}) where T
    N = Int(B) ÷ 2 + 1 # polynomial degree
    g = grid(S, Block(min(2N, MAX_PLOT_BLOCKS))) # double sampling
    θ = [map(_angle,g[1,:]); 0]
    [permutedims(RadialCoordinate.(1,θ));
     g g[:,1];
     permutedims(RadialCoordinate.(0,θ))]
end

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{Zernike, AbstractVector}}, x) where T
    Z,c = u.args
    B = findblock(axes(Z,2), last(colsupport(c)))
    N = Int(B) ÷ 2 + 1 # polynomial degree
    F = ZernikeITransform{T}(min(2N, MAX_PLOT_BLOCKS), Z.a, Z.b)
    C = F * c[Block.(OneTo(min(2N, MAX_PLOT_BLOCKS)))] # transform to grid
    [permutedims(u[x[1,:]]); # evaluate on edge of disk
     C C[:,1];
     fill(u[x[end,1]], 1, size(x,2))] # evaluate at origin and repeat
end

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{Weighted{<:Any,<:Zernike}, AbstractVector}}, x) where T
    U = plotvalues(unweighted(u), x)
    w = weight(u.args[1])
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
*(P::ZernikeTransform{T}, f::Matrix{T}) where T = ModalTrav(P.disk2cxf \ (P.analysis * f))
*(P::ZernikeITransform, f::AbstractVector) = P.synthesis * (P.disk2cxf * ModalTrav(f).matrix)

plan_grid_transform(Z::Zernike{T}, B::Tuple{Block{1}}, dims=1:1) where T = grid(Z,B[1]), ZernikeTransform{T}(Int(B[1]), Z.a, Z.b)

##
# Laplacian
###

@simplify function *(Δ::Laplacian, WZ::Weighted{<:Any,<:Zernike})
    @assert WZ.P.a == 0 && WZ.P.b == 1
    T = eltype(eltype(WZ))
    WZ.P * ModalInterlace{T}(broadcast(k ->  Diagonal(-cumsum(k:8:∞)), 4:4:∞), (ℵ₀,ℵ₀), (0,0))
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


###
# sum
###

function Base._sum(P::Zernike{T}, dims) where T
    @assert dims == 1
    @assert P.a == P.b == 0
    Hcat(sqrt(convert(T, π)), Zeros{T}(1,∞))
end


###
# Partial derivatives
###

# Helper function for the case weighted(1) to zernike(0)
_normal_jacobi(a::T,b::T,n::Int) where T = sqrt(2^(a+b+1) / (2n + a + b + 1) * gamma(n+a+1)*gamma(n+b+1) / (gamma(n+a+b+1) * gamma(n+1)))

@simplify function *(∂ʸ::PartialDerivative{2}, WZ::Weighted{<:Any,<:Zernike})
    @assert WZ.P.a == 0 && WZ.P.b == 1
    T = eltype(eltype(WZ))

    k = mortar(Base.OneTo.(oneto(∞)))     # k counts the the angular mode (+1)
    n = mortar(Fill.(oneto(∞),oneto(∞))) .- 1  # n counts the block number which corresponds to the order
    m = k .- isodd.(k).*iseven.(n) .- iseven.(k).*isodd.(n) # Fourier mode number

    # Raising Fourier mode coefficients, split into sin and cos parts + m=0 change
    c=-(_normal_jacobi.(T(0), T(1), (n .- m) .÷ 2) ./ _normal_jacobi.(T(1), T(0), (n .- m) .÷ 2 .+ m))
    c1 = c .* ((n  .- m) .÷ 2 .+ 1) .* (-1).^(iseven.(k.-n)) .* (iszero.(m) * sqrt(T(2)) + (1 .- iszero.(m)))
    c2 = c1 .* iseven.(n .- k)
    c3 = c1 - c2

    # Lowering Fourier mode coefficients, split into sin and cos parts + m=1 change
    d1 = c .* ((n  .- m) .÷ 2 .+ 1) .* (-1).^(isodd.(k.-n)) .* (isone.(m) * sqrt(T(2)) + (1 .- isone.(m)))
    d2 = d1 .* iseven.(n .- k)
    d3 = d1 - d2

    # Put coefficients together
    A = BlockBandedMatrices._BandedBlockBandedMatrix(BlockBroadcastArray(hcat, c3, Zeros((axes(n,1),)), c2)', axes(n,1), (1,-1), (2,0))
    B = BlockBandedMatrices._BandedBlockBandedMatrix(BlockBroadcastArray(hcat, d3, Zeros((axes(n,1),)), d2)', axes(n,1), (1,-1), (0,2))
    Zernike{T}(0) * (A + B)
end

@simplify function *(∂ʸ::PartialDerivative{2}, Z::Zernike)
    @assert Z.a == 0
    T = eltype(eltype(Z))
    b = convert(T, Z.b)

    k = mortar(Base.OneTo.(oneto(∞)))     # k counts the the angular mode (+1)
    n = mortar(Fill.(oneto(∞),oneto(∞))) .- 1  # n counts the block number which corresponds to the order
    m = k .- isodd.(k).*iseven.(n) .- iseven.(k).*isodd.(n)

    # Computes the entries for the component that lowers the Fourier mode
    x=axes(Jacobi(0,0),1)
    D = BroadcastVector(P->Derivative(x) * P, HalfWeighted{:b}.(Normalized.(Jacobi.(b,1:∞))))
    Ds = BroadcastVector{AbstractMatrix{Float64}}((P,D) -> P \ D, HalfWeighted{:b}.(Normalized.(Jacobi.(b+1,0:∞))) , D)
    M = ModalInterlace(Ds, (ℵ₀,ℵ₀), (0,0))
    db = ones(axes(n)) .* (view(view(M, 1:∞, 1:∞),band(0)))

    # Computes the entries for the component that lowers the Fourier mode
    # the -1 is a hack is the parameters is a trick to make ModalInterlace think that the
    # 0-parameter is the second Fourier mode (we use the -1 as a dummy matrix in the ModalInterlace)
    D = BroadcastVector(P->Derivative(x) * P, Normalized.(Jacobi.(b,-1:∞)))
    Ds = BroadcastVector{AbstractMatrix{Float64}}((P,D) -> P \ D, Normalized.(Jacobi.(b+1,0:∞)), D)
    Dss = BroadcastVector{AbstractMatrix{Float64}}(P -> Diagonal(view(P, band(1))), Ds)
    M = ModalInterlace(Dss, (ℵ₀,ℵ₀), (0,0))
    d = ones(axes(n)) .*  view(view(M, 1:∞, 1:∞),band(0))

    # Lowering Fourier mode coefficients, split into sin and cos parts + m=1 change
    c1 = d .* (-1).^(isodd.(k.-n)) .* (isone.(m) .* sqrt(T(2)) + (1 .- isone.(m)))
    c2 = c1 .* iseven.(n .- k)
    c3 = c1 - c2

    # Raising Fourier mode coefficients, split into sin and cos parts + m=0 change
    d1 = db .* (-1).^(iseven.(k.-n)) .* (iszero.(m) * sqrt(T(2)) + (1 .- iszero.(m)))
    d2 = d1 .* iseven.(n .- k)
    d3 = d1 - d2

    # Put coefficients together
    A = BlockBandedMatrices._BandedBlockBandedMatrix(BlockBroadcastArray(hcat, c3, Zeros((axes(n,1),)), c2)', axes(n,1), (1,-1), (0,2))
    B = BlockBandedMatrices._BandedBlockBandedMatrix(BlockBroadcastArray(hcat, d3, Zeros((axes(n,1),)), d2)', axes(n,1), (1,-1), (2,0))

    Zernike{T}(Z.b+1) * (A+B)'
end