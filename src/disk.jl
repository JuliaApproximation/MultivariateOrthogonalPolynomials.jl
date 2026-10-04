ClassicalOrthogonalPolynomials.checkpoints(d::UnitDisk{T}) where T = [SVector{2,T}(0.1,0.2), SVector{2,T}(0.2,0.3)]
pointchoice(d::UnitDisk{T}) where T = SVector{2,T}(0,0)

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

is a quasi-matrix orthogonal `r^(2a) * (1-r^2)^b`. The polynomials are not normalized:
the entry of degree `ℓ` and Fourier mode `m` is

    r^|m| * jacobip((ℓ-|m|) ÷ 2, b, |m|+a, 2r^2-1) * (signbit(m) ? sin(|m|*θ) : cos(|m|*θ))

Use `Normalized(Zernike(a, b))` for the orthonormal polynomials.
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

AbstractQuasiArray{T}(::Zernike) where T = Zernike{T}()
AbstractQuasiMatrix{T}(::Zernike) where T = Zernike{T}()

"""
    Zernike(b)

is a quasi-matrix orthogonal `(1-r^2)^b`
"""
Zernike(b) = Zernike(zero(b), b)
Zernike() = Zernike{Float64}()

axes(P::Zernike{T}) where T = (Inclusion(UnitDisk{real(T)}()),blockedrange(oneto(∞)))

==(w::Zernike, v::Zernike) = w.a == v.a && w.b == v.b

copy(A::Zernike) = A

show(io::IO, P::Zernike) = summary(io, P)
summary(io::IO, P::Zernike) = print(io, "Zernike($(P.a), $(P.b))")

orthogonalityweight(Z::Zernike) = ZernikeWeight(Z.a, Z.b)

basis_axes(::Inclusion{<:Any,<:UnitDisk}, v) = Zernike()

###
# Normalized Zernike
###

const NormalizedZernike{T} = Normalized{T,Zernike{T}}
const ZernikeOrNormalized{T} = Union{Zernike{T},NormalizedZernike{T}}

MemoryLayout(::Type{<:NormalizedZernike}) = MultivariateOPLayout{2}()

_zernike(Z::Zernike) = Z
_zernike(Q::NormalizedZernike) = Q.P

# a Zernike basis of the same kind (normalized or not) with different parameters
_zernikesimilar(::Zernike{T}, a, b) where T = Zernike{T}(a, b)
_zernikesimilar(::NormalizedZernike{T}, a, b) where T = Normalized(Zernike{T}(a, b))

==(A::NormalizedZernike, B::NormalizedZernike) = A.P == B.P
==(::Zernike, ::NormalizedZernike) = false
==(::NormalizedZernike, ::Zernike) = false

copy(Q::NormalizedZernike) = Q

summary(io::IO, Q::NormalizedZernike) = print(io, "Normalized(Zernike($(Q.P.a), $(Q.P.b)))")


###
# Evaluation
###

zerniker(ℓ, m, a, b, r::T) where T = r^m * jacobip((ℓ-m) ÷ 2, b, m+a, 2r^2-1)
zerniker(ℓ, m, b, r) = zerniker(ℓ, m, zero(b), b, r)
zerniker(ℓ, m, r) = zerniker(ℓ, m, zero(r), r)

normalizedzerniker(ℓ, m, a, b, r::T) where T = sqrt(convert(T,2)^(m+a+b+2-iszero(m))/π) * r^m * normalizedjacobip((ℓ-m) ÷ 2, b, m+a, 2r^2-1)
normalizedzerniker(ℓ, m, b, r) = normalizedzerniker(ℓ, m, zero(b), b, r)
normalizedzerniker(ℓ, m, r) = normalizedzerniker(ℓ, m, zero(r), r)

function _zernikez(radial, ℓ, ms, a, b, rθ::RadialCoordinate)
    r,θ = rθ.r,rθ.θ
    m = abs(ms)
    radial(ℓ, m, a, b, r) * (signbit(ms) ? sin(m*θ) : cos(m*θ))
end

for (z, radial) in ((:zernikez, :zerniker), (:normalizedzernikez, :normalizedzerniker))
    @eval begin
        $z(ℓ, ms, a, b, rθ::RadialCoordinate) = _zernikez($radial, ℓ, ms, a, b, rθ)
        $z(ℓ, ms, a, b, xy::StaticVector{2}) = $z(ℓ, ms, a, b, RadialCoordinate(xy))
        $z(ℓ, ms, b, xy::StaticVector{2}) = $z(ℓ, ms, zero(b), b, xy)
        $z(ℓ, ms, xy::StaticVector{2,T}) where T = $z(ℓ, ms, zero(T), xy)
    end
end

# degree ℓ and signed Fourier mode m of the entry B
function _zernikelm(B::BlockIndex{1})
    ℓ = Int(block(B))-1
    k = blockindex(B)
    m = iseven(ℓ) ? k-isodd(k) : k-iseven(k)
    ℓ, (isodd(k+ℓ) ? 1 : -1) * m
end

getindex(Z::Zernike, rθ::RadialCoordinate, B::BlockIndex{1}) = zernikez(_zernikelm(B)..., Z.a, Z.b, rθ)
getindex(Q::NormalizedZernike, rθ::RadialCoordinate, B::BlockIndex{1}) = normalizedzernikez(_zernikelm(B)..., Q.P.a, Q.P.b, rθ)

for Typ in (:Zernike, :NormalizedZernike)
    @eval begin
        getindex(Z::$Typ, xy::StaticVector{2}, B::BlockIndex{1}) = Z[RadialCoordinate(xy), B]
        getindex(Z::$Typ, xy::StaticVector{2}, B::Block{1}) = [Z[xy, B[j]] for j=1:Int(B)]
        getindex(Z::$Typ, xy::StaticVector{2}, JR::BlockOneTo) = mortar([Z[xy,Block(J)] for J = 1:Int(JR[end])])
    end
end

# Normalized is not a MultivariateOrthogonalPolynomial so we mirror the generic indexing
const DiskPoint = Union{StaticVector{2},RadialCoordinate}
getindex(Q::NormalizedZernike, 𝐱::RadialCoordinate, B::Block{1}) = [Q[𝐱, B[j]] for j=1:Int(B)]
getindex(Q::NormalizedZernike, 𝐱::RadialCoordinate, JR::BlockOneTo) = mortar([Q[𝐱,Block(J)] for J = 1:Int(JR[end])])
getindex(Q::NormalizedZernike, 𝐱::DiskPoint, JR::BlockRange{1}) = Q[𝐱, Block.(OneTo(Int(maximum(JR))))][JR]
getindex(Q::NormalizedZernike, 𝐱::DiskPoint, j::Integer) = Q[𝐱, findblockindex(axes(Q,2), j)]
getindex(Q::NormalizedZernike, 𝐱::DiskPoint, jr::AbstractVector{<:Integer}) = Q[𝐱, Block.(OneTo(Int(findblock(axes(Q,2), maximum(jr)))))][jr]
getindex(Q::NormalizedZernike, 𝐱::StaticVector{2}, jr::AbstractUnitRange{Int}) = Q[𝐱, Block.(OneTo(Int(findblock(axes(Q,2), maximum(jr)))))][jr] # disambiguation
getindex(Q::NormalizedZernike, 𝐱::DiskPoint, jr::AbstractVector{<:BlockIndex{1}}) = [Q[𝐱, j] for j in jr]

QuasiArrays.mul(Q::NormalizedZernike, b::AbstractVector) = ApplyQuasiArray(*, Q, BlockedVector(b, (axes(Q,2),)))

###
# Normalization constants
###

"""
    zernikenormalizationconstant(T, n, m, a, b)

gives `c` such that `normalizedzerniker(ℓ, m, a, b, r) == c * zerniker(ℓ, m, a, b, r)` where `n == (ℓ-m) ÷ 2`.
"""
function zernikenormalizationconstant(::Type{T}, n::Integer, m::Integer, a, b) where T
    α, β = convert(T, b), convert(T, m+a)
    # ratio of 2^(α+β+1) to the norm squared of the Jacobi polynomial
    lr = iszero(n) ? loggamma(α+β+2) - loggamma(α+1) - loggamma(β+1) :
                     log(2n+α+β+1) + loggamma(n+α+β+1) + loggamma(n+one(T)) - loggamma(n+α+1) - loggamma(n+β+1)
    sqrt(convert(T,2)^(1-iszero(m)) / convert(T,π) * exp(lr))
end

# the scaling such that Normalized(Z) == Z * Diagonal(ZernikeNormalizationConstant(Z.a, Z.b))
struct ZernikeNormalizationConstant{T} <: AbstractBlockVector{T}
    a::T
    b::T
end

ZernikeNormalizationConstant{T}(Z::Zernike) where T = ZernikeNormalizationConstant{T}(Z.a, Z.b)

axes(::ZernikeNormalizationConstant) = (blockedrange(oneto(∞)),)
copy(c::ZernikeNormalizationConstant) = c

MemoryLayout(::Type{<:ZernikeNormalizationConstant}) = LazyLayout()
Base.BroadcastStyle(::Type{<:ZernikeNormalizationConstant}) = LazyArrayStyle{1}()
Base.BroadcastStyle(::Type{<:Diagonal{<:Any,<:ZernikeNormalizationConstant}}) = LazyArrayStyle{2}()

function getindex(c::ZernikeNormalizationConstant{T}, Kk::BlockIndex{1}) where T
    ℓ, ms = _zernikelm(Kk)
    m = abs(ms)
    zernikenormalizationconstant(T, (ℓ-m) ÷ 2, m, c.a, c.b)
end
getindex(c::ZernikeNormalizationConstant, k::Integer) = c[findblockindex(axes(c,1),k)]
Base.view(c::ZernikeNormalizationConstant, K::Block{1}) = [c[K[j]] for j = 1:Int(K)]

normalizationconstant(Z::Zernike{T}) where T = ZernikeNormalizationConstant{real(T)}(Z)

# the normalization constants laid out as the coefficient matrix of a ModalTrav:
# row i corresponds to n = i-1 and column j to Fourier mode j ÷ 2
_zernikemodalconstant(::Type{T}, (M,N)::NTuple{2,Int}, a, b) where T =
    [zernikenormalizationconstant(T, i-1, j ÷ 2, a, b) for i = 1:M, j = 1:N]

# coefficients in Normalized(Z) to coefficients in Z
_normalized2zernike(Z::Zernike, c::ModalTrav) = ModalTrav(_zernikemodalconstant(real(eltype(Z)), size(c.matrix), Z.a, Z.b) .* c.matrix)
# coefficients in Z to coefficients in Normalized(Z)
_zernike2normalized(Z::Zernike, c::AbstractVector) = c ./ normalizationconstant(Z)[axes(c,1)]
_zernike2normalized(::NormalizedZernike, c::AbstractVector) = c


###
# Jacobi matrices
###
function jacobimatrix(::Val{1}, Q::NormalizedZernike{T}) where T
    Z = Q.P
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

function jacobimatrix(::Val{2}, Q::NormalizedZernike{T}) where T
    Z = Q.P
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

# Normalized(Z) == Z * D so x * Z == Z * D * X * inv(D) where X is the Jacobi matrix of Normalized(Z)
function jacobimatrix(v::Union{Val{1},Val{2}}, Z::Zernike)
    s = Normalized(Z).scaling
    _bandedblockbandeddiagonal(s) * jacobimatrix(v, Normalized(Z)) * _bandedblockbandeddiagonal(inv.(s))
end

# a Diagonal would lose the block-banded structure in the product
_bandedblockbandeddiagonal(d::AbstractVector) = _BandedBlockBandedMatrix(BlockBroadcastArray(hcat, d)', axes(d,1), (0,0), (0,0))

for Inc in (:FirstInclusion, :LastInclusion)
    @eval function Base.broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::$Inc, Q::NormalizedZernike)
        axes(x,1) == axes(Q,1) || throw(DimensionMismatch())
        Q*jacobimatrix(Val($(Inc == :FirstInclusion ? 1 : 2)), Q)
    end
end

###
# Transforms
###

function grid(S::Zernike, B::Block{1})
    T = real(eltype(S))
    N = Int(B) ÷ 2 + 1 # matrix rows
    M = 4N-3 # matrix columns

    r = sinpi.((N .-(0:N-1) .- one(T)/2) ./ (2N))

    # The angular grid:
    θ = (0:M-1)*convert(T,2)/M
    RadialCoordinate.(r, π*θ')
end

grid(Q::NormalizedZernike, B::Block{1}) = grid(Q.P, B)
grid(Q::NormalizedZernike, n::Int) = grid(Q.P, n)
plotgrid(Q::NormalizedZernike, B::Block{1}) = plotgrid(Q.P, B)

_angle(rθ::RadialCoordinate) = rθ.θ

function plotgrid(S::Zernike{T}, B::Block{1}) where T
    N = Int(B) ÷ 2 + 1 # polynomial degree
    g = grid(S, Block(min(2N, MAX_PLOT_BLOCKS))) # double sampling
    θ = [map(_angle,g[1,:]); 0]
    [permutedims(RadialCoordinate.(1,θ));
     g g[:,1];
     permutedims(RadialCoordinate.(0,θ))]
end

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{ZernikeOrNormalized, AbstractVector}}, x) where T
    Z,c = u.args
    B = findblock(axes(Z,2), last(colsupport(c)))
    N = Int(B) ÷ 2 + 1 # polynomial degree
    F = ZernikeITransform{T}(min(2N, MAX_PLOT_BLOCKS), _zernike(Z).a, _zernike(Z).b)
    C = F * _zernike2normalized(Z, c[Block.(OneTo(min(2N, MAX_PLOT_BLOCKS)))]) # transform to grid
    [permutedims(u[x[1,:]]); # evaluate on edge of disk
     C C[:,1];
     fill(u[x[end,1]], 1, size(x,2))] # evaluate at origin and repeat
end

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{Weighted{<:Any,<:ZernikeOrNormalized}, AbstractVector}}, x) where T
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

deduceeltype(T, f) = all(isreal, f) ? T : Complex{T}
*(P::ZernikeTransform{T}, f::AbstractArray) where T = P * convert(Matrix{deduceeltype(T, f)}, f)
*(P::ZernikeTransform{T}, f::Matrix{Complex{T}}) where T = ModalTrav((P.disk2cxf \ (P.analysis * real(f))) + im * (P.disk2cxf \ (P.analysis * imag(f))))
*(P::ZernikeTransform{T}, f::Matrix{T}) where T = ModalTrav(P.disk2cxf \ (P.analysis * f))
*(P::ZernikeITransform, f::AbstractVector) = P.synthesis * (P.disk2cxf * ModalTrav(f).matrix)

inv(P::ZernikeTransform) = ZernikeITransform(P.N, P.disk2cxf, inv(P.analysis))

# transforms for Zernike(a,b) rescale those for Normalized(Zernike(a,b))
struct UnnormalizedZernikeTransform{T, ZZ<:Zernike} <: Plan{T}
    Z::ZZ
    F::ZernikeTransform{T}
end

struct UnnormalizedZernikeITransform{T, ZZ<:Zernike} <: Plan{T}
    Z::ZZ
    F::ZernikeITransform{T}
end

*(P::UnnormalizedZernikeTransform, f::AbstractArray) = _normalized2zernike(P.Z, P.F * f)
*(P::UnnormalizedZernikeITransform, c::AbstractVector) = P.F * _zernike2normalized(P.Z, c)

inv(P::UnnormalizedZernikeTransform) = UnnormalizedZernikeITransform(P.Z, inv(P.F))

plan_transform(Q::NormalizedZernike{T}, (N,)::Tuple{Block{1}}, dims=1) where T = ZernikeTransform{real(T)}(Int(N), Q.P.a, Q.P.b)
plan_transform(Z::Zernike{T}, (N,)::Tuple{Block{1}}, dims=1) where T = UnnormalizedZernikeTransform(Z, ZernikeTransform{real(T)}(Int(N), Z.a, Z.b))
plan_transform(Q::NormalizedZernike, Bs::NTuple{N,Int}, dims=ntuple(identity,Val(N))) where N = plan_transform(Q, findblock.(Ref(axes(Q,2)), Bs), dims)

# use the fast transform rather than re-expanding Normalized(Z) in Z
transform_ldiv(Q::NormalizedZernike, C::AbstractQuasiArray) = ContinuumArrays.transform_ldiv_size(size(Q), Q, C)
transform_ldiv(V::SubQuasiArray{<:Any,2,<:NormalizedZernike,<:Tuple{Inclusion,BlockSlice{BlockOneTo}}}, C::AbstractQuasiArray) = factorize(V) \ C

###
# Fourier modes
#
# In Fourier mode m = 0, 1, … the radial part of Z is r^m times _zernikejacobi(T, Z)[m+1] evaluated at 2r^2-1
# times the constant exp2(_zernikemodeexp(T, Z, m)).
###

_zernikejacobi(::Type{T}, Z::Zernike) where T = Jacobi{T}.(Z.b, Z.a:∞)
_zernikejacobi(::Type{T}, Q::NormalizedZernike) where T = Normalized.(Jacobi{T}.(Q.P.b, Q.P.a:∞))

_zernikemodeexp(::Type{T}, ::Zernike, m) where T = zero(T)
_zernikemodeexp(::Type{T}, Q::NormalizedZernike, m) where T = (m + Q.P.a + Q.P.b + 2 - iszero(m) - log2(convert(T,π)))/2

# ratio of the mode constants of B to those of A
_zernikeratio(::Type{T}, A, B) where T = exp2.(_zernikemodeexp.(T, Ref(B), 0:∞) .- _zernikemodeexp.(T, Ref(A), 0:∞))

##
# Laplacian
###

function laplacian(WZ::Weighted{T,<:ZernikeOrNormalized}; dims...) where T
    Z = _zernike(WZ.P)
    @assert Z.a == 0 && Z.b == 1
    # diagonal so independent of normalization
    WZ.P * ModalInterlace{T}(broadcast(k ->  Diagonal(-cumsum(k:8:∞)), 4:4:∞), (ℵ₀,ℵ₀), (0,0))
end

function laplacian(Z::ZernikeOrNormalized{T}; dims...) where T
    a,b = _zernike(Z).a,_zernike(Z).b
    @assert a == 0
    Z₂ = _zernikesimilar(Z, a, b+2)
    D = Derivative(Inclusion(ChebyshevInterval{T}()))
    Δs = BroadcastVector{AbstractMatrix{T}}((C,B,A,c) -> 8c*(HalfWeighted{:b}(C)\(D*HalfWeighted{:b}(B)))*(B\(D*A)), _zernikejacobi(T, Z₂), Normalized.(Jacobi{T}.(b+1,(a+1):∞)), _zernikejacobi(T, Z), _zernikeratio(real(T), Z₂, Z))
    Δ = ModalInterlace(Δs, (ℵ₀,ℵ₀), (-2,2))
    Z₂ * Δ
end

###
# Fractional Laplacian
###

function abslaplacian(WZ::Weighted{<:Any,<:ZernikeOrNormalized}, α; dims...)
    Z = _zernike(WZ.P)
    @assert Z.a == 0 && Z.b == α
    # diagonal so independent of normalization
    WZ.P * Diagonal(WeightedZernikeFractionalLaplacianDiag{typeof(α)}(α))
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

function \(A::ZernikeOrNormalized, B::ZernikeOrNormalized)
    TV = promote_type(eltype(A), eltype(B))
    A == B && return Eye{TV}((axes(A,2),))
    Z_A, Z_B = _zernike(A), _zernike(B)
    st = Int(Z_A.a - Z_B.a + Z_A.b - Z_B.b)
    ModalInterlace{TV}((_zernikejacobi(TV, A) .\ _zernikejacobi(TV, B)) .* _zernikeratio(real(TV), A, B), (ℵ₀,ℵ₀), (0,2st))
end

function \(A::ZernikeOrNormalized, wB::Weighted{<:Any,<:ZernikeOrNormalized})
    B = wB.P
    TV = promote_type(eltype(A), eltype(B))
    Z_A, Z_B = _zernike(A), _zernike(B)
    A == B && iszero(Z_B.a) && iszero(Z_B.b) && return Eye{TV}((axes(A,2),))
    @assert iszero(Z_B.a)
    # (1-r^2)^b == 2^(-b) * (1-s)^b where s = 2r^2-1
    ModalInterlace{TV}((_zernikejacobi(TV, A) .\ HalfWeighted{:a}.(_zernikejacobi(TV, B))) .* (_zernikeratio(real(TV), A, B) ./ convert(real(TV), 2)^Z_B.b), (ℵ₀,ℵ₀), (2Int(Z_B.b), 2Int(Z_A.a+Z_A.b)))
end


###
# sum
###

function Base._sum(P::ZernikeOrNormalized{T}, dims) where T
    @assert dims == 1
    Z = _zernike(P)
    @assert Z.a == Z.b == 0
    # the first polynomial is constant and the rest integrate to zero
    Hcat(convert(T, π) * P[SVector{2,real(T)}(0,0), 1], Zeros{T}(1,∞))
end