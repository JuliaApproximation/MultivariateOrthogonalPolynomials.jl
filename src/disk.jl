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


abstract type AbstractZernike{T} <: BivariateOrthogonalPolynomial{T} end

"""
    Zernike(a, b)

is a quasi-matrix orthogonal `r^(2a) * (1-r^2)^b`. The polynomials are not normalized:
the entry of degree `ℓ` and Fourier mode `m` is

    r^|m| * jacobip((ℓ-|m|) ÷ 2, b, |m|+a, 2r^2-1) * (signbit(m) ? sin(|m|*θ) : cos(|m|*θ))

Use `Normalized(Zernike(a, b))` for the orthonormal polynomials.
"""
struct Zernike{T} <: AbstractZernike{T}
    a::T
    b::T
    Zernike{T}(a::T, b::T) where T = new{T}(a, b)
end
Zernike{T}(a, b) where T = Zernike{T}(convert(T,a), convert(T,b))
Zernike(a::T, b::V) where {T,V} = Zernike{float(promote_type(T,V))}(a, b)
Zernike{T}(b) where T = Zernike{T}(zero(b), b)
Zernike{T}() where T = Zernike{T}(zero(T))

AbstractQuasiArray{T}(Z::Zernike) where T = Zernike{T}(Z.a, Z.b)
AbstractQuasiMatrix{T}(Z::Zernike) where T = Zernike{T}(Z.a, Z.b)

"""
    Zernike(b)

is a quasi-matrix orthogonal `(1-r^2)^b`
"""
Zernike(b) = Zernike(zero(b), b)
Zernike() = Zernike{Float64}()

"""
    ComplexZernike(a, b)

is a quasi-matrix orthogonal `r^(2a) * (1-r^2)^b` with complex Fourier modes. It is ordered the same
as `Zernike(a, b)` but with `sin(|m|*θ)` replaced by `exp(-im*|m|*θ)` and `cos(|m|*θ)` by `exp(im*|m|*θ)`,
that is, the entry of degree `ℓ` and Fourier mode `m` is

    r^|m| * jacobip((ℓ-|m|) ÷ 2, b, |m|+a, 2r^2-1) * exp(im*m*θ)

Use `Normalized(ComplexZernike(a, b))` for the orthonormal polynomials. The type parameter `T` is the (complex) element type.
"""
struct ComplexZernike{T,V} <: AbstractZernike{T}
    a::V
    b::V
    ComplexZernike{T,V}(a::V, b::V) where {T,V} = new{T,V}(a, b)
end
ComplexZernike{T}(a, b) where T = ComplexZernike{T,real(T)}(convert(real(T),a), convert(real(T),b))
ComplexZernike(a::T, b::V) where {T,V} = ComplexZernike{complex(float(promote_type(T,V)))}(a, b)
ComplexZernike{T}(b) where T = ComplexZernike{T}(zero(b), b)
ComplexZernike{T}() where T = ComplexZernike{T}(zero(real(T)))
ComplexZernike(b) = ComplexZernike(zero(b), b)
ComplexZernike() = ComplexZernike{ComplexF64}()

AbstractQuasiArray{T}(Z::ComplexZernike) where T = ComplexZernike{T}(Z.a, Z.b)
AbstractQuasiMatrix{T}(Z::ComplexZernike) where T = ComplexZernike{T}(Z.a, Z.b)

# Zernike-type basis of the same kind as Z with parameters a and b
_zernike(::Zernike{T}, a, b) where T = Zernike{T}(a, b)
_zernike(::ComplexZernike{T}, a, b) where T = ComplexZernike{T}(a, b)

axes(P::AbstractZernike{T}) where T = (Inclusion(UnitDisk{real(T)}()),blockedrange(oneto(∞)))

==(w::Zernike, v::Zernike) = w.a == v.a && w.b == v.b
==(w::ComplexZernike, v::ComplexZernike) = w.a == v.a && w.b == v.b
==(::Zernike, ::ComplexZernike) = false
==(::ComplexZernike, ::Zernike) = false

copy(A::AbstractZernike) = A

show(io::IO, P::AbstractZernike) = summary(io, P)
summary(io::IO, P::Zernike) = print(io, "Zernike($(P.a), $(P.b))")
summary(io::IO, P::ComplexZernike) = print(io, "ComplexZernike($(P.a), $(P.b))")

orthogonalityweight(Z::AbstractZernike) = ZernikeWeight(Z.a, Z.b)

basis_axes(::Inclusion{<:Any,<:UnitDisk}, v) = Zernike()

###
# Evaluation
###

zerniker(ℓ, m, a, b, r::T) where T = r^m * jacobip((ℓ-m) ÷ 2, b, m+a, 2r^2-1)
zerniker(ℓ, m, b, r) = zerniker(ℓ, m, zero(b), b, r)
zerniker(ℓ, m, r) = zerniker(ℓ, m, zero(r), r)

normalizedzerniker(ℓ, m, a, b, r::T) where T = sqrt(convert(T,2)^(m+a+b+2-iszero(m))/π) * r^m * normalizedjacobip((ℓ-m) ÷ 2, b, m+a, 2r^2-1)
normalizedzerniker(ℓ, m, b, r) = normalizedzerniker(ℓ, m, zero(b), b, r)
normalizedzerniker(ℓ, m, r) = normalizedzerniker(ℓ, m, zero(r), r)

# radial part of Normalized(ComplexZernike), which differs from normalizedzerniker as the
# complex Fourier modes have the same norm for all m
normalizedcomplexzerniker(ℓ, m, a, b, r::T) where T = sqrt(convert(T,2)^(m+a+b+1)/π) * r^m * normalizedjacobip((ℓ-m) ÷ 2, b, m+a, 2r^2-1)

_zernikeangular(ms, θ) = (m = abs(ms); signbit(ms) ? sin(m*θ) : cos(m*θ))
_complexzernikeangular(ms, θ) = cis(ms*θ)

_zernikez(radial, angular, ℓ, ms, a, b, rθ::RadialCoordinate) = radial(ℓ, abs(ms), a, b, rθ.r) * angular(ms, rθ.θ)

for (z, radial, angular) in ((:zernikez, :zerniker, :_zernikeangular), (:normalizedzernikez, :normalizedzerniker, :_zernikeangular),
                             (:complexzernikez, :zerniker, :_complexzernikeangular), (:normalizedcomplexzernikez, :normalizedcomplexzerniker, :_complexzernikeangular))
    @eval begin
        $z(ℓ, ms, a, b, rθ::RadialCoordinate) = _zernikez($radial, $angular, ℓ, ms, a, b, rθ)
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

const NormalizedZernike{T} = Normalized{T,Zernike{T}}
const NormalizedComplexZernike{T} = Normalized{T,<:ComplexZernike{T}}

getindex(Z::Zernike, rθ::RadialCoordinate, B::BlockIndex{1}) = zernikez(_zernikelm(B)..., Z.a, Z.b, rθ)
getindex(Z::ComplexZernike, rθ::RadialCoordinate, B::BlockIndex{1}) = complexzernikez(_zernikelm(B)..., Z.a, Z.b, rθ)
getindex(Q::NormalizedZernike, rθ::RadialCoordinate, B::BlockIndex{1}) = normalizedzernikez(_zernikelm(B)..., Q.P.a, Q.P.b, rθ)
getindex(Q::NormalizedComplexZernike, rθ::RadialCoordinate, B::BlockIndex{1}) = normalizedcomplexzernikez(_zernikelm(B)..., Q.P.a, Q.P.b, rθ)
getindex(Q::Union{NormalizedZernike,NormalizedComplexZernike}, xy::StaticVector{2}, B::BlockIndex{1}) = Q[RadialCoordinate(xy), B]

getindex(Z::AbstractZernike, xy::StaticVector{2}, B::BlockIndex{1}) = Z[RadialCoordinate(xy), B]
getindex(Z::AbstractZernike, xy::StaticVector{2}, B::Block{1}) = [Z[xy, B[j]] for j=1:Int(B)]
getindex(Z::AbstractZernike, xy::StaticVector{2}, JR::BlockOneTo) = mortar([Z[xy,Block(J)] for J = 1:Int(JR[end])])

###
# Normalization constants
###

"""
    zernikenormalizationconstant(T, n, m, a, b)

gives `c` such that `normalizedzerniker(ℓ, m, a, b, r) == c * zerniker(ℓ, m, a, b, r)` where `n == (ℓ-m) ÷ 2`.
"""
zernikenormalizationconstant(::Type{T}, n::Integer, m::Integer, a, b) where T =
    sqrt(convert(T,2)^(1-iszero(m))) * complexzernikenormalizationconstant(T, n, m, a, b)

"""
    complexzernikenormalizationconstant(T, n, m, a, b)

gives `c` such that `normalizedcomplexzerniker(ℓ, m, a, b, r) == c * zerniker(ℓ, m, a, b, r)` where `n == (ℓ-m) ÷ 2`.
"""
function complexzernikenormalizationconstant(::Type{T}, n::Integer, m::Integer, a, b) where T
    α, β = convert(T, b), convert(T, m+a)
    # ratio of 2^(α+β+1) to the norm squared of the Jacobi polynomial
    lr = iszero(n) ? loggamma(α+β+2) - loggamma(α+1) - loggamma(β+1) :
                     log(2n+α+β+1) + loggamma(n+α+β+1) + loggamma(n+one(T)) - loggamma(n+α+1) - loggamma(n+β+1)
    sqrt(exp(lr) / convert(T,π))
end

abstract type AbstractZernikeNormalizationConstant{T} <: AbstractBlockVector{T} end

# the scaling such that Normalized(Z) == Z * Diagonal(ZernikeNormalizationConstant(Z.a, Z.b))
struct ZernikeNormalizationConstant{T} <: AbstractZernikeNormalizationConstant{T}
    a::T
    b::T
end

# the scaling such that Normalized(Z) == Z * Diagonal(ComplexZernikeNormalizationConstant(Z.a, Z.b)) for Z::ComplexZernike
struct ComplexZernikeNormalizationConstant{T} <: AbstractZernikeNormalizationConstant{T}
    a::T
    b::T
end

ZernikeNormalizationConstant{T}(Z::Zernike) where T = ZernikeNormalizationConstant{T}(Z.a, Z.b)
ComplexZernikeNormalizationConstant{T}(Z::ComplexZernike) where T = ComplexZernikeNormalizationConstant{T}(Z.a, Z.b)

_zernikenormalizationconstant(::ZernikeNormalizationConstant, T, n, m, a, b) = zernikenormalizationconstant(T, n, m, a, b)
_zernikenormalizationconstant(::ComplexZernikeNormalizationConstant, T, n, m, a, b) = complexzernikenormalizationconstant(T, n, m, a, b)

axes(::AbstractZernikeNormalizationConstant) = (blockedrange(oneto(∞)),)
copy(c::AbstractZernikeNormalizationConstant) = c

MemoryLayout(::Type{<:AbstractZernikeNormalizationConstant}) = LazyLayout()
Base.BroadcastStyle(::Type{<:AbstractZernikeNormalizationConstant}) = LazyArrayStyle{1}()
Base.BroadcastStyle(::Type{<:Diagonal{<:Any,<:AbstractZernikeNormalizationConstant}}) = LazyArrayStyle{2}()

function getindex(c::AbstractZernikeNormalizationConstant{T}, Kk::BlockIndex{1}) where T
    ℓ, ms = _zernikelm(Kk)
    m = abs(ms)
    _zernikenormalizationconstant(c, T, (ℓ-m) ÷ 2, m, c.a, c.b)
end
getindex(c::AbstractZernikeNormalizationConstant, k::Integer) = c[findblockindex(axes(c,1),k)]
Base.view(c::AbstractZernikeNormalizationConstant, K::Block{1}) = [c[K[j]] for j = 1:Int(K)]

normalizationconstant(Z::Zernike{T}) where T = ZernikeNormalizationConstant{real(T)}(Z)
normalizationconstant(Z::ComplexZernike{T}) where T = ComplexZernikeNormalizationConstant{real(T)}(Z)

# the normalization constants laid out as the coefficient matrix of a ModalTrav:
# row i corresponds to n = i-1 and column j to Fourier mode j ÷ 2
_zernikemodalconstant(::Type{T}, (M,N)::NTuple{2,Int}, a, b) where T =
    [zernikenormalizationconstant(T, i-1, j ÷ 2, a, b) for i = 1:M, j = 1:N]

# coefficients in Normalized(Z) to coefficients in Z
_normalized2zernike(Z::Zernike, c::ModalTrav) = ModalTrav(_zernikemodalconstant(real(eltype(Z)), size(c.matrix), Z.a, Z.b) .* c.matrix)
# coefficients in Z to coefficients in Normalized(Z)
_zernike2normalized(Z::Zernike, c::AbstractVector) = c ./ normalizationconstant(Z)[axes(c,1)]

###
# Zernike <-> ComplexZernike
###

# Map coefficients of sin(m*θ) and cos(m*θ), stored in columns 2m and 2m+1 of a ModalTrav,
# to α*(c + im*s) and α*(c - im*s), the coefficients of exp(-im*m*θ) and exp(im*m*θ). As
#   s*sin(m*θ) + c*cos(m*θ) == (c + im*s)/2 * exp(-im*m*θ) + (c - im*s)/2 * exp(im*m*θ)
# α == 1/2 for Zernike to ComplexZernike and α == 1/sqrt(2) for their normalized counterparts.
function _zernike2complex(c::ModalTrav, α)
    A = c.matrix
    B = similar(A, complex(promote_type(eltype(A), typeof(α))))
    B[:,1] .= view(A,:,1)
    for j = 2:2:size(A,2)
        S, C = view(A,:,j), view(A,:,j+1)
        B[:,j] .= α .* (C .+ im .* S)
        B[:,j+1] .= α .* (C .- im .* S)
    end
    ModalTrav(B)
end

# The block-diagonal matrix that mixes the entries with Fourier modes -|m| and |m| in each block, which
# are adjacent with the negative mode first. A column with a negative mode has diagonal entry d₋ and
# sub-diagonal entry l₋, a column with a positive mode has super-diagonal entry u₊ and diagonal entry d₊,
# and the m == 0 entries are left unchanged.
function _zernikemixing(::Type{T}, d₋, l₋, u₊, d₊) where T
    k = mortar(Base.OneTo.(oneto(∞)))     # k counts the entry in a block
    n = mortar(Fill.(oneto(∞),oneto(∞)))  # n counts the block number which corresponds to the order (+1)
    neg = isodd.(k .+ n)                  # entries with negative Fourier mode
    pos = iseven.(k .+ n) .* ((k .> 1) .| iseven.(n)) # entries with positive Fourier mode
    # the bands are conjugated as they are stored as an adjoint
    # (copying a transpose of a BlockBroadcastArray conjugates its entries)
    du = pos .* conj(convert(T, u₊))
    d = neg .* conj(convert(T, d₋)) .+ pos .* conj(convert(T, d₊)) .+ (1 .- neg .- pos) .* one(T)
    dl = neg .* conj(convert(T, l₋))
    _BandedBlockBandedMatrix(BlockBroadcastArray(hcat, du, d, dl)', axes(n,1), (0,0), (1,1))
end

# sin(m*θ) == (im*exp(-im*m*θ) - im*exp(im*m*θ))/2 and cos(m*θ) == (exp(-im*m*θ) + exp(im*m*θ))/2
_complexzernike2zernike(::Type{T}, α) where T = _zernikemixing(T, im*α, -im*α, α, α)
# exp(-im*m*θ) == cos(m*θ) - im*sin(m*θ) and exp(im*m*θ) == cos(m*θ) + im*sin(m*θ)
_zernike2complexzernike(::Type{T}, β) where T = _zernikemixing(T, -im*β, β, im*β, β)

function \(A::ComplexZernike{T}, B::Zernike{V}) where {T,V}
    TV = promote_type(T,V)
    M = _complexzernike2zernike(TV, one(real(TV))/2)
    (A.a == B.a && A.b == B.b) ? M : (A \ ComplexZernike{complex(TV)}(B.a, B.b)) * M
end

function \(A::Zernike{T}, B::ComplexZernike{V}) where {T,V}
    TV = promote_type(T,V)
    M = _zernike2complexzernike(TV, one(real(TV)))
    (A.a == B.a && A.b == B.b) ? M : (A \ Zernike{real(TV)}(B.a, B.b)) * M
end


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

# Normalized(ComplexZernike) == Normalized(Zernike) * U' for the unitary U == Normalized(ComplexZernike) \ Normalized(Zernike)
function jacobimatrix(v::Union{Val{1},Val{2}}, Q::NormalizedComplexZernike{T}) where T
    Z = Q.P
    β = inv(sqrt(convert(real(T),2)))
    _complexzernike2zernike(T, β) * jacobimatrix(v, Normalized(Zernike{real(T)}(Z.a, Z.b))) * _zernike2complexzernike(T, β)
end

function jacobimatrix(v::Union{Val{1},Val{2}}, Z::ComplexZernike{T}) where T
    R = Zernike{real(T)}(Z.a, Z.b)
    (Z \ R) * jacobimatrix(v, R) * (R \ Z)
end

# a Diagonal would lose the block-banded structure in the product
_bandedblockbandeddiagonal(d::AbstractVector) = _BandedBlockBandedMatrix(BlockBroadcastArray(hcat, d)', axes(d,1), (0,0), (0,0))

###
# Transforms
###

function grid(S::AbstractZernike, B::Block{1})
    T = real(eltype(S))
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
    C = F * _zernike2normalized(Z, c[Block.(OneTo(min(2N, MAX_PLOT_BLOCKS)))]) # transform to grid
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

deduceeltype(T, f) = all(isreal, f) ? T : Complex{T}
*(P::ZernikeTransform{T}, f::AbstractArray) where T = P * convert(Matrix{deduceeltype(T, f)}, f)
*(P::ZernikeTransform{T}, f::Matrix{Complex{T}}) where T = ModalTrav((P.disk2cxf \ (P.analysis * real(f))) + im * (P.disk2cxf \ (P.analysis * imag(f))))
*(P::ZernikeTransform{T}, f::Matrix{T}) where T = ModalTrav(P.disk2cxf \ (P.analysis * f))
*(P::ZernikeITransform, f::AbstractVector) = P.synthesis * (P.disk2cxf * ModalTrav(f).matrix)

inv(P::ZernikeTransform) = ZernikeITransform(P.N, P.disk2cxf, inv(P.analysis))

plan_transform(Q::NormalizedZernike{T}, (N,)::Tuple{Block{1}}, dims=1) where T = ZernikeTransform{real(T)}(Int(N), Q.P.a, Q.P.b)
# transforms for Zernike(a,b) rescale those for Normalized(Zernike(a,b))
plan_transform(Z::Zernike{T}, (N,)::Tuple{Block{1}}, dims=1) where T = ApplyPlan(Base.Fix1(_normalized2zernike, Z), ZernikeTransform{real(T)}(Int(N), Z.a, Z.b))
# transforms for ComplexZernike(a,b) mix the Fourier modes of those for Zernike(a,b)
plan_transform(Q::NormalizedComplexZernike{T}, (N,)::Tuple{Block{1}}, dims=1) where T =
    ApplyPlan(Base.Fix2(_zernike2complex, inv(sqrt(convert(real(T),2)))), ZernikeTransform{real(T)}(Int(N), Q.P.a, Q.P.b))
plan_transform(Z::ComplexZernike{T}, (N,)::Tuple{Block{1}}, dims=1) where T =
    ApplyPlan(Base.Fix2(_zernike2complex, one(real(T))/2) ∘ Base.Fix1(_normalized2zernike, Zernike{real(T)}(Z.a, Z.b)), ZernikeTransform{real(T)}(Int(N), Z.a, Z.b))

##
# Laplacian
###

# The operators below act on each Fourier mode m via the radial part, which is the same for
# sin(|m|*θ) and cos(|m|*θ) as for exp(-im*|m|*θ) and exp(im*|m|*θ), so are shared by Zernike and ComplexZernike.

function laplacian(WZ::Weighted{T,<:AbstractZernike}; dims...) where T
    @assert WZ.P.a == 0 && WZ.P.b == 1
    WZ.P * ModalInterlace{T}(broadcast(k ->  Diagonal(-cumsum(k:8:∞)), 4:4:∞), (ℵ₀,ℵ₀), (0,0))
end

function laplacian(Z::AbstractZernike{T}; dims...) where T
    a,b = Z.a,Z.b
    @assert a == 0
    R = real(T)
    D = Derivative(Inclusion(ChebyshevInterval{R}()))
    Δs = BroadcastVector{AbstractMatrix{R}}((C,B,A) -> 8(HalfWeighted{:b}(C)\(D*HalfWeighted{:b}(B)))*(B\(D*A)), Jacobi{R}.(b+2,a:∞), Jacobi{R}.(b+1,(a+1):∞), Jacobi{R}.(b,a:∞))
    Δ = ModalInterlace(Δs, (ℵ₀,ℵ₀), (-2,2))
    _zernike(Z, a, b+2) * Δ
end

###
# Fractional Laplacian
###

function abslaplacian(WZ::Weighted{<:Any,<:AbstractZernike}, α; dims...)
    @assert WZ.P.a == 0 && WZ.P.b == α
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

function _zernikeconversion(::Type{TV}, A::AbstractZernike, B::AbstractZernike) where TV
    R = real(TV)
    A == B && return Eye{TV}((axes(A,2),))
    st = Int(A.a - B.a + A.b - B.b)
    ModalInterlace{TV}(Jacobi{R}.(A.b,A.a:∞) .\ Jacobi{R}.(B.b,B.a:∞), (ℵ₀,ℵ₀), (0,2st))
end

# the conversions between ComplexZernike act the same on each Fourier mode so are real
\(A::Zernike{T}, B::Zernike{V}) where {T,V} = _zernikeconversion(promote_type(T,V), A, B)
\(A::ComplexZernike{T}, B::ComplexZernike{V}) where {T,V} = _zernikeconversion(real(promote_type(T,V)), A, B)

function _zernikelowering(::Type{TV}, A::AbstractZernike, B::AbstractZernike) where TV
    R = real(TV)
    A.a == B.a == A.b == B.b == 0 && return Eye{TV}((axes(A,2),))
    @assert iszero(B.a)
    # (1-r^2)^b == 2^(-b) * (1-s)^b where s = 2r^2-1
    ModalInterlace{TV}((Jacobi{R}.(A.b, A.a:∞) .\ HalfWeighted{:a}.(Jacobi{R}.(B.b, B.a:∞))) ./ convert(R, 2)^B.b, (ℵ₀,ℵ₀), (2Int(B.b), 2Int(A.a+A.b)))
end

\(A::Zernike{T}, wB::Weighted{V,<:Zernike}) where {T,V} = _zernikelowering(promote_type(T,V), A, wB.P)
\(A::ComplexZernike{T}, wB::Weighted{V,<:ComplexZernike}) where {T,V} = _zernikelowering(real(promote_type(T,V)), A, wB.P)


###
# sum
###

function Base._sum(P::AbstractZernike{T}, dims) where T
    @assert dims == 1
    @assert P.a == P.b == 0
    Hcat(convert(T, π), Zeros{T}(1,∞))
end
