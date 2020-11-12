module MultivariateOrthogonalPolynomials
using OrthogonalPolynomialsQuasi, FastTransforms, BlockBandedMatrices, BlockArrays, DomainSets, 
      QuasiArrays, StaticArrays, ContinuumArrays, InfiniteArrays, InfiniteLinearAlgebra, 
      LazyArrays, SpecialFunctions, LinearAlgebra, BandedMatrices, LazyBandedMatrices

import Base: axes, in, ==, *, ^, \, copy, OneTo, getindex
import DomainSets: boundary

import QuasiArrays: LazyQuasiMatrix, LazyQuasiArrayStyle
import ContinuumArrays: @simplify, Weight, grid, TransformFactorization

import BlockArrays: block, blockindex, BlockSlice
import BlockBandedMatrices: _BandedBlockBandedMatrix
import LinearAlgebra: factorize

export Triangle, JacobiTriangle, TriangleWeight, WeightedTriangle, PartialDerivative, Laplacian

#########
# PartialDerivative{k}
# takes a partial derivative in the k-th index.
#########


struct PartialDerivative{k,T,D} <: LazyQuasiMatrix{T}
    axis::Inclusion{T,D}
end

PartialDerivative{k,T}(axis::Inclusion{<:Any,D}) where {k,T,D} = PartialDerivative{k,T,D}(axis)
PartialDerivative{k,T}(domain) where {k,T} = PartialDerivative{k,T}(Inclusion(domain))
PartialDerivative{k}(axis) where k = PartialDerivative{k,eltype(axis)}(axis)

axes(D::PartialDerivative) = (D.axis, D.axis)
==(a::PartialDerivative{k}, b::PartialDerivative{k}) where k = a.axis == b.axis
copy(D::PartialDerivative{k}) where k = PartialDerivative{k}(copy(D.axis))

struct Laplacian{T,D} <: LazyQuasiMatrix{T}
    axis::Inclusion{T,D}
end

Laplacian{T}(axis::Inclusion{<:Any,D}) where {T,D} = Laplacian{T,D}(axis)
Laplacian{T}(domain) where T = Laplacian{T}(Inclusion(domain))
Laplacian(axis) = Laplacian{eltype(axis)}(axis)

axes(D::Laplacian) = (D.axis, D.axis)
==(a::Laplacian, b::Laplacian) = a.axis == b.axis
copy(D::Laplacian) = Laplacian(copy(D.axis), D.k)

^(D::PartialDerivative, k::Integer) = ApplyQuasiArray(^, D, k)
^(D::Laplacian, k::Integer) = ApplyQuasiArray(^, D, k)


abstract type MultivariateOrthogonalPolynomial{T,D} <: Basis{T} end
const BivariateOrthogonalPolynomial{T} = MultivariateOrthogonalPolynomial{T,2}
const BlockOneTo = BlockRange{1,Tuple{OneTo{Int}}}


getindex(P::MultivariateOrthogonalPolynomial{<:Any,D}, xy::SVector{D}, JR::BlockOneTo) where D = 
    error("Overload")
getindex(P::MultivariateOrthogonalPolynomial{<:Any,D}, xy::SVector{D}, J::Block{1}) where D = P[xy, Block.(OneTo(Int(J)))][J]
getindex(P::MultivariateOrthogonalPolynomial{<:Any,D}, xy::SVector{D}, JR::BlockRange{1}) where D = P[xy, Block.(OneTo(Int(maximum(JR))))][JR]
getindex(P::MultivariateOrthogonalPolynomial{<:Any,D}, xy::SVector{D}, Jj::BlockIndex{1}) where D = P[xy, block(Jj)][blockindex(Jj)]
getindex(P::MultivariateOrthogonalPolynomial{<:Any,D}, xy::SVector{D}, j::Integer) where D = P[xy, findblockindex(axes(P,2), j)]

const FirstInclusion = BroadcastQuasiVector{<:Any, typeof(first), <:Tuple{Inclusion}}
const LastInclusion = BroadcastQuasiVector{<:Any, typeof(last), <:Tuple{Inclusion}}

function Base.broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::FirstInclusion, P::BivariateOrthogonalPolynomial)
    axes(x,1) == axes(P,1) || throw(DimensionMismatch())
    P*jacobimatrix(Val(1), P)
end

function Base.broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::LastInclusion, P::BivariateOrthogonalPolynomial)
    axes(x,1) == axes(P,1) || throw(DimensionMismatch())
    P*jacobimatrix(Val(2), P)
end

"""
   forwardrecurrence!(v, A, B, C, x, y)

evaluates the bivaraite orthogonal polynomials at points `(x,y)` ,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
matrices, e.g., A[N] is (N+1) x N
"""
function forwardrecurrence!(v::AbstractBlockVector{T}, A::AbstractVector, B::AbstractVector, C::AbstractVector, x, y) where T
    N = blocklength(v)
    N == 0 && return v
    length(A)+1 ≥ N && length(B)+1 ≥ N && length(C)+1 ≥ N || throw(ArgumentError("A, B, C must contain at least $(N-1) entries"))
    v[Block(1)] .= one(T)
    N = 1 && return v
    p_1 = view(v,Block(2))
    p_0 = view(v,Block(1))
    mul!(p_1, B[1], p_0)
    muladd!(one(T), A[1], [x; y], one(T), p_1)

    @inbounds for n = 2:N-1
        
        p1,p0 = _forwardrecurrence_next(n, A, B, C, x, p0, p1),p1
        v[n+1] = p1
    end    

    p1 = convert(T, N == 1 ? p0 : muladd(A[1],x,B[1])*p0) # avoid accessing A[1]/B[1] if empty
    _forwardrecurrence!(v, A, B, C, x, convert(T, p0), p1)
end


forwardrecurrence(N::Integer, A::AbstractVector, B::AbstractVector, C::AbstractVector, x, y) =
    forwardrecurrence!(PseudoBlockVector{promote_type(eltype(eltype(A)),eltype(eltype(B)),eltype(eltype(C)),typeof(x),typeof(y))}(undef, 1:N), A, B, C, x, y)


include("Triangle/Triangle.jl")


end # module
