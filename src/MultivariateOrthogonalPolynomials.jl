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


include("Triangle/Triangle.jl")


end # module
