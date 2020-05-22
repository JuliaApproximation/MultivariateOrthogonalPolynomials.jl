module MultivariateOrthogonalPolynomials
using OrthogonalPolynomialsQuasi, FastTransforms, BlockBandedMatrices, BlockArrays, DomainSets, 
      QuasiArrays, StaticArrays, ContinuumArrays, InfiniteArrays, LazyArrays, SpecialFunctions, LinearAlgebra

import Base: axes, in, ==, *
import DomainSets: boundary

import QuasiArrays: LazyQuasiMatrix
import ContinuumArrays: @simplify, Weight

import BlockBandedMatrices: _BandedBlockBandedMatrix

export Triangle, JacobiTriangle, TriangleWeight, WeightedTriangle, PartialDerivative

#########
# PartialDerivative
#########


struct PartialDerivative{k,T,D} <: LazyQuasiMatrix{T}
    axis::Inclusion{T,D}
end

PartialDerivative{k,T}(axis::Inclusion{<:Any,D}) where {k,T,D} = PartialDerivative{k,T,D}(axis)
PartialDerivative{k,T}(domain) where {k,T} = PartialDerivative{k,T}(Inclusion(domain))
PartialDerivative{k}(axis) where k = PartialDerivative{k,eltype(axis)}(axis)

axes(D::PartialDerivative) = (D.axis, D.axis)
==(a::PartialDerivative, b::PartialDerivative) = a.axis == b.axis && a.k == b.k
copy(D::PartialDerivative) = PartialDerivative(copy(D.axis), D.k)

include("Triangle/Triangle.jl")


end # module
