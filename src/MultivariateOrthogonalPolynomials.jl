module MultivariateOrthogonalPolynomials
using OrthogonalPolynomialsQuasi, FastTransforms, BlockBandedMatrices, BlockArrays, DomainSets, 
      QuasiArrays, StaticArrays, ContinuumArrays, InfiniteArrays, InfiniteLinearAlgebra, 
      LazyArrays, SpecialFunctions, LinearAlgebra, BandedMatrices, LazyBandedMatrices, ArrayLayouts,
      SphericalHarmonics

import Base: axes, in, ==, *, ^, \, copy, OneTo, getindex, size
import DomainSets: boundary

import QuasiArrays: LazyQuasiMatrix, LazyQuasiArrayStyle
import ContinuumArrays: @simplify, Weight, grid, TransformFactorization, Expansion

import BlockArrays: block, blockindex, BlockSlice, viewblock
import BlockBandedMatrices: _BandedBlockBandedMatrix
import LinearAlgebra: factorize
import LazyArrays: arguments, paddeddata

import OrthogonalPolynomialsQuasi: jacobimatrix

export Triangle, JacobiTriangle, TriangleWeight, WeightedTriangle, PartialDerivative, Laplacian, MultivariateOrthogonalPolynomial, BivariateOrthogonalPolynomial




include("Triangle/Triangle.jl")


end # module
