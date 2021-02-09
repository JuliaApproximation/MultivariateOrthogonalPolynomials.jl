module MultivariateOrthogonalPolynomials
using ClassicalOrthogonalPolynomials, FastTransforms, BlockBandedMatrices, BlockArrays, DomainSets, 
      QuasiArrays, StaticArrays, ContinuumArrays, InfiniteArrays, InfiniteLinearAlgebra, 
      LazyArrays, SpecialFunctions, LinearAlgebra, BandedMatrices, LazyBandedMatrices, ArrayLayouts,
      HarmonicOrthogonalPolynomials

import Base: axes, in, ==, *, ^, \, copy, OneTo, getindex, size
import DomainSets: boundary

import QuasiArrays: LazyQuasiMatrix, LazyQuasiArrayStyle
import ContinuumArrays: @simplify, Weight, grid, TransformFactorization, Expansion

import BlockArrays: block, blockindex, BlockSlice, viewblock
import BlockBandedMatrices: _BandedBlockBandedMatrix
import LinearAlgebra: factorize
import LazyArrays: arguments, paddeddata

import ClassicalOrthogonalPolynomials: jacobimatrix
import HarmonicOrthogonalPolynomials: BivariateOrthogonalPolynomial, MultivariateOrthogonalPolynomial, PartialDerivative, BlockOneTo

export Triangle, JacobiTriangle, TriangleWeight, WeightedTriangle, PartialDerivative, Laplacian, MultivariateOrthogonalPolynomial, BivariateOrthogonalPolynomial

if VERSION < v"1.6-"
      oneto(n) = Base.OneTo(n)
else
      import Base: oneto
end



include("Triangle/Triangle.jl")


end # module
