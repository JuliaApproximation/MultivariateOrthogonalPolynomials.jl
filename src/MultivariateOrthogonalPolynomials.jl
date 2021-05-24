module MultivariateOrthogonalPolynomials
using ClassicalOrthogonalPolynomials, FastTransforms, BlockBandedMatrices, BlockArrays, DomainSets, 
      QuasiArrays, StaticArrays, ContinuumArrays, InfiniteArrays, InfiniteLinearAlgebra, 
      LazyArrays, SpecialFunctions, LinearAlgebra, BandedMatrices, LazyBandedMatrices, ArrayLayouts,
      HarmonicOrthogonalPolynomials

import Base: axes, in, ==, *, ^, \, copy, OneTo, getindex, size, oneto
import DomainSets: boundary

import QuasiArrays: LazyQuasiMatrix, LazyQuasiArrayStyle
import ContinuumArrays: @simplify, Weight, grid, plotgrid, TransformFactorization, Expansion

import ArrayLayouts: MemoryLayout, sublayout, sub_materialize
import BlockArrays: block, blockindex, BlockSlice, viewblock
import BlockBandedMatrices: _BandedBlockBandedMatrix, AbstractBandedBlockBandedMatrix, _BandedMatrix, blockbandwidths, subblockbandwidths
import LinearAlgebra: factorize
import LazyArrays: arguments, paddeddata, LazyArrayStyle, LazyLayout
import LazyBandedMatrices: LazyBandedBlockBandedLayout, AbstractBandedBlockBandedLayout, AbstractLazyBandedBlockBandedLayout
import InfiniteArrays: InfiniteCardinal

import ClassicalOrthogonalPolynomials: jacobimatrix, Weighted, orthogonalityweight, HalfWeighted
import HarmonicOrthogonalPolynomials: BivariateOrthogonalPolynomial, MultivariateOrthogonalPolynomial, Plan,
                                          PartialDerivative, BlockOneTo, BlockRange1, interlace

export UnitTriangle, UnitDisk, JacobiTriangle, TriangleWeight, WeightedTriangle, PartialDerivative, Laplacian, 
      MultivariateOrthogonalPolynomial, BivariateOrthogonalPolynomial, Zernike, RadialCoordinate,
      zerniker, zernikez, Weighted, Block, ZernikeWeight, AbsLaplacianPower

include("ModalInterlace.jl")
include("disk.jl")
include("triangle.jl")


end # module
