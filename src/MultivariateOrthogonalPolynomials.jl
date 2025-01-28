module MultivariateOrthogonalPolynomials
using StaticArrays: iszero
using QuasiArrays: AbstractVector
using ClassicalOrthogonalPolynomials, FastTransforms, BlockBandedMatrices, BlockArrays, DomainSets,
      QuasiArrays, StaticArrays, ContinuumArrays, InfiniteArrays, InfiniteLinearAlgebra,
      LazyArrays, SpecialFunctions, LinearAlgebra, BandedMatrices, LazyBandedMatrices, ArrayLayouts,
      HarmonicOrthogonalPolynomials, RecurrenceRelationships

import Base: axes, in, ==, +, -, /, *, ^, \, copy, copyto!, OneTo, getindex, size, oneto, all, resize!, BroadcastStyle, similar, fill!, setindex!, convert, show, summary, diff
import Base.Broadcast: Broadcasted, broadcasted, DefaultArrayStyle
import DomainSets: boundary, EuclideanDomain

import QuasiArrays: LazyQuasiMatrix, LazyQuasiArrayStyle, domain
import ContinuumArrays: @simplify, Weight, weight, grid, plotgrid, TransformFactorization, ExpansionLayout, plotvalues, unweighted, plan_transform, checkpoints, transform_ldiv, AbstractBasisLayout, basis_axes, Inclusion, grammatrix, weaklaplacian, layout_broadcasted, laplacian, abslaplacian, laplacian_axis, abslaplacian_axis, diff_layout

import ArrayLayouts: MemoryLayout, sublayout, sub_materialize
import BlockArrays: block, blockindex, BlockSlice, viewblock, blockcolsupport, AbstractBlockStyle, BlockStyle
import BlockBandedMatrices: _BandedBlockBandedMatrix, AbstractBandedBlockBandedMatrix, _BandedMatrix, blockbandwidths, subblockbandwidths
import LinearAlgebra: factorize
import LazyArrays: arguments, paddeddata, LazyArrayStyle, LazyLayout, PaddedLayout, applylayout, LazyMatrix, ApplyMatrix
import LazyBandedMatrices: LazyBandedBlockBandedLayout, AbstractBandedBlockBandedLayout, AbstractLazyBandedBlockBandedLayout, _krontrav_axes, DiagTravLayout, invdiagtrav, ApplyBandedBlockBandedLayout, krontrav
import InfiniteArrays: InfiniteCardinal, OneToInf

import ClassicalOrthogonalPolynomials: jacobimatrix, Weighted, orthogonalityweight, HalfWeighted, WeightedBasis, pad, recurrencecoefficients, clenshaw, weightedgrammatrix, Clenshaw
import HarmonicOrthogonalPolynomials: BivariateOrthogonalPolynomial, MultivariateOrthogonalPolynomial, Plan,
                                          AngularMomentum, BlockOneTo, BlockRange1, interlace,
                                          MultivariateOPLayout, AbstractMultivariateOPLayout, MAX_PLOT_BLOCKS

export MultivariateOrthogonalPolynomial, BivariateOrthogonalPolynomial,
       UnitTriangle, UnitDisk,
       JacobiTriangle, TriangleWeight, WeightedTriangle,
       DunklXuDisk, DunklXuDiskWeight, WeightedDunklXuDisk,
       Zernike, ZernikeWeight, zerniker, zernikez,
       AngularMomentum,
       RadialCoordinate, Weighted, Block, jacobimatrix, KronPolynomial, RectPolynomial,
       grammatrix, oneto, coordinates, Laplacian, AbsLaplacian


laplacian_axis(::Inclusion{<:SVector{2}}, A; dims...) = diff(A, (2,0); dims...) + diff(A, (0, 2); dims...)
abslaplacian_axis(::Inclusion{<:SVector{2}}, A; dims...) = -(diff(A, (2,0); dims...) + diff(A, (0, 2); dims...))
coordinates(P) = (first.(axes(P,1)), last.(axes(P,1)))

function diff_layout(::AbstractBasisLayout, a, (k,j)::NTuple{2,Int}; dims...)
    (k < 0 || j < 0) && throw(ArgumentError("order must be non-negative"))
    k == j == 0 && return a
    ((k,j) == (1,0) || (k,j) == (0,1)) && return diff(a, Val((k,j)); dims...)
    k ≥ j && diff(diff(a, (1,0)), (k-1,j))
    diff(diff(a, (0,1)), (k,j-1))
end


include("ModalInterlace.jl")
include("clenshawkron.jl")
include("rect.jl")
include("disk.jl")
include("rectdisk.jl")
include("triangle.jl")


end # module
