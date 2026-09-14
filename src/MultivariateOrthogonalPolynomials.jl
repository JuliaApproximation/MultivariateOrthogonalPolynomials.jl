module MultivariateOrthogonalPolynomials
using StaticArrays: iszero
using QuasiArrays: AbstractVector, _getindex
using ClassicalOrthogonalPolynomials, FastTransforms, BlockBandedMatrices, BlockArrays, DomainSets,
      QuasiArrays, StaticArrays, ContinuumArrays, InfiniteArrays, InfiniteLinearAlgebra,
      LazyArrays, SpecialFunctions, LinearAlgebra, BandedMatrices, LazyBandedMatrices, ArrayLayouts,
      HarmonicOrthogonalPolynomials, RecurrenceRelationships

import Base: axes, in, ==, +, -, /, *, ^, \, inv, copy, copyto!, OneTo, getindex, size, oneto, all, resize!, BroadcastStyle, similar, fill!, setindex!, convert, show, summary, diff
import Base.Broadcast: Broadcasted, broadcasted, DefaultArrayStyle
import DomainSets: boundary, EuclideanDomain

import QuasiArrays: LazyQuasiMatrix, LazyQuasiArrayStyle, domain, vec_layout, reshape_layout, pointchoice
import ContinuumArrays: @simplify, Weight, weight, grid, plotgrid, TransformFactorization, ExpansionLayout, plotvalues, unweighted, plan_transform, checkpoints, transform_ldiv, AbstractBasisLayout, basis_axes, Inclusion, grammatrix, weaklaplacian, layout_broadcasted, laplacian, abslaplacian, laplacian_axis, abslaplacian_axis, diff_layout, operatororder, broadcastbasis, KronExpansionLayout

import ArrayLayouts: MemoryLayout, sublayout, sub_materialize
import BlockArrays: block, blockindex, BlockSlice, viewblock, blockcolsupport, AbstractBlockStyle, BlockStyle
import BlockBandedMatrices: _BandedBlockBandedMatrix, AbstractBandedBlockBandedMatrix, _BandedMatrix, blockbandwidths, subblockbandwidths
import LinearAlgebra: factorize
import LazyArrays: arguments, paddeddata, LazyArrayStyle, LazyLayout, PaddedLayout, applylayout, LazyMatrix, ApplyMatrix, ApplyLayout
import LazyBandedMatrices: LazyBandedBlockBandedLayout, AbstractBandedBlockBandedLayout, AbstractLazyBandedBlockBandedLayout, _krontrav_axes, DiagTravLayout, diagtrav, invdiagtrav, ApplyBandedBlockBandedLayout, krontrav
import InfiniteArrays: InfiniteCardinal, OneToInf

import ClassicalOrthogonalPolynomials: jacobimatrix, Weighted, orthogonalityweight, HalfWeighted, WeightedBasis, pad, recurrencecoefficients, clenshaw, weightedgrammatrix, Clenshaw, OPLayout, normalized
import HarmonicOrthogonalPolynomials: BivariateOrthogonalPolynomial, MultivariateOrthogonalPolynomial, Plan,
                                          AngularMomentum, angularmomentum, BlockOneTo, BlockRange1, interlace,
                                          MultivariateOPLayout, AbstractMultivariateOPLayout, MAX_PLOT_BLOCKS

export MultivariateOrthogonalPolynomial, BivariateOrthogonalPolynomial,
       UnitTriangle, UnitDisk,
       JacobiTriangle, TriangleWeight, WeightedTriangle,
       DunklXuDisk, DunklXuDiskWeight, WeightedDunklXuDisk,
       Zernike, ZernikeWeight, zerniker, zernikez,
       AngularMomentum,
       RadialCoordinate, Weighted, Block, jacobimatrix, KronPolynomial, RectPolynomial,
       grammatrix, oneto, coordinates, Laplacian, AbsLaplacian, laplacian, abslaplacian, angularmomentum, weaklaplacian


laplacian_axis(::Inclusion{<:SVector{2}}, A; dims...) = diff(A, (2,0); dims...) + diff(A, (0, 2); dims...)
abslaplacian_axis(::Inclusion{<:SVector{2}}, A; dims...) = -(diff(A, (2,0); dims...) + diff(A, (0, 2); dims...))
coordinates(P::AbstractQuasiArray) = (first.(axes(P,1)), last.(axes(P,1)))
coordinates(P::Domain) = coordinates(Inclusion(P))


include("ModalInterlace.jl")
include("clenshawkron.jl")
include("rect.jl")
include("disk.jl")
include("rectdisk.jl")
include("triangle.jl")


end # module
