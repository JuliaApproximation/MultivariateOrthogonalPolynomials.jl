module MultivariateOrthogonalPolynomials
using Base, RecipesBase, ApproxFun, BandedMatrices, BlockArrays,
    FastTransforms, FastGaussQuadrature, StaticArrays, FillArrays,
    LinearAlgebra, Libdl, SpecialFunctions, LazyArrays, InfiniteArrays

# package code goes here
import Base: values,getindex,setindex!,*, +, -, ==,<,<=,>,
                >=,/,^,\,∪,transpose, in, convert, issubset


import BandedMatrices: inbands_getindex, inbands_setindex!

import BlockArrays: blocksizes, BlockSizes, getblock, global2blockindex

# ApproxFun general import
import ApproxFunBase: BandedMatrix, blocksize,
                  linesum,complexlength, BandedBlockBandedMatrix,
                  real, eps, isapproxinteger, FiniteRange, DFunction,
                  TransformPlan, ITransformPlan

# Domains import
import ApproxFunBase: fromcanonical, tocanonical, domainscompatible

# Operator import
import ApproxFunBase:    bandwidths,SpaceOperator, ConversionWrapper, DerivativeWrapper,
                  rangespace, domainspace, InterlaceOperator,
                  promotedomainspace,  CalculusOperator, interlace, Multiplication,
                   choosedomainspace, SubOperator, ZeroOperator,
                    Dirichlet, DirichletWrapper, Neumann, Laplacian, ConstantTimesOperator, Conversion,
                    Derivative, ConcreteMultiplication, ConcreteConversion, ConcreteLaplacian,
                    ConcreteDerivative, TimesOperator, MultiplicationWrapper, TridiagonalOperator


# Spaces import
import ApproxFunBase:   ConstantSpace, NoSpace, prectype,
                    SumSpace,PiecewiseSpace, ArraySpace, @containsconstants,
                    UnsetSpace, canonicalspace, canonicaldomain, domain, evaluate,
                    AnyDomain, plan_transform,plan_itransform,
                    transform,itransform,transform!,itransform!,
                    isambiguous, fromcanonical, tocanonical, checkpoints, ∂, spacescompatible,
                    mappoint, UnivariateSpace, setdomain, setcanonicaldomain, canonicaldomain,
                    Space, points, space, conversion_rule, maxspace_rule,
                    union_rule, coefficients, RealUnivariateSpace, PiecewiseSegment, rangetype, cfstype

# Multivariate import
import ApproxFunBase: Domain2d, DirectSumSpace, AbstractProductSpace, factor,
                    BivariateFun,  ProductFun, LowRankFun, lap, columnspace,
                    blockbandwidths, subblockbandwidths, fromtensor, totensor, isbandedblockbanded,
                    Tensorizer, tensorizer, block, blockstart, blockstop, blocklengths,
                    domaintensorizer, rangetensorizer, blockrange, Block, BlockRange1

# Singularities
import ApproxFunBase: WeightSpace, weight

# Vec is for two points
import ApproxFunBase: Vec

# Jacobi import
import ApproxFunOrthogonalPolynomials: jacobip, JacobiSD, PolynomialSpace, order



include("c_transforms.jl")

include("Triangle/Triangle.jl")
include("Disk/Disk.jl")

include("clenshaw.jl")

# include("Sphere/SphericalHarmonics.jl")

include("show.jl")
include("plot.jl")

end # module
