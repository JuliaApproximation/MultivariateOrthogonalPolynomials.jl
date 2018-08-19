__precompile__()

module MultivariateOrthogonalPolynomials
using Base, RecipesBase, ApproxFun, BandedMatrices, BlockArrays,
    FastTransforms, FastGaussQuadrature, StaticArrays, FillArrays,
    LinearAlgebra, Libdl, SpecialFunctions

# package code goes here
import Base: values,getindex,setindex!,*,.*,+,.+,-,.-,==,<,<=,>,
                >=,./,/,.^,^,\,∪,transpose


import BandedMatrices: mul!, inbands_getindex, inbands_setindex!

# ApproxFun general import
import ApproxFun: BandedMatrix, order, blocksize,
                  linesum,complexlength, BandedBlockBandedMatrix,
                  real, eps, isapproxinteger, ∞, FiniteRange, DFunction,
                  TransformPlan, ITransformPlan

# Domains import
import ApproxFun: fromcanonical, tocanonical, domainscompatible

# Operator import
import ApproxFun:    bandinds,SpaceOperator, ConversionWrapper, DerivativeWrapper,
                  rangespace, domainspace, InterlaceOperator,
                  promotedomainspace,  CalculusOperator, interlace, Multiplication,
                   choosedomainspace, SubOperator, ZeroOperator,
                    Dirichlet, DirichletWrapper, Neumann, Laplacian, ConstantTimesOperator, Conversion,
                    Derivative, ConcreteMultiplication, ConcreteConversion, ConcreteLaplacian,
                    ConcreteDerivative, TimesOperator, MultiplicationWrapper, TridiagonalOperator


# Spaces import
import ApproxFun: PolynomialSpace, ConstantSpace, NoSpace, prectype,
                    SumSpace,PiecewiseSpace, ArraySpace,
                    UnsetSpace, canonicalspace, canonicaldomain, domain, evaluate,
                    AnyDomain, plan_transform,plan_itransform,
                    transform,itransform,transform!,itransform!,
                    isambiguous, fromcanonical, tocanonical, checkpoints, ∂, spacescompatible,
                    mappoint, UnivariateSpace, setdomain, setcanonicaldomain, canonicaldomain,
                    Space, points, space, conversion_rule, maxspace_rule,
                    union_rule, coefficients, RealUnivariateSpace, PiecewiseSegment, rangetype, cfstype

# Multivariate import
import ApproxFun: BivariateDomain,DirectSumSpace, AbstractProductSpace, factor,
                    BivariateFun,  ProductFun, LowRankFun, lap, columnspace,
                    blockbandinds, subblockbandinds, fromtensor, totensor, isbandedblockbanded,
                    Tensorizer, tensorizer, block, blockstart, blockstop, blocklengths,
                    domaintensorizer, rangetensorizer, blockrange, Block, BlockRange1


# Jacobi import
import ApproxFun: jacobip, JacobiSD

# Singularities
import ApproxFun: WeightSpace, weight

# Vec is for two points
import ApproxFun: Vec



include("c_tri2cheb.jl")

include("Triangle.jl")
include("DirichletTriangle.jl")

include("SphericalHarmonics.jl")

include("show.jl")
include("plot.jl")

end # module
