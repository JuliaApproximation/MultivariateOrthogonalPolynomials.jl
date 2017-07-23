__precompile__()

module MultivariateOrthogonalPolynomials
    using Base, Compat, RecipesBase, ApproxFun, BandedMatrices, FastGaussQuadrature, StaticArrays

# package code goes here
import Base: values,getindex,setindex!,*,.*,+,.+,-,.-,==,<,<=,>,
                >=,./,/,.^,^,\,∪,transpose


import BandedMatrices: αA_mul_B_plus_βC!, inbands_getindex, inbands_setindex!

importall ApproxFun

# ApproxFun general import
import ApproxFun: BandedMatrix, order, blocksize,
                  linesum,complexlength, BandedBlockBandedMatrix, bbbzeros,
                  real, eps, isapproxinteger, ∞, FiniteRange, F

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
                   mappoint, UnivariateSpace, setdomain, Space, points, space, conversion_rule, maxspace_rule,
                   union_rule, coefficients, RealUnivariateSpace, PiecewiseSegment

# Multivariate import
import ApproxFun: BivariateDomain,DirectSumSpace, AbstractProductSpace, factor,
                    BivariateFun,  ProductFun, LowRankFun, lap, columnspace,
                    blockbandinds, subblockbandinds, fromtensor, totensor, isbandedblockbanded,
                    Tensorizer, tensorizer, block, blockstart, blockstop, blocklengths,
                    domaintensorizer, rangetensorizer, blockrange, Block


# Jacobi import
import ApproxFun: jacobip, JacobiSD

# Singularities
import ApproxFun: WeightSpace, weight

# Vec is for two points
import ApproxFun: Vec



include("Triangle.jl")
include("DirichletTriangle.jl")

include("show.jl")
include("plot.jl")

end # module
