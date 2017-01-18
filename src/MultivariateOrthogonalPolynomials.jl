__precompile__()

module MultivariateOrthogonalPolynomials
    using Base, Compat, RecipesBase, ApproxFun, BandedMatrices, FastGaussQuadrature, FixedSizeArrays

# package code goes here
import Base: values,getindex,setindex!,*,.*,+,.+,-,.-,==,<,<=,>,
                >=,./,/,.^,^,\,∪,transpose


import BandedMatrices: αA_mul_B_plus_βC!, inbands_getindex, inbands_setindex!

importall ApproxFun

# ApproxFun general import
import ApproxFun: BandedMatrix, order, blocksize,
                  linesum,complexlength, BandedBlockBandedMatrix, bbbzeros,
                  real, eps, isapproxinteger, ∞, FiniteRange

# Operator import
import ApproxFun:    bandinds,SpaceOperator, ConversionWrapper, DerivativeWrapper,
                  rangespace, domainspace,
                  promotedomainspace,  CalculusOperator, interlace, Multiplication,
                   choosedomainspace, SubOperator,
                    Dirichlet, Neumann, Laplacian, ConstantTimesOperator, Conversion,
                    dirichlet, neumann, Derivative, ConcreteMultiplication, ConcreteConversion, ConcreteLaplacian,
                    ConcreteDerivative, TimesOperator, MultiplicationWrapper, TridiagonalOperator


# Spaces import
import ApproxFun: PolynomialSpace, ConstantSpace, NoSpace,
                    SumSpace,PiecewiseSpace, ArraySpace,RealBasis,ComplexBasis,AnyBasis,
                    UnsetSpace, canonicalspace, canonicaldomain, domain, evaluate,
                    AnyDomain, plan_transform,plan_itransform,
                    transform,itransform,transform!,itransform!,
                    isambiguous, fromcanonical, tocanonical, checkpoints, ∂, spacescompatible,
                   mappoint, UnivariateSpace, setdomain, Space, points, space, conversion_rule, maxspace_rule,
                   coefficients, RealUnivariateSpace

# Multivariate import
import ApproxFun: BivariateDomain,DirectSumSpace,TupleSpace, AbstractProductSpace,
                    BivariateFun,  ProductFun, LowRankFun, lap, columnspace,
                    blockbandinds, subblockbandinds, fromtensor, totensor, isbandedblockbanded,
                    Tensorizer, tensorizer, block, blockstart, blockstop, blocklengths,
                    domaintensorizer, rangetensorizer, blockrange, Block


# Jacobi import
import ApproxFun: jacobip, JacobiSD

# Singularities
import ApproxFun: WeightSpace, weight



include("Triangle.jl")

include("plot.jl")

end # module
