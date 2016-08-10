module MultivariateOrthogonalPolynomials
    using Base, Compat, Plots, ApproxFun, BandedMatrices, FastGaussQuadrature

# package code goes here
import Base: values,getindex,setindex!,*,.*,+,.+,-,.-,==,<,<=,>,
                >=,./,/,.^,^,\,∪,transpose

importall ApproxFun

# ApproxFun general import
import ApproxFun: BandedMatrix,order,
                  linesum,complexlength,
                  real, eps, isapproxinteger

# Operator import
import ApproxFun:    bandinds,SpaceOperator, ConversionWrapper, DerivativeWrapper,
                  rangespace, domainspace, addentries!, BandedOperator,
                  promotedomainspace,  CalculusOperator, interlace, Multiplication,
                  DiagonalArrayOperator, Recurrence, FiniteFunctional, choosedomainspace,
                    Dirichlet, Neumann, Laplacian, ConstantTimesOperator, Conversion, isfunctional,
                    dirichlet, neumann, Derivative, ConcreteMultiplication, ConcreteConversion, ConcreteLaplacian,
                    ConcreteDerivative, TimesOperator, MultiplicationWrapper, TridiagonalOperator


# Spaces import
import ApproxFun: PolynomialSpace,ConstantSpace,
                    SumSpace,PiecewiseSpace, ArraySpace,RealBasis,ComplexBasis,AnyBasis,
                    UnsetSpace, canonicalspace, domain, evaluate,
                    AnyDomain, plan_transform,plan_itransform,
                    transform,itransform,transform!,itransform!,
                    isambiguous, fromcanonical, tocanonical, checkpoints, ∂, spacescompatible,
                   mappoint, UnivariateSpace, setdomain, Space, points, space, conversion_rule, maxspace_rule,
                   coefficients, RealUnivariateSpace

# Multivariate import
import ApproxFun: BivariateDomain,DirectSumSpace,TupleSpace, AbstractProductSpace,
                    BivariateFun,  ProductFun, LowRankFun, lap, columnspace, diagop, isproductop, discretize,
                    schurfact, kronfact, isdiagop, blockbandinds, fromtensor, totensor, totree


# Jacobi import
import ApproxFun: jacobip, JacobiSD





include("JacobiSquare.jl")
include("DiskSpace.jl")
include("Triangle.jl")

include("plot.jl")

end # module
