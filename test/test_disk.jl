using Revise, ApproxFun, MultivariateOrthogonalPolynomials
import MultivariateOrthogonalPolynomials: checkerboard, icheckerboard

f = (x,y) -> x*y+cos(y-0.1)+sin(x)+1; ff = Fun((r,θ) -> f(r*cos(θ),r*sin(θ)), (-1..1) × PeriodicSegment());

cfs = ApproxFunBase.coefficientmatrix(ff)
icheckerboard(checkerboard(cfs))[1:size(cfs,1),:]

cfs