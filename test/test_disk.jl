using Revise, ApproxFun, MultivariateOrthogonalPolynomials
import MultivariateOrthogonalPolynomials: checkerboard, icheckerboard

f = (x,y) -> x*y+cos(y-0.1)+sin(x)+1; 
ff = Fun(f, ChebyshevDisk(), 1000)
@test ff(0.1,0.2) ≈ f(0.1,0.2)

ff = Fun(f, ChebyshevDisk())
@test ff(0.1,0.2) ≈ f(0.1,0.2)

