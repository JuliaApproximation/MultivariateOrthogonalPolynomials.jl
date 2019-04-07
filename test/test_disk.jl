using ApproxFun
f = (x,y) -> x*y+cos(y-0.1)+sin(x)+1; ff = Fun((r,θ) -> f(r*cos(θ),r*sin(θ)), (-1..1) × PeriodicSegment());
ApproxFunBase.coefficientmatrix(ff)