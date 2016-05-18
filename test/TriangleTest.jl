using FixedSizeArrays,Plots,BandedMatrices,ApproxFun,MultivariateOrthogonalPolynomials, Base.Test
import MultivariateOrthogonalPolynomials: Recurrence

K=KoornwinderTriangle(0,0,0)

f=Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1))(0.1,0.2)
@test_approx_eq f(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)


f=Fun((x,y)->exp(x*cos(y)),K)
@test_approx_eq f(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test_approx_eq Fun(f,KoornwinderTriangle(1,0,0))(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test_approx_eq Fun(f,KoornwinderTriangle(1,1,0))(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test_approx_eq Fun(f,KoornwinderTriangle(1,1,1))(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test_approx_eq Fun(f,KoornwinderTriangle(0,0,1))(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)




f=Fun([1.],K)
@test_approx_eq Fun(f,KoornwinderTriangle(0,0,1))(0.1,0.2) f(0.1,0.2)
f=Fun([0.,1.],K)
@test_approx_eq Fun(f,KoornwinderTriangle(0,0,1))(0.1,0.2) f(0.1,0.2)
f=Fun([0.,0.,1.],K)
@test_approx_eq Fun(f,KoornwinderTriangle(0,0,1))(0.1,0.2) f(0.1,0.2)




f=Fun((x,y)->exp(x*cos(y)),K)
D=Derivative(space(f),[1,0])
@test_approx_eq_eps (D*f)(0.1,0.2) ((x,y)->cos(y)*exp(x*cos(y)))(0.1,0.2) 100000eps()

D=Derivative(space(f),[0,1])
@test_approx_eq_eps (D*f)(0.1,0.2) ((x,y)->-x*sin(y)*exp(x*cos(y)))(0.1,0.2) 100000eps()

D=Derivative(space(f),[1,1])
@test_approx_eq_eps (D*f)(0.1,0.2) ((x,y)->-sin(y)*exp(x*cos(y)) -x*sin(y)*cos(y)exp(x*cos(y)))(0.1,0.2)  1000000eps()

D=Derivative(space(f),[0,2])
@test_approx_eq_eps (D*f)(0.1,0.2) ((x,y)->-x*cos(y)*exp(x*cos(y)) + x^2*sin(y)^2*exp(x*cos(y)))(0.1,0.2)  1000000eps()

D=Derivative(space(f),[2,0])
@test_approx_eq_eps (D*f)(0.1,0.2) ((x,y)->cos(y)^2*exp(x*cos(y)))(0.1,0.2)  1000000eps()



## Recurrence



S=KoornwinderTriangle(1,1,1)

Mx=Recurrence(1,S)
My=Recurrence(2,S)
f=Fun((x,y)->exp(x*sin(y)),S)

@test_approx_eq (Mx*f)(0.1,0.2) 0.1*f(0.1,0.2)
@test_approx_eq (My*f)(0.1,0.2) 0.2*f(0.1,0.2)
@test_approx_eq ((Mx+My)*f)(0.1,0.2) 0.3*f(0.1,0.2)
