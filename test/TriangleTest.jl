using FixedSizeArrays,Plots,BandedMatrices,ApproxFun,MultivariateOrthogonalPolynomials, Base.Test


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
(D*f)(0.1,0.2),((x,y)->exp(x*cos(y)))(0.1,0.2)

Fun(f,KoornwinderTriangle(0,0,1))


C[1,2]


n=1;k=1
k/(2k+1)*(n+k+1)/(2n+2)

(n+k+2)/(2n+2)*(k+1)/(2k+1)
k/(2k+1)*(n-k+1)/(2n+2)


f=Fun([0.,0.,0.,0.,0.,1.],K)
    f(0.1,0.2)




C=Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(0,0,1))

C=Conversion(K,KoornwinderTriangle(1,1,0))
    (C*f)(0.1,0.2)

f=Fun((x,y)->exp(x*cos(y)),MultivariateOrthogonalPolynomials.ProductTriangle(K))


Fun(f,K)


Fun((x,y)->exp(x*cos(y)),K)


f=f.coefficients

C=ApproxFun.totensor(f)
D=Float64[2.0^(-k) for k=0:size(C,1)-1]


C.'

(C.')*diagm(D)
@which coefficients(f.coefficients,K,MultivariateOrthogonalPolynomials.ProductTriangle(K))


f=Fun([11,21,12,31,22,13],K)
C=ApproxFun.totensor(f.coefficients)
D=Float64[2.0^(-k) for k=0:size(C,1)-1]

ApproxFun.fromtensor((C').*D)

f=Fun([0.,0.,0.,0.,0.,1.],K)
    C=ApproxFun.totensor(f.coefficients)
    D=Float64[2.0^(-k) for k=0:size(C,1)-1]
    g=ProductFun((C')*diagm(D),K)
    g(x,y)

D
C'*diagm(D)
D
f(x,y)

coefficients(g)

C=C

onversion(K,KoornwinderTriangle(1,0,0))



full(C[1:10,1:10])

n,k=1,0

(n+k+1)/(2n+2)


(n+k+2)/(2n+2)

f=Fun([0.,1.],K)
    C*f

F=ProductFun(ApproxFun.totensor(f.coefficients),space(f))
K=space(f)

@which ApproxFun.tocanonical(domain(K),0.1,0.2)

ApproxFun.column

x,y=0.1,0.2

-1+2y/(1-x)

2x-1
ApproxFun.columnspace(space(f),2)
Fun([1.],ApproxFun.columnspace(space(f),2))(x)

space(K,2)

F(0.1,0.2)

@which evaluate(f.coefficients,space(f),Vec(0.1,0.2))

@which evaluate(f,0.1,0.2)
(C*f)(0.1,0.2)
