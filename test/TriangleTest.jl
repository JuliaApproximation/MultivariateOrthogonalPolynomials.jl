using FixedSizeArrays,Plots,BandedMatrices,
        ApproxFun,MultivariateOrthogonalPolynomials, Base.Test
    import MultivariateOrthogonalPolynomials: Recurrence, ProductTriangle, clenshaw, block, TriangleWeight,plan_evaluate
    import ApproxFun: testbandedblockbandedoperator, Block, BandedBlockBandedMatrix, bbbzeros, blockcolrange, blocksize





pf=ProductFun((x,y)->exp(x*cos(y)),ProductTriangle(1,1,1),40,40)
@test_approx_eq pf(0.1,0.2) exp(0.1*cos(0.2))


f=Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1))
@test_approx_eq f(0.1,0.2) exp(0.1*cos(0.2))


# Test conversion
C=Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(1,0,0))
testbandedblockbandedoperator(C)

C=Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(0,1,0))
testbandedblockbandedoperator(C)

C=Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(0,0,1))
testbandedblockbandedoperator(C)


C=Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(1,1,1))
testbandedblockbandedoperator(C)

Δ = Laplacian(space(f))
testbandedblockbandedoperator(Δ)

# Test recurrence operators
Jx=MultivariateOrthogonalPolynomials.Recurrence(1,space(f))
testbandedblockbandedoperator(Jx)

@test ApproxFun.colstop(Jx,1) == 2
@test ApproxFun.colstop(Jx,2) == 4
@test ApproxFun.colstop(Jx,3) == 5
@test ApproxFun.colstop(Jx,4) == 7


@test norm((Jx*f-Fun((x,y)->x*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-11


Jy=MultivariateOrthogonalPolynomials.Recurrence(2,space(f))



testbandedblockbandedoperator(Jy)

@test_approx_eq Jy[3,1] 1/3

@test ApproxFun.colstop(Jy,1) == 3
@test ApproxFun.colstop(Jy,2) == 5
@test ApproxFun.colstop(Jy,3) == 6
@test ApproxFun.colstop(Jy,4) == 8

@test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),KoornwinderTriangle(1,0,1))).coefficients) < 1E-11

pyf=ProductFun((x,y)->y*exp(x*cos(y)),ProductTriangle(1,0,1))
@test_approx_eq pyf(0.1,0.2) 0.2exp(0.1*cos(0.2))





C=ApproxFun.ConcreteConversion(KoornwinderTriangle(1,0,1),KoornwinderTriangle(1,1,1))
testbandedblockbandedoperator(C)
@test eltype(C)==Float64
norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,0,1))-f).coefficients) < 1E-11
C=ApproxFun.ConcreteConversion(KoornwinderTriangle(1,1,0),KoornwinderTriangle(1,1,1))
testbandedblockbandedoperator(C)
norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,0))-f).coefficients) < 1E-11
C=ApproxFun.ConcreteConversion(KoornwinderTriangle(0,1,1),KoornwinderTriangle(1,1,1))
testbandedblockbandedoperator(C)
norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(0,1,1))-f).coefficients) < 1E-11


Jy=MultivariateOrthogonalPolynomials.Recurrence(2,space(f))→space(f)
testbandedblockbandedoperator(Jy)


@test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-10




K=KoornwinderTriangle(1,1,1)

f=Fun(K,[1.])
@test_approx_eq Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) f(0.1,0.2)
f=Fun(K,[0.,1.])
@test_approx_eq Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) f(0.1,0.2)
f=Fun(K,[0.,0.,1.])
@test_approx_eq Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) f(0.1,0.2)
f=Fun(K,[0.,0.,0.,1.])
@test_approx_eq Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) f(0.1,0.2)
f=Fun(K,[0.,0.,0.,0.,1.])
@test_approx_eq Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) f(0.1,0.2)


f=Fun((x,y)->exp(x*cos(y)),K)
@test_approx_eq f(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test_approx_eq Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test_approx_eq Fun(f,KoornwinderTriangle(2,2,1))(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test_approx_eq Fun(f,KoornwinderTriangle(2,2,2))(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test_approx_eq Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)





K=KoornwinderTriangle(0,0,0)
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
@test_approx_eq_eps (My*f)(0.1,0.2) 0.2*f(0.1,0.2) 1E-12
@test_approx_eq_eps ((Mx+My)*f)(0.1,0.2) 0.3*f(0.1,0.2) 1E-12


Jx=Mx
Jy=My → S

@test_approx_eq_eps (Jy*f)(0.1,0.2) 0.2*f(0.1,0.2) 1E-12

x,y=0.1,0.2

P0=[Fun(S,[1.])(x,y)]
P1=Float64[Fun(S,[zeros(k);1.])(x,y) for k=1:2]
P2=Float64[Fun(S,[zeros(k);1.])(x,y) for k=3:5]


K=Block(1)
@test_approx_eq Matrix(Jx[K,K])*P0+Matrix(Jx[K+1,K])'*P1 x*P0
@test_approx_eq Matrix(Jy[K,K])*P0+Matrix(Jy[K+1,K])'*P1 y*P0

K=Block(2)
@test_approx_eq Matrix(Jx[K-1,K])'*P0+Matrix(Jx[K,K])'*P1+Matrix(Jx[K+1,K])'*P2 x*P1
@test_approx_eq Matrix(Jy[K-1,K])'*P0+Matrix(Jy[K,K])'*P1+Matrix(Jy[K+1,K])'*P2 y*P1


A,B,C=Matrix(Jy[K+1,K])',Matrix(Jy[K,K])',Matrix(Jy[K-1,K])'


@test_approx_eq C*P0+B*P1+A*P2 y*P1


## Weighted


S=TriangleWeight(1.,1.,1.,KoornwinderTriangle(1,1,1))

@test_approx_eq (Derivative(S,[1,0])*Fun(S,[1.]))(0.1,0.2) ((x,y)->y*(1-x-y)-x*y)(0.1,0.2)

@test_approx_eq (Laplacian(S)*Fun(S,[1.]))(0.1,0.2) -2*0.1-2*0.2



Δ=Laplacian(S)

f=Fun(S,rand(3))
h=0.01
QR=qrfact(I-h*Δ)
@time u=\(QR,f;tolerance=1E-7)

@time g=Fun(f,rangespace(QR))
@time \(QR,g;tolerance=1E-7)
x=y=linspace(0.,1.,10)

X=[ApproxFun.fromcanonical(domain(f),xy)[1] for xy in tuple.(x,y')]
Y=[ApproxFun.fromcanonical(domain(f),xy)[2] for xy in tuple.(x,y')]


using Plots
plotly()


surface(X,Y,u.(X,Y))

a=rangespace(Evaluation(Ultraspherical(9),-1))
@which ApproxFun.conversion_type(a,Chebyshev())

@which maxspace(a,Chebyshev())

S=Ultraspherical(1)
B=Evaluation(S,-1)
C=Conversion(rangespace(B),S)
(I-C*B)*Integral(S)
