using StaticArrays,Plots,BandedMatrices,
        ApproxFun,MultivariateOrthogonalPolynomials, Base.Test
    import MultivariateOrthogonalPolynomials: Lowering, ProductTriangle, clenshaw, block, TriangleWeight,plan_evaluate
    import ApproxFun: testbandedblockbandedoperator, Block, BandedBlockBandedMatrix, bbbzeros, blockcolrange, blocksize, Vec




pf=ProductFun((x,y)->exp(x*cos(y)),ProductTriangle(1,1,1),40,40)
@test pf(0.1,0.2) ≈ exp(0.1*cos(0.2))


f=Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1))
@test Fun(f,ProductTriangle(1,1,1))(0.1,0.2) ≈ exp(0.1*cos(0.2))

Jx=MultivariateOrthogonalPolynomials.Lowering{1}(space(f))
testbandedblockbandedoperator(Jx)
@test Fun(Jx*f,ProductTriangle(0,1,1))(0.1,0.2) ≈ 0.1exp(0.1*cos(0.2))

Jy=MultivariateOrthogonalPolynomials.Lowering{2}(space(f))
testbandedblockbandedoperator(Jy)

@test Jy[3,1] ≈ 1/3

@test ApproxFun.colstop(Jy,1) == 3
@test ApproxFun.colstop(Jy,2) == 5
@test ApproxFun.colstop(Jy,3) == 6
@test ApproxFun.colstop(Jy,4) == 8


@test Fun(Jy*f,ProductTriangle(1,0,1))(0.1,0.2) ≈ 0.2exp(0.1*cos(0.2))
@test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),KoornwinderTriangle(1,0,1))).coefficients) < 1E-11

Jz=MultivariateOrthogonalPolynomials.Lowering{3}(space(f))
testbandedblockbandedoperator(Jz)
@test Fun(Jz*f,ProductTriangle(1,1,0))(0.1,0.2) ≈ (1-0.1-0.2)exp(0.1*cos(0.2))

@test f(0.1,0.2) ≈ exp(0.1*cos(0.2))


# Test conversion
C=Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(1,0,0))
testbandedblockbandedoperator(C)

C=Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(0,1,0))
testbandedblockbandedoperator(C)

C=Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(0,0,1))
testbandedblockbandedoperator(C)


C=Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(1,1,1))
testbandedblockbandedoperator(C)


## 0,0,0 case

Jx=Lowering{1}(KoornwinderTriangle(0,0,0))
testbandedblockbandedoperator(Jx)
Jy=Lowering{2}(KoornwinderTriangle(0,0,0))
testbandedblockbandedoperator(Jy)
Jz=Lowering{3}(KoornwinderTriangle(0,0,0))
testbandedblockbandedoperator(Jz)



Cx=I:KoornwinderTriangle(-1,0,0)→KoornwinderTriangle(0,0,0)
testbandedblockbandedoperator(Cx)

Cy=I:KoornwinderTriangle(0,-1,0)→KoornwinderTriangle(0,0,0)
testbandedblockbandedoperator(Cy)

Cz=I:KoornwinderTriangle(0,0,-1)→KoornwinderTriangle(0,0,0)
testbandedblockbandedoperator(Cz)

f=Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(0,0,0))
@test f(0.1,0.2) ≈ exp(0.1*cos(0.2))

## Laplacian
Δ = Laplacian(space(f))
testbandedblockbandedoperator(Δ)





pyf=ProductFun((x,y)->y*exp(x*cos(y)),ProductTriangle(1,0,1))
@test pyf(0.1,0.2) ≈ 0.2exp(0.1*cos(0.2))





C=Conversion(KoornwinderTriangle(1,0,1),KoornwinderTriangle(1,1,1))
testbandedblockbandedoperator(C)
@test eltype(C)==Float64
f=Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1))
norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,0,1))-f).coefficients) < 1E-11
C=Conversion(KoornwinderTriangle(1,1,0),KoornwinderTriangle(1,1,1))
testbandedblockbandedoperator(C)
norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,0))-f).coefficients) < 1E-11
C=Conversion(KoornwinderTriangle(0,1,1),KoornwinderTriangle(1,1,1))
testbandedblockbandedoperator(C)
norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(0,1,1))-f).coefficients) < 1E-11


Jx=MultivariateOrthogonalPolynomials.Lowering{1}(space(f))→space(f)
testbandedblockbandedoperator(Jx)
@test norm((Jx*f-Fun((x,y)->x*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-10


Jy=MultivariateOrthogonalPolynomials.Lowering{2}(space(f))→space(f)
testbandedblockbandedoperator(Jy)
@test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-10

Jz=MultivariateOrthogonalPolynomials.Lowering{3}(space(f))→space(f)
testbandedblockbandedoperator(Jz)
@test norm((Jz*f-Fun((x,y)->(1-x-y)*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-10



K=KoornwinderTriangle(1,1,1)


# test values
f=Fun(K,[1.])
@test f(0.1,0.2) ≈ 1
f=Fun(K,[0.,1.])
@test f(0.1,0.2) ≈ -1.4
f=Fun(K,[0.,0.,1.])
@test f(0.1,0.2) ≈ -1
f=Fun(K,[0.,0.,0.,1.])
@test f(0.1,0.2) ≈ 1.18
f=Fun(K,[0.,0.,0.,0.,1.])
@test f(0.1,0.2) ≈ 1.2


K=KoornwinderTriangle(2,1,1)

C=Conversion(KoornwinderTriangle(1,1,1),K)
testbandedblockbandedoperator(C)

testbandedblockbandedoperator(Lowering{1}(K))

Jx=(Lowering{1}(K)→K)
testbandedblockbandedoperator(Jx)

f=Fun(K,[1.])
@test f(0.1,0.2) ≈ 1
f=Fun(K,[0.,1.])
@test f(0.1,0.2) ≈ -2.3
f=Fun(K,[0.,0.,1.])
@test f(0.1,0.2) ≈ -1
f=Fun(K,[0.,0.,0.,1.])
@test f(0.1,0.2) ≈ 3.16
f=Fun(K,[0.,0.,0.,0.,1.])
@test f(0.1,0.2) ≈ 2.1


K=KoornwinderTriangle(1,2,1)

Jx=(Lowering{1}(K)→K)
testbandedblockbandedoperator(Jx)


f=Fun(K,[1.])
@test f(0.1,0.2) ≈ 1
f=Fun(K,[0.,1.])
@test f(0.1,0.2) ≈ -1.3
f=Fun(K,[0.,0.,1.])
@test f(0.1,0.2) ≈ -1.7
f=Fun(K,[0.,0.,0.,1.])
@test f(0.1,0.2) ≈ 0.96
f=Fun(K,[0.,0.,0.,0.,1.])
@test f(0.1,0.2) ≈ 1.87

# Test conversions
K=KoornwinderTriangle(1,1,1)

f=Fun(K,[1.])
@test Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
@test Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
@test Fun(f,KoornwinderTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)
f=Fun(K,[0.,1.])
@test Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
@test Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
@test Fun(f,KoornwinderTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)
f=Fun(K,[0.,0.,1.])
@test Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
@test Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
@test Fun(f,KoornwinderTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)
f=Fun(K,[0.,0.,0.,1.])
@test Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
@test Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
@test Fun(f,KoornwinderTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)
f=Fun(K,[0.,0.,0.,0.,1.])
@test Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
@test Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
@test Fun(f,KoornwinderTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)

f=Fun((x,y)->exp(x*cos(y)),K)
@test f(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test Fun(f,KoornwinderTriangle(2,2,1))(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test Fun(f,KoornwinderTriangle(2,2,2))(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)


K=KoornwinderTriangle(0,0,0)
f=Fun((x,y)->exp(x*cos(y)),K)
D=Derivative(space(f),[1,0])
@test (D*f)(0.1,0.2) ≈ ((x,y)->cos(y)*exp(x*cos(y)))(0.1,0.2) atol=100000eps()

D=Derivative(space(f),[0,1])
@test (D*f)(0.1,0.2) ≈  ((x,y)->-x*sin(y)*exp(x*cos(y)))(0.1,0.2) atol=100000eps()

D=Derivative(space(f),[1,1])
@test (D*f)(0.1,0.2) ≈ ((x,y)->-sin(y)*exp(x*cos(y)) -x*sin(y)*cos(y)exp(x*cos(y)))(0.1,0.2)  atol=1000000eps()

D=Derivative(space(f),[0,2])
@test (D*f)(0.1,0.2) ≈ ((x,y)->-x*cos(y)*exp(x*cos(y)) + x^2*sin(y)^2*exp(x*cos(y)))(0.1,0.2)  atol=1000000eps()

D=Derivative(space(f),[2,0])
@test (D*f)(0.1,0.2) ≈ ((x,y)->cos(y)^2*exp(x*cos(y)))(0.1,0.2)  atol=1000000eps()






## Recurrence



S=KoornwinderTriangle(1,1,1)

Mx=Lowering{1}(S)
My=Lowering{2}(S)
f=Fun((x,y)->exp(x*sin(y)),S)

@test (Mx*f)(0.1,0.2) ≈ 0.1*f(0.1,0.2)
@test (My*f)(0.1,0.2) ≈ 0.2*f(0.1,0.2) atol=1E-12
@test ((Mx+My)*f)(0.1,0.2) ≈ 0.3*f(0.1,0.2) atol=1E-12


Jx=Mx → S
Jy=My → S

@test (Jy*f)(0.1,0.2) ≈ 0.2*f(0.1,0.2) atol=1E-12

x,y=0.1,0.2

P0=[Fun(S,[1.])(x,y)]
P1=Float64[Fun(S,[zeros(k);1.])(x,y) for k=1:2]
P2=Float64[Fun(S,[zeros(k);1.])(x,y) for k=3:5]


K=Block(1)
@test Matrix(Jx[K,K])*P0+Matrix(Jx[K+1,K])'*P1 ≈ x*P0
@test Matrix(Jy[K,K])*P0+Matrix(Jy[K+1,K])'*P1 ≈ y*P0

K=Block(2)
@test Matrix(Jx[K-1,K])'*P0+Matrix(Jx[K,K])'*P1+Matrix(Jx[K+1,K])'*P2 ≈ x*P1
@test Matrix(Jy[K-1,K])'*P0+Matrix(Jy[K,K])'*P1+Matrix(Jy[K+1,K])'*P2 ≈ y*P1


A,B,C=Matrix(Jy[K+1,K])',Matrix(Jy[K,K])',Matrix(Jy[K-1,K])'


@test C*P0+B*P1+A*P2 ≈ y*P1


## Weighted


S=TriangleWeight(1.,1.,1.,KoornwinderTriangle(1,1,1))
C=Conversion(S,KoornwinderTriangle(0,0,0))

testbandedblockbandedoperator(C)
f=Fun(S,rand(10))
@test f(0.1,0.2) ≈ (C*f)(0.1,0.2)



@test (Derivative(S,[1,0])*Fun(S,[1.]))(0.1,0.2) ≈ ((x,y)->y*(1-x-y)-x*y)(0.1,0.2)

@test (Laplacian(S)*Fun(S,[1.]))(0.1,0.2) ≈ -2*0.1-2*0.2




## Diagonal D

S = TriangleWeight(1,1,1,KoornwinderTriangle(1,1,1))


Dx = Derivative(S,[1,0])
Dy = Derivative(S,[0,1])

for k=1:10
    v=[zeros(k);1.0]
    @test (Dy*Fun(S,v))(0.1,0.2) ≈ (Derivative([0,1])*Fun(Fun(S,v),KoornwinderTriangle(1,1,1)))(0.1,0.2)
    @test (Dx*Fun(S,v))(0.1,0.2) ≈ (Derivative([1,0])*Fun(Fun(S,v),KoornwinderTriangle(1,1,1)))(0.1,0.2)
end


d = Triangle()
B = Dirichlet(d)
Δ = Laplacian(domainspace(B))

@which Dirichlet(d)

ApproxFun.promotedomainspace(Δ, domainspace(Δ))
ApproxFun.promotedomainspace(B, domainspace(Δ))

L = [B;Δ]

∂u = Fun((x,y) -> exp(x)*sin(y), ∂(d))
Fun([∂u;0],rangespace(L))

L = [B;Laplacian()]
Fun([∂u;0],rangespace(L))


v=[1.0]


Δ=Laplacian(S)

f=Fun(S,rand(3))
h=0.01
QR=qrfact(I-h*Δ)
@time u=\(QR,f;tolerance=1E-7)
@time g=Fun(f,rangespace(QR))
@time \(QR,g;tolerance=1E-7)
