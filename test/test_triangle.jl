using StaticArrays, BandedMatrices, BlockArrays, FastTransforms,
        ApproxFunBase, ApproxFunOrthogonalPolynomials, ApproxFun, 
        MultivariateOrthogonalPolynomials, LinearAlgebra, Test
import MultivariateOrthogonalPolynomials: Lowering, DuffyTriangle,
                        clenshaw, block, TriangleWeight,plan_evaluate, weight
import ApproxFunBase: testbandedblockbandedoperator, Block, BandedBlockBandedMatrix, blockcolrange, blocksize,
                Vec, plan_transform
import ApproxFunOrthogonalPolynomials: jacobip                
import BandedMatrices: colstop

@testset "Triangle domain" begin
    d = Triangle()
    @test fromcanonical(d, 1,0) == fromcanonical(d, Vec(1,0)) == tocanonical(d, Vec(1,0)) == Vec(1,0)
    @test fromcanonical(d, 0,1) == fromcanonical(d, Vec(0,1)) == tocanonical(d, Vec(0,1)) == Vec(0,1)
    @test fromcanonical(d, 0,0) == fromcanonical(d, Vec(0,0)) == tocanonical(d, Vec(0,0)) == Vec(0,0)

    d = Triangle(Vec(2,3),Vec(3,4),Vec(1,6))
    @test fromcanonical(d,0,0) == d.a == Vec(2,3)
    @test fromcanonical(d,1,0) == d.b == Vec(3,4)
    @test fromcanonical(d,0,1) == d.c == Vec(1,6)
    @test tocanonical(d,d.a) == Vec(0,0)
    @test tocanonical(d,d.b) == Vec(1,0)
    @test tocanonical(d,d.c) == Vec(0,1)
end

let P = (n,k,a,b,c,x,y) -> x == 1.0 ? ((1-x))^k*jacobip(n-k,2k+b+c+1,a,1.0)*jacobip(k,c,b,-1.0) :
        ((1-x))^k*jacobip(n-k,2k+b+c+1,a,2x-1)*jacobip(k,c,b,2y/(1-x)-1)
    function cfseval(a,b,c,r,x,y)
        j = 1; ret = 0.0
        for n=0:length(r),k=0:n
            ret += r[j]*P(n,k,a,b,c,x,y)
            j += 1
            j > length(r) && break
        end
        ret
    end
    @testset "Degenerate conversion" begin
        C = Conversion(JacobiTriangle(0.0,-0.5,-0.5),JacobiTriangle(0.0,0.5,-0.5))
        testbandedblockbandedoperator(C)
        r = randn(10)
        x,y = 0.1, 0.2
        @test cfseval(0.0,-0.5,-0.5,r,x,y) ≈ cfseval(0.0,0.5,-0.5,[C[k,j] for k=1:10,j=1:10]*r,x,y)
        @test cfseval(0.0,-0.5,-0.5,r,x,y) ≈ cfseval(0.0,0.5,-0.5,C[1:10,1:10]*r,x,y)

        C = Conversion(JacobiTriangle(0.0,-0.5,-0.5),JacobiTriangle(0.0,-0.5,0.5))
        testbandedblockbandedoperator(C)
        r = randn(10)
        @test cfseval(0.0,-0.5,-0.5,r,x,y) ≈ cfseval(0.0,-0.5,0.5,[C[k,j] for k=1:10,j=1:10]*r,x,y)
        @test cfseval(0.0,-0.5,-0.5,r,x,y) ≈ cfseval(0.0,-0.5,0.5,C[1:10,1:10]*r,x,y)
    end
end

@testset "JacobiTriangle constructors" begin
    @time f = Fun((x,y)->cos(100x*y),JacobiTriangle(0.0,-0.5,-0.5)); # 0.08s
    P = plan_evaluate(f)
    @test f(0.1,0.2) ≈ P(0.1,0.2) ≈ cos(100*0.1*0.2)
    @test values(f) ≈ P.(points(f))

    @time f = Fun((x,y)->cos(500x*y),JacobiTriangle(0.0,-0.5,-0.5),40_000); # 0.05
    @test f(0.1,0.2) ≈ cos(500*0.1*0.2)

    @time f = Fun((x,y)->cos(100x*y),JacobiTriangle(0.0,0.5,-0.5)); # 0.08s
    @time f = Fun((x,y)->cos(500x*y),JacobiTriangle(0.0,0.5,-0.5),40_000); # 0.06
    @test f(0.1,0.2) ≈ cos(500*0.1*0.2)

    @time f = Fun((x,y)->cos(100x*y),JacobiTriangle(0.0,0.5,0.5)); # 0.07s
    @time f = Fun((x,y)->cos(500x*y),JacobiTriangle(0.0,0.5,0.5),40_000); # 0.06
    @test f(0.1,0.2) ≈ cos(500*0.1*0.2)

    f = Fun((x,y)->cos(100x*y),JacobiTriangle(0.0,-0.5,0.5)); # 1.15s
    f = Fun((x,y)->cos(500x*y),JacobiTriangle(0.0,-0.5,0.5),40_000); # 0.2
    @test f(0.1,0.2) ≈ cos(500*0.1*0.2)

    d = Triangle(Vec(0,0),Vec(3,4),Vec(1,6))
    @time f = Fun((x,y)->cos(x*y),JacobiTriangle(0.0,-0.5,-0.5,d)); # 0.08s
    @test f(2,4) ≈ cos(2*4)
    @time f = Fun((x,y)->cos(x*y),JacobiTriangle(0.0,-0.5,0.5,d)); # 0.08s
    @test f(2,4) ≈ cos(2*4)
end


@testset "old constructors" begin
    f = Fun((x,y)->exp(x*cos(y)),JacobiTriangle(1,1,1))
    @test f(0.1,0.2) ≈ exp(0.1*cos(0.2))
    f=Fun((x,y)->exp(x*cos(y)),JacobiTriangle(0,0,0))
    @test f(0.1,0.2) ≈ exp(0.1*cos(0.2))
    d = Triangle(Vec(0,0),Vec(3,4),Vec(1,6))
    f = Fun((x,y)->exp(x*cos(y)),JacobiTriangle(1,1,1,d))
    @test f(2,4) ≈ exp(2*cos(4))


    K=JacobiTriangle(1,1,1)
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

    K=JacobiTriangle(2,1,1)
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

    K=JacobiTriangle(1,2,1)
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
end


@testset "Triangle Jacobi" begin
    f = Fun((x,y)->exp(x*cos(y)),JacobiTriangle(1,1,1))
    Jx = MultivariateOrthogonalPolynomials.Lowering{1}(space(f))
    testbandedblockbandedoperator(Jx)

    Jy = MultivariateOrthogonalPolynomials.Lowering{2}(space(f))
    testbandedblockbandedoperator(Jy)

    @test Jy[3,1] ≈ 1/3

    @test colstop(Jy,1) == 3
    @test colstop(Jy,2) == 5
    @test colstop(Jy,3) == 6
    @test colstop(Jy,4) == 8


    @test (Jy*f)(0.1,0.2) ≈ 0.2exp(0.1*cos(0.2))
    @test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),JacobiTriangle(1,0,1))).coefficients) < 1E-11

    Jz = MultivariateOrthogonalPolynomials.Lowering{3}(space(f))
    testbandedblockbandedoperator(Jz)
    @test (Jz*f)(0.1,0.2) ≈ (1-0.1-0.2)exp(0.1*cos(0.2))

    @test f(0.1,0.2) ≈ exp(0.1*cos(0.2))

    Jx=Lowering{1}(JacobiTriangle(0,0,0))
    testbandedblockbandedoperator(Jx)
    Jy=Lowering{2}(JacobiTriangle(0,0,0))
    testbandedblockbandedoperator(Jy)
    Jz=Lowering{3}(JacobiTriangle(0,0,0))
    testbandedblockbandedoperator(Jz)

    Jx=MultivariateOrthogonalPolynomials.Lowering{1}(space(f))→space(f)
    testbandedblockbandedoperator(Jx)
    @test norm((Jx*f-Fun((x,y)->x*exp(x*cos(y)),JacobiTriangle(1,1,1))).coefficients) < 1E-10


    Jy=MultivariateOrthogonalPolynomials.Lowering{2}(space(f))→space(f)
    testbandedblockbandedoperator(Jy)
    @test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),JacobiTriangle(1,1,1))).coefficients) < 1E-10

    Jz=MultivariateOrthogonalPolynomials.Lowering{3}(space(f))→space(f)
    testbandedblockbandedoperator(Jz)
    @test norm((Jz*f-Fun((x,y)->(1-x-y)*exp(x*cos(y)),JacobiTriangle(1,1,1))).coefficients) < 1E-10

    for K in (JacobiTriangle(2,1,1),JacobiTriangle(1,2,1))
        testbandedblockbandedoperator(Lowering{1}(K))
        Jx=(Lowering{1}(K)→K)
        testbandedblockbandedoperator(Jx)
    end

    d = Triangle(Vec(0,0),Vec(3,4),Vec(1,6))
    f = Fun((x,y)->exp(x*cos(y)),JacobiTriangle(1,1,1,d))
    Jx,Jy = MultivariateOrthogonalPolynomials.jacobioperators(space(f))
    x,y = fromcanonical(d,0.1,0.2)
    @test (Jx*f)(x,y) ≈ x*f(x,y)
    @test (Jy*f)(x,y) ≈ y*f(x,y)
end

@testset "Triangle Conversion" begin
    C = Conversion(JacobiTriangle(0,0,0),JacobiTriangle(1,0,0))
    testbandedblockbandedoperator(C)

    C = Conversion(JacobiTriangle(0,0,0),JacobiTriangle(0,1,0))
    testbandedblockbandedoperator(C)

    C = Conversion(JacobiTriangle(0,0,0),JacobiTriangle(0,0,1))
    testbandedblockbandedoperator(C)
    C=Conversion(JacobiTriangle(0,0,0),JacobiTriangle(1,1,1))
    testbandedblockbandedoperator(C)

    Cx = I:JacobiTriangle(-1,0,0)→JacobiTriangle(0,0,0)
    testbandedblockbandedoperator(Cx)

    Cy = I:JacobiTriangle(0,-1,0)→JacobiTriangle(0,0,0)
    testbandedblockbandedoperator(Cy)

    Cz = I:JacobiTriangle(0,0,-1)→JacobiTriangle(0,0,0)
    testbandedblockbandedoperator(Cz)

    C=Conversion(JacobiTriangle(1,0,1),JacobiTriangle(1,1,1))
    testbandedblockbandedoperator(C)
    @test eltype(C)==Float64

    f=Fun((x,y)->exp(x*cos(y)),JacobiTriangle(1,1,1))
    norm((C*Fun((x,y)->exp(x*cos(y)),JacobiTriangle(1,0,1))-f).coefficients) < 1E-11
    C=Conversion(JacobiTriangle(1,1,0),JacobiTriangle(1,1,1))
    testbandedblockbandedoperator(C)
    norm((C*Fun((x,y)->exp(x*cos(y)),JacobiTriangle(1,1,0))-f).coefficients) < 1E-11
    C=Conversion(JacobiTriangle(0,1,1),JacobiTriangle(1,1,1))
    testbandedblockbandedoperator(C)
    norm((C*Fun((x,y)->exp(x*cos(y)),JacobiTriangle(0,1,1))-f).coefficients) < 1E-11


    C=Conversion(JacobiTriangle(1,1,1),JacobiTriangle(2,1,1))
    testbandedblockbandedoperator(C)

    # Test conversions
    K=JacobiTriangle(1,1,1)
    f=Fun(K,[1.])
    @test Fun(f,JacobiTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,JacobiTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,JacobiTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)
    f=Fun(K,[0.,1.])
    @test Fun(f,JacobiTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,JacobiTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,JacobiTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)
    f=Fun(K,[0.,0.,1.])
    @test Fun(f,JacobiTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,JacobiTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,JacobiTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)
    f=Fun(K,[0.,0.,0.,1.])
    @test Fun(f,JacobiTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,JacobiTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,JacobiTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)
    f=Fun(K,[0.,0.,0.,0.,1.])
    @test Fun(f,JacobiTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,JacobiTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,JacobiTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)

    f=Fun((x,y)->exp(x*cos(y)),K)
    @test f(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
    @test Fun(f,JacobiTriangle(2,1,1))(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
    @test Fun(f,JacobiTriangle(2,2,1))(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
    @test Fun(f,JacobiTriangle(2,2,2))(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
    @test Fun(f,JacobiTriangle(1,1,2))(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
end

@testset "Triangle Derivative/Laplacian" begin
    Dx = Derivative(JacobiTriangle(1,0,1), [1,0])
    testbandedblockbandedoperator(Dx)

    Dy = Derivative(JacobiTriangle(1,0,1), [0,1])
    testbandedblockbandedoperator(Dy)

    Δ = Laplacian(JacobiTriangle(1,1,1))
    testbandedblockbandedoperator(Δ)
end

@testset "Triangle derivatives" begin
    K=JacobiTriangle(0,0,0)
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

    d = Triangle(Vec(0,0),Vec(3,4),Vec(1,6))
    K = JacobiTriangle(1,0,1,d)
    f=Fun((x,y)->exp(x*cos(y)),K)
    Dx = Derivative(space(f), [1,0])
    x,y = fromcanonical(d,0.1,0.2)
    @test (Dx*f)(x,y) ≈ cos(y)*exp(x*cos(y)) atol=1E-8
    Dy = Derivative(space(f), [0,1])
    @test (Dy*f)(x,y) ≈ -x*sin(y)*exp(x*cos(y)) atol=1E-8

    Δ = Laplacian(space(f))
    testbandedblockbandedoperator(Δ)
    @test (Δ*f)(x,y) ≈ exp(x*cos(y))*(-x*cos(y) + cos(y)^2 + x^2*sin(y)^2) atol=1E-6
end


@testset "Triangle recurrence" begin
    S=JacobiTriangle(1,1,1)

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
end

@testset "TriangleWeight" begin
    S=TriangleWeight(1.,1.,1.,JacobiTriangle(1,1,1))
    C=Conversion(S,JacobiTriangle(0,0,0))
    testbandedblockbandedoperator(C)
    f=Fun(S,rand(10))
    @test f(0.1,0.2) ≈ (C*f)(0.1,0.2)



    @test (Derivative(S,[1,0])*Fun(S,[1.]))(0.1,0.2) ≈ ((x,y)->y*(1-x-y)-x*y)(0.1,0.2)

    @test (Laplacian(S)*Fun(S,[1.]))(0.1,0.2) ≈ -2*0.1-2*0.2

    S = TriangleWeight(1,1,1,JacobiTriangle(1,1,1))


    Dx = Derivative(S,[1,0])
    Dy = Derivative(S,[0,1])

    for k=1:10
        v=[zeros(k);1.0]
        @test (Dy*Fun(S,v))(0.1,0.2) ≈ (Derivative([0,1])*Fun(Fun(S,v),JacobiTriangle(1,1,1)))(0.1,0.2)
        @test (Dx*Fun(S,v))(0.1,0.2) ≈ (Derivative([1,0])*Fun(Fun(S,v),JacobiTriangle(1,1,1)))(0.1,0.2)
    end

    Δ=Laplacian(S)

    f=Fun(S,rand(3))
    h=0.01
    QR=qr(I-h*Δ)
    @time u=\(QR,f;tolerance=1E-7)
    @time g=Fun(f,rangespace(QR))
    @time \(QR,g;tolerance=1E-7)

    # other domain
    S = TriangleWeight(1,1,1,JacobiTriangle(1,1,1))
    Dx = Derivative(S,[1,0])
    Dy = Derivative(S,[0,1])
    Div = (Dx → JacobiTriangle(0,0,0))  + (Dy → JacobiTriangle(0,0,0))
    f = Fun(S,rand(10))

    @test (Div*f)(0.1,0.2) ≈ (Dx*f)(0.1,0.2)+(Dy*f)(0.1,0.2)
    @test maxspace(rangespace(Dx),rangespace(Dy)) == rangespace(Dx + Dy) == JacobiTriangle(0,0,0)


    d = Triangle(Vec(2,3),Vec(3,4),Vec(1,6))
    S=TriangleWeight(1.,1.,1.,JacobiTriangle(1,1,1,d))
    @test domain(S) ≡ d
    @test maxspace(S,JacobiTriangle(0,0,0,d)) == JacobiTriangle(0,0,0,d)
    Dx = Derivative(S,[1,0])
    Dy = Derivative(S,[0,1])
    @test maxspace(rangespace(Dx),rangespace(Dy)) == rangespace(Dx + Dy) == JacobiTriangle(0,0,0,d)
    @test rangespace(Laplacian(S)) ≡ JacobiTriangle(1,1,1,d)
    f = Fun(S, randn(10))
    x,y = fromcanonical(S, 0.1,0.2)
    @test weight(S,x,y) ≈ 0.1*0.2*(1-0.1-0.2)
    g = Fun(f, JacobiTriangle(0,0,0,d))
    @test g(x,y) ≈ f(x,y)
    @test (Derivative([1,0])*g)(x,y) ≈ (Derivative([1,0])*f)(x,y)
    @test (Laplacian()*g)(x,y) ≈ (Laplacian()*f)(x,y)

    f=Fun(S,rand(3))
    h=0.01
    Δ = Laplacian(S)
    @test maxspace(rangespace(Δ), S) == JacobiTriangle(1,1,1,d)
    QR=qr(I-h*Δ)
    @time u=\(QR,f;tolerance=1E-7)
    @time g=Fun(f,rangespace(QR))
    @time \(QR,g;tolerance=1E-7)
end

@testset "Multiplication" begin
    x,y = Fun(Triangle())
    M_0 = Multiplication(one(x), space(x))
    testbandedblockbandedoperator(M_0)
    @test M_0*x ≈ x
    @test space(1+x) isa JacobiTriangle
    M = Multiplication(1+x, space(x))
    @test (M*y)(0.1,0.2) ≈ (1+0.1)*0.2
end
