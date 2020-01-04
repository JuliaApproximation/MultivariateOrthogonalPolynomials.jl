using ApproxFun, MultivariateOrthogonalPolynomials, Test, StaticArrays
import MultivariateOrthogonalPolynomials: DirichletTriangle
import ApproxFunBase: testbandedblockbandedoperator, testblockbandedoperator, Vec
x,y=0.1,0.2


@testset "Dirchlet conversions" begin
    C=Conversion(DirichletTriangle{1,0,0}(),JacobiTriangle(0,0,0))
        testbandedblockbandedoperator(C)
        u=C*Fun(domainspace(C),[1.0])
        @test u(0.1,0.2) ≈ 1
        u=C*Fun(domainspace(C),[0,1.0])
        @test u(0.1,0.2) ≈ 0.1*Fun(JacobiTriangle(1,0,0),[1.])(0.1,0.2)
        u=C*Fun(domainspace(C),[0,0,1.0])
        @test u(0.1,0.2) ≈ Fun(JacobiTriangle(0,0,0),[0,0,1.])(0.1,0.2)
        u=C*Fun(domainspace(C),[0,0,0,1.0])
        @test u(0.1,0.2) ≈ 0.1*Fun(JacobiTriangle(1,0,0),[0,1.])(0.1,0.2)
        u=C*Fun(domainspace(C),[0,0,0,0,1.0])
        @test u(0.1,0.2) ≈ 0.1*Fun(JacobiTriangle(1,0,0),[0,0,1.])(0.1,0.2)
        u=C*Fun(domainspace(C),[0,0,0,0,0,1.0])
        @test u(0.1,0.2) ≈ Fun(JacobiTriangle(0,0,0),[0,0,0,0,0,1.])(0.1,0.2)


    C=Conversion(DirichletTriangle{0,1,0}(),JacobiTriangle(0,0,0))
        testbandedblockbandedoperator(C)
        u=C*Fun(domainspace(C),[1.0])
        @test u(0.1,0.2) ≈ 1
        u=C*Fun(domainspace(C),[0,1.0])
        @test u(0.1,0.2) ≈ Fun(Legendre(0..1),[0,1.])(0.1)
        u=C*Fun(domainspace(C),[zeros(2);1.0])
        @test u(0.1,0.2) ≈ 0.2Fun(JacobiTriangle(0,1,0),[1.])(0.1,0.2)
        u=C*Fun(domainspace(C),[zeros(3);1.0])
        @test u(0.1,0.2) ≈ Fun(Legendre(0..1),[0,0,1.])(0.1)
        u=C*Fun(domainspace(C),[zeros(4);1.0])
        @test u(0.1,0.2) ≈ 0.2*Fun(JacobiTriangle(0,1,0),[0,1.])(0.1,0.2)
        u=C*Fun(domainspace(C),[zeros(5);1.0])
        @test u(0.1,0.2) ≈ 0.2*Fun(JacobiTriangle(0,1,0),[0,0,1.])(0.1,0.2)
        u=C*Fun(domainspace(C),[zeros(6);1.0])
        @test u(0.1,0.2) ≈ Fun(Legendre(0..1),[0,0,0,1.])(0.1)

    C=Conversion(DirichletTriangle{0,0,1}(),JacobiTriangle(0,0,0))
        testbandedblockbandedoperator(C)
        u=C*Fun(domainspace(C),[1.0])
        @test u(0.1,0.2) ≈ 1
        u=C*Fun(domainspace(C),[0,1.0])
        @test u(0.1,0.2) ≈ Fun(Legendre(0..1),[0,1.])(0.1)
        u=C*Fun(domainspace(C),[zeros(2);1.0])
        @test u(0.1,0.2) ≈ (1-x-y)*Fun(JacobiTriangle(0,0,1),[1.])(0.1,0.2)
        u=C*Fun(domainspace(C),[zeros(3);1.0])
        @test u(0.1,0.2) ≈ Fun(Legendre(0..1),[0,0,1.])(0.1)
        u=C*Fun(domainspace(C),[zeros(4);1.0])
        @test u(0.1,0.2) ≈ (1-x-y)*Fun(JacobiTriangle(0,0,1),[0,1.])(0.1,0.2)
        u=C*Fun(domainspace(C),[zeros(5);1.0])
        @test u(0.1,0.2) ≈ (1-x-y)*Fun(JacobiTriangle(0,0,1),[0,0,1.])(0.1,0.2)
        u=C*Fun(domainspace(C),[zeros(6);1.0])
        @test u(0.1,0.2) ≈ Fun(Legendre(0..1),[0,0,0,1.])(0.1)

    C=Conversion(DirichletTriangle{1,1,0}(),DirichletTriangle{0,1,0}(),JacobiTriangle(0,0,0))
        testbandedblockbandedoperator(C)
        n,k=0,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ 1
        n,k=1,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*Fun(Jacobi(1,0,0..1),[zeros(n-1);1.0])(x)
        n,k=1,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ y*Fun(JacobiTriangle(0,1,0),[zeros(sum(0:(n-1))+n-1);1.0])(x,y)
        n,k=2,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*Fun(Jacobi(1,0,0..1),[zeros(n-1);1.0])(x)
        n,k=2,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*y*Fun(JacobiTriangle(1,1,0),[zeros(sum(0:(n-2))+k-1);1.0])(x,y)
        n,k=2,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ y*Fun(JacobiTriangle(0,1,0),[zeros(sum(0:(n-1))+n-1);1.0])(x,y)
        n,k=3,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*Fun(Jacobi(1,0,0..1),[zeros(n-1);1.0])(x)
        n,k=3,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*y*Fun(JacobiTriangle(1,1,0),[zeros(sum(0:(n-2))+k-1);1.0])(x,y)
        n,k=3,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*y*Fun(JacobiTriangle(1,1,0),[zeros(sum(0:(n-2))+k-1);1.0])(x,y)
        n,k=3,3;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ y*Fun(JacobiTriangle(0,1,0),[zeros(sum(0:(n-1))+n-1);1.0])(x,y)

    C̃=Conversion(DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),JacobiTriangle(0,0,0))
        testbandedblockbandedoperator(C̃)
        @test C[1:20,1:20] ≈ C̃[1:20,1:20]



    C=Conversion(DirichletTriangle{1,0,1}(),DirichletTriangle{0,0,1}(),JacobiTriangle(0,0,0))
        testbandedblockbandedoperator(C)
        n,k=0,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ 1
        n,k=1,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*Fun(Jacobi(1,0,0..1),[zeros(n-1);1.0])(x)
        n,k=1,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ (1-x-y)*Fun(JacobiTriangle(0,0,1),[zeros(sum(0:(n-1))+n-1);1.0])(x,y)
        n,k=2,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*Fun(Jacobi(1,0,0..1),[zeros(n-1);1.0])(x)
        n,k=2,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*(1-x-y)*Fun(JacobiTriangle(1,0,1),[zeros(sum(0:(n-2))+k-1);1.0])(x,y)
        n,k=2,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ (1-x-y)*Fun(JacobiTriangle(0,0,1),[zeros(sum(0:(n-1))+n-1);1.0])(x,y)
        n,k=3,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*Fun(Jacobi(1,0,0..1),[zeros(n-1);1.0])(x)
        n,k=3,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*(1-x-y)*Fun(JacobiTriangle(1,0,1),[zeros(sum(0:(n-2))+k-1);1.0])(x,y)
        n,k=3,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*(1-x-y)*Fun(JacobiTriangle(1,0,1),[zeros(sum(0:(n-2))+k-1);1.0])(x,y)
        n,k=3,3;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ (1-x-y)*Fun(JacobiTriangle(0,0,1),[zeros(sum(0:(n-1))+n-1);1.0])(x,y)


    C̃=Conversion(DirichletTriangle{1,0,1}(),DirichletTriangle{1,0,0}(),JacobiTriangle(0,0,0))
        testbandedblockbandedoperator(C̃)
        @test C[1:20,1:20] ≈ C̃[1:20,1:20]


    C=Conversion(DirichletTriangle{0,1,1}(),DirichletTriangle{0,1,0}(),JacobiTriangle(0,0,0))
        testbandedblockbandedoperator(C)
        n,k=0,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ 1
        n,k=1,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ (1-x)*Fun(Jacobi(0,1,0..1),[zeros(n-1);1.0])(x)
        n,k=1,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ (1-x-2y)*Fun(Jacobi(0,1,0..1),[zeros(n-1);1.0])(x)
        n,k=2,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ (1-x)*Fun(Jacobi(0,1,0..1),[zeros(n-1);1.0])(x)
        n,k=2,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ (1-x-2y)*Fun(Jacobi(0,1,0..1),[zeros(n-1);1.0])(x)
        n,k=2,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ y*(1-x-y)*Fun(JacobiTriangle(0,1,1),[zeros(sum(0:(n-2))+k-2);1.0])(x,y)
        n,k=3,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ (1-x)*Fun(Jacobi(0,1,0..1),[zeros(n-1);1.0])(x)
        n,k=3,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ (1-x-2y)*Fun(Jacobi(0,1,0..1),[zeros(n-1);1.0])(x)
        n,k=3,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ y*(1-x-y)*Fun(JacobiTriangle(0,1,1),[zeros(sum(0:(n-2))+k-2);1.0])(x,y)
        n,k=3,3;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ y*(1-x-y)*Fun(JacobiTriangle(0,1,1),[zeros(sum(0:(n-2))+k-2);1.0])(x,y)

    C̃=Conversion(DirichletTriangle{0,1,1}(),DirichletTriangle{0,0,1}(),JacobiTriangle(0,0,0))
        testbandedblockbandedoperator(C̃)
        @test C[1:20,1:20] ≈ C̃[1:20,1:20]


    C=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{0,1,1}(),DirichletTriangle{0,0,1}(),JacobiTriangle(0,0,0))
        testbandedblockbandedoperator(C)
        n,k=0,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ 1
        n,k=1,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ 1-2x
        n,k=1,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ 1-x-2y
        n,k=2,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*(1-x)*Fun(Jacobi(1,1,0..1),[zeros(n-2);1.0])(x)
        n,k=2,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*(1-x-2y)*Fun(Jacobi(1,1,0..1),[zeros(n-2);1.0])(x)
        n,k=2,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ y*(1-x-y)*Fun(JacobiTriangle(0,1,1),[zeros(sum(0:(n-2))+n-2);1.0])(x,y)
        n,k=3,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*(1-x)*Fun(Jacobi(1,1,0..1),[zeros(n-2);1.0])(x)
        n,k=3,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*(1-x-2y)*Fun(Jacobi(1,1,0..1),[zeros(n-2);1.0])(x)
        n,k=3,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*y*(1-x-y)*Fun(JacobiTriangle(1,1,1),[zeros(sum(0:(n-3))+k-2);1.0])(x,y)
        n,k=3,3;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ y*(1-x-y)*Fun(JacobiTriangle(0,1,1),[zeros(sum(0:(n-2))+n-2);1.0])(x,y)
        n,k=4,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*(1-x)*Fun(Jacobi(1,1,0..1),[zeros(n-2);1.0])(x)
        n,k=4,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*(1-x-2y)*Fun(Jacobi(1,1,0..1),[zeros(n-2);1.0])(x)
        n,k=4,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*y*(1-x-y)*Fun(JacobiTriangle(1,1,1),[zeros(sum(0:(n-3))+k-2);1.0])(x,y)
        n,k=4,3;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ x*y*(1-x-y)*Fun(JacobiTriangle(1,1,1),[zeros(sum(0:(n-3))+k-2);1.0])(x,y)
        n,k=4,4;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
        @test u(x,y) ≈ y*(1-x-y)*Fun(JacobiTriangle(0,1,1),[zeros(sum(0:(n-2))+n-2);1.0])(x,y)

    C̃=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,0,1}(),DirichletTriangle{0,0,1}(),JacobiTriangle(0,0,0))
        testbandedblockbandedoperator(C̃)
        @test C[1:20,1:20] ≈ C̃[1:20,1:20]

    C̃=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,0,1}(),DirichletTriangle{1,0,0}(),JacobiTriangle(0,0,0))
        @test C[1:20,1:20] ≈ C̃[1:20,1:20]

    C̃=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{0,1,0}(),JacobiTriangle(0,0,0))
        @test C[1:20,1:20] ≈ C̃[1:20,1:20]

    C̃=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),JacobiTriangle(0,0,0))
        @test C[1:20,1:20] ≈ C̃[1:20,1:20]
end

@testset "Dirichlet evaluation" begin
    f=Fun((x,y)->exp(x-0.12*cos(y)),JacobiTriangle(0,0,0))

    C=Conversion(DirichletTriangle{1,0,0}(),JacobiTriangle(0,0,0))
    R=Conversion(domainspace(C),Legendre(Segment(Vec(0.,0.),Vec(0.,1.))))
    u=C\f
    @test (R*u)(0.,0.3) ≈ f(0.,0.3)
    @test u(0.1,0.2) ≈ f(0.1,0.2)


    C=Conversion(DirichletTriangle{0,1,0}(),JacobiTriangle(0,0,0))
    R=Conversion(domainspace(C),Legendre(Segment(Vec(0.,0.),Vec(1.,0.))))
    u=C\f
    @test (R*u)(0.3,0.) ≈ f(0.3,0.)
    @test u(0.1,0.2) ≈ f(0.1,0.2)

    C=Conversion(DirichletTriangle{0,0,1}(),JacobiTriangle(0,0,0))
    R=Conversion(domainspace(C),Legendre(Segment(Vec(0.,1.),Vec(1.,0.))))
    u=C\f
    @test (R*u)(0.3,1-0.3) ≈ f(0.3,1-0.3)
    @test u(0.1,0.2) ≈ f(0.1,0.2)

    C=Conversion(DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),JacobiTriangle(0,0,0))
    Rx=Conversion(DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),Legendre(Segment(Vec(0.,0.),Vec(0.,1.))))
    Ry=Conversion(DirichletTriangle{1,1,0}(),DirichletTriangle{0,1,0}(),Legendre(Segment(Vec(0.,0.),Vec(1.,0.))))
    u=C\f
    @test (Rx*u)(0.,0.3) ≈ f(0.,0.3)
    @test (Ry*u)(0.3,0.) ≈ f(0.3,0.)
    @test u(0.1,0.2) ≈ f(0.1,0.2)

    C=Conversion(DirichletTriangle{1,0,1}(),DirichletTriangle{1,0,0}(),JacobiTriangle(0,0,0))
    Rx=Conversion(DirichletTriangle{1,0,1}(),DirichletTriangle{1,0,0}(),Legendre(Segment(Vec(0.,0.),Vec(0.,1.))))
    Rz=Conversion(DirichletTriangle{1,0,1}(),DirichletTriangle{0,0,1}(),Legendre(Segment(Vec(0.,1.),Vec(1.,0.))))
    u=C\f
    @test (Rx*u)(0.,0.3) ≈ f(0.,0.3)
    @test (Rz*u)(0.3,1-0.3) ≈ f(0.3,1-0.3)
    @test u(0.1,0.2) ≈ f(0.1,0.2)

    C=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),JacobiTriangle(0,0,0))
    Rx=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),Legendre(Segment(Vec(0.,0.),Vec(0.,1.))))
    Ry=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{0,1,0}(),Legendre(Segment(Vec(0.,0.),Vec(1.,0.))))
    Rz=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{0,1,1}(),DirichletTriangle{0,0,1}(),Legendre(Segment(Vec(0.,1.),Vec(1.,0.))))
    u=C\f
    @test (Rx*u)(0.,0.3) ≈ f(0.,0.3)
    @test (Ry*u)(0.3,0.) ≈ f(0.3,0.)
    @test (Rz*u)(0.3,1-0.3) ≈ f(0.3,1-0.3)
    @test u(0.1,0.2) ≈ f(0.1,0.2)
end

@testset "Dirichlet restriction" begin
    f=Fun((x,y)->exp(x-0.12*cos(y)),JacobiTriangle(0,0,0))
    C=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),JacobiTriangle(0,0,0))
    u=C\f

    B=Dirichlet(DirichletTriangle{1,1,1}())
    testblockbandedoperator(B)
    ∂u=B*u
    @test ∂u(0.3,0.)≈ f(0.3,0.)
    @test ∂u(0.,0.3)≈ f(0.,0.3)
    @test ∂u(0.3,1-0.3)≈ f(0.3,1-0.3)

    d = Triangle(Vec(2,3),Vec(3,4),Vec(1,6))
    f=Fun((x,y)->exp(x-0.12*cos(y)),JacobiTriangle(0,0,0,d))
    C=Conversion(DirichletTriangle{1,1,1}(d),DirichletTriangle{1,1,0}(d),DirichletTriangle{1,0,0}(d),JacobiTriangle(0,0,0,d))
    u=C\f

    B=Dirichlet(DirichletTriangle{1,1,1}(d))
    testblockbandedoperator(B)
    ∂u=B*u
    @test ∂u(2.5,3.5)≈ f(2.5,3.5)
    @test ∂u(1.5,4.5)≈ f(1.5,4.5)
    @test ∂u(2,5)≈ f(2,5)

end

@testset "Triangle Dirichlet Derivatives" begin
    @testset "Triangle() Derivative" begin
        S = DirichletTriangle{1,0,1}()
        Dx = Derivative(S, [1,0])
        testbandedblockbandedoperator(Dx)
        @test rangespace(Dx) == JacobiTriangle()
        f = Fun(S, randn(20))
        @test (Dx*f)(0.1,0.2) ≈ (Derivative([1,0])*Fun(f,JacobiTriangle(0,0,0)))(0.1,0.2)

        Dx = Derivative(S, [2,0])
        testbandedblockbandedoperator(Dx)
        @test rangespace(Dx) == JacobiTriangle(1,0,1)
        f = Fun(S, randn(20))
        @test (Dx*f)(0.1,0.2) ≈ (Derivative([2,0])*Fun(f,JacobiTriangle(0,0,0)))(0.1,0.2)

        Dx = Derivative(S, [2,1])
        testbandedblockbandedoperator(Dx)
        @test rangespace(Dx) == JacobiTriangle(1,1,2)
        f = Fun(S, randn(20))
        @test (Dx*f)(0.1,0.2) ≈ (Derivative([2,1])*Fun(f,JacobiTriangle(0,0,0)))(0.1,0.2)

        S = DirichletTriangle{0,1,1}()
        Dy = Derivative(S, [0,1])
        testbandedblockbandedoperator(Dy)
        @test rangespace(Dy) == JacobiTriangle()
        f = Fun(S, randn(20))
        @test (Dy*f)(0.1,0.2) ≈ (Derivative([0,1])*Fun(f,JacobiTriangle(0,0,0)))(0.1,0.2)

        Dx = Derivative(S, [1,0])
        testbandedblockbandedoperator(Dx)
        @test rangespace(Dx) == JacobiTriangle(1,0,1)
        f = Fun(S, randn(20))
        @test (Dx*f)(0.1,0.2) ≈ (Derivative([1,0])*Fun(f,JacobiTriangle(0,0,0)))(0.1,0.2)

        Dx = Derivative(S, [2,0])
        testbandedblockbandedoperator(Dx)
        @test rangespace(Dx) == JacobiTriangle(2,0,2)
        f = Fun(S, randn(20))
        @test (Dx*f)(0.1,0.2) ≈ (Derivative([2,0])*Fun(f,JacobiTriangle(0,0,0)))(0.1,0.2)

        Dx = Derivative(S, [1,1])
        testbandedblockbandedoperator(Dx)
        @test rangespace(Dx) == JacobiTriangle(1,0,1)
        f = Fun(S, randn(20))
        @test (Dx*f)(0.1,0.2) ≈ (Derivative([1,1])*Fun(f,JacobiTriangle(0,0,0)))(0.1,0.2)


        S = DirichletTriangle{1,1,1}()
        Dx = Derivative(S,[1,0])
        @test rangespace(Dx) == JacobiTriangle()
        testbandedblockbandedoperator(Dx)
        f = Fun(S, randn(20))
        @test (Dx*f)(0.1,0.2) ≈ (Derivative([1,0])*Fun(f,JacobiTriangle(0,0,0)))(0.1,0.2)
        Dy = Derivative(S, [0,1])
        testbandedblockbandedoperator(Dy)
        @test rangespace(Dy) == JacobiTriangle()
        f = Fun(S, randn(20))
        @test (Dy*f)(0.1,0.2) ≈ (Derivative([0,1])*Fun(f,JacobiTriangle(0,0,0)))(0.1,0.2)
    end

    @testset "Other triangle Derivative" begin
        d = Triangle(Vec(2,3),Vec(3,4),Vec(1,6))
        S = DirichletTriangle{1,0,1}(d)
        Dx = Derivative(S, [1,0])
        testbandedblockbandedoperator(Dx)
        @test rangespace(Dx) == JacobiTriangle(1,1,1,d)
        f = Fun(S, randn(20))
        @test (Dx*f)(0.1,0.2) ≈ (Derivative([1,0])*Fun(f,JacobiTriangle(0,0,0,d)))(0.1,0.2)

        Dx = Derivative(S, [2,0])
        testbandedblockbandedoperator(Dx)
        @test rangespace(Dx) == JacobiTriangle(2,2,2,d)
        f = Fun(S, randn(20))
        @test (Dx*f)(0.1,0.2) ≈ (Derivative([2,0])*Fun(f,JacobiTriangle(0,0,0,d)))(0.1,0.2)

        Dx = Derivative(S, [2,1])
        testbandedblockbandedoperator(Dx)
        @test rangespace(Dx) == JacobiTriangle(3,3,3,d)
        f = Fun(S, randn(20))
        @test (Dx*f)(0.1,0.2) ≈ (Derivative([2,1])*Fun(f,JacobiTriangle(0,0,0,d)))(0.1,0.2)

        S = DirichletTriangle{0,1,1}(d)
        Dy = Derivative(S, [0,1])
        testbandedblockbandedoperator(Dy)
        @test rangespace(Dy) == JacobiTriangle(1,1,1,d)
        f = Fun(S, randn(20))
        @test (Dy*f)(0.1,0.2) ≈ (Derivative([0,1])*Fun(f,JacobiTriangle(0,0,0,d)))(0.1,0.2)

        Dx = Derivative(S, [1,0])
        testbandedblockbandedoperator(Dx)
        @test rangespace(Dx) == JacobiTriangle(1,1,1,d)
        f = Fun(S, randn(20))
        @test (Dx*f)(0.1,0.2) ≈ (Derivative([1,0])*Fun(f,JacobiTriangle(0,0,0,d)))(0.1,0.2)


        S = DirichletTriangle{1,1,1}(d)
        Dx = Derivative(S,[1,0])
        @test rangespace(Dx) == JacobiTriangle(0,0,0,d)
        testbandedblockbandedoperator(Dx)
        f = Fun(S, randn(20))
        @test (Dx*f)(0.1,0.2) ≈ (Derivative([1,0])*Fun(f,JacobiTriangle(0,0,0,d)))(0.1,0.2)
        Dy = Derivative(S, [0,1])
        testbandedblockbandedoperator(Dy)
        @test rangespace(Dy) == JacobiTriangle(0,0,0,d)
        f = Fun(S, randn(20))
        @test (Dy*f)(0.1,0.2) ≈ (Derivative([0,1])*Fun(f,JacobiTriangle(0,0,0,d)))(0.1,0.2)
    end
end

@testset "Triangle Dirichlet Laplacian" begin
    @testset "Triangle Laplace" begin
        S = DirichletTriangle{1,1,1}()
        Δ=Laplacian(S)
        testbandedblockbandedoperator(Δ)
        @test rangespace(Δ) == JacobiTriangle(1,1,1)
        f = Fun(S, randn(20))
        @test (Δ*f)(0.1,0.2) ≈ (Laplacian()*Fun(f,JacobiTriangle(0,0,0)))(0.1,0.2)
        B=Dirichlet(S)
        f=Fun((x,y)->real(exp(x+im*y)),rangespace(B))
        u=\([B;Δ],[f;0.];tolerance=1E-13)
        @test u(0.1,0.2) ≈ real(exp(0.1+0.2im))
    end

    @testset "Triangle Laplace via JacobiTriangle(0,0,0)" begin
        C=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),JacobiTriangle(0,0,0))
        B=Dirichlet(DirichletTriangle{1,1,1}())
        Δ=Laplacian(rangespace(C))
        f=Fun((x,y)->real(exp(x+im*y)),rangespace(B))
        u=\([B;Δ*C],[f;0.];tolerance=1E-13)

        @test u(0.1,0.2) ≈ real(exp(0.1+0.2im))
    end

    @testset "Other triangle Laplace" begin
        d = Triangle(Vec(2,3),Vec(3,4),Vec(1,6))
        S = DirichletTriangle{1,1,1}(d)
        Δ=Laplacian(S)
        @test rangespace(Δ) == JacobiTriangle(1,1,1,d)
        testbandedblockbandedoperator(Δ)
        f = Fun(S, randn(5))
        x,y = fromcanonical(S, 0.1,0.2)
        @test (Δ*f)(x,y) ≈ (Laplacian()*Fun(f, JacobiTriangle(0,0,0,d)))(0.1,0.2)

        B = Dirichlet(d)
        f=Fun((x,y)->real(exp(x+im*y)),rangespace(B))
        u=\([B;Δ],[f;0.];tolerance=1E-13)

        @test u(x,y) ≈ real(exp(x+y*im))
    end
end
