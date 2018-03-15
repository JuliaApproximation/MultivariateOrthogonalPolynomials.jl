using ApproxFun, MultivariateOrthogonalPolynomials, Base.Test, StaticArrays
    import MultivariateOrthogonalPolynomials: DirichletTriangle
    import ApproxFun: testbandedblockbandedoperator, testblockbandedoperator
    x,y=0.1,0.2

C=Conversion(DirichletTriangle{1,0,0}(),KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(C)
    u=C*Fun(domainspace(C),[1.0])
    @test u(0.1,0.2) ≈ 1
    u=C*Fun(domainspace(C),[0,1.0])
    @test u(0.1,0.2) ≈ 0.1*Fun(KoornwinderTriangle(1,0,0),[1.])(0.1,0.2)
    u=C*Fun(domainspace(C),[0,0,1.0])
    @test u(0.1,0.2) ≈ Fun(KoornwinderTriangle(0,0,0),[0,0,1.])(0.1,0.2)
    u=C*Fun(domainspace(C),[0,0,0,1.0])
    @test u(0.1,0.2) ≈ 0.1*Fun(KoornwinderTriangle(1,0,0),[0,1.])(0.1,0.2)
    u=C*Fun(domainspace(C),[0,0,0,0,1.0])
    @test u(0.1,0.2) ≈ 0.1*Fun(KoornwinderTriangle(1,0,0),[0,0,1.])(0.1,0.2)
    u=C*Fun(domainspace(C),[0,0,0,0,0,1.0])
    @test u(0.1,0.2) ≈ Fun(KoornwinderTriangle(0,0,0),[0,0,0,0,0,1.])(0.1,0.2)


C=Conversion(DirichletTriangle{0,1,0}(),KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(C)
    u=C*Fun(domainspace(C),[1.0])
    @test u(0.1,0.2) ≈ 1
    u=C*Fun(domainspace(C),[0,1.0])
    @test u(0.1,0.2) ≈ Fun(Legendre(0..1),[0,1.])(0.1)
    u=C*Fun(domainspace(C),[zeros(2);1.0])
    @test u(0.1,0.2) ≈ 0.2Fun(KoornwinderTriangle(0,1,0),[1.])(0.1,0.2)
    u=C*Fun(domainspace(C),[zeros(3);1.0])
    @test u(0.1,0.2) ≈ Fun(Legendre(0..1),[0,0,1.])(0.1)
    u=C*Fun(domainspace(C),[zeros(4);1.0])
    @test u(0.1,0.2) ≈ 0.2*Fun(KoornwinderTriangle(0,1,0),[0,1.])(0.1,0.2)
    u=C*Fun(domainspace(C),[zeros(5);1.0])
    @test u(0.1,0.2) ≈ 0.2*Fun(KoornwinderTriangle(0,1,0),[0,0,1.])(0.1,0.2)
    u=C*Fun(domainspace(C),[zeros(6);1.0])
    @test u(0.1,0.2) ≈ Fun(Legendre(0..1),[0,0,0,1.])(0.1)

C=Conversion(DirichletTriangle{0,0,1}(),KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(C)
    u=C*Fun(domainspace(C),[1.0])
    @test u(0.1,0.2) ≈ 1
    u=C*Fun(domainspace(C),[0,1.0])
    @test u(0.1,0.2) ≈ Fun(Legendre(0..1),[0,1.])(0.1)
    u=C*Fun(domainspace(C),[zeros(2);1.0])
    @test u(0.1,0.2) ≈ (1-x-y)*Fun(KoornwinderTriangle(0,0,1),[1.])(0.1,0.2)
    u=C*Fun(domainspace(C),[zeros(3);1.0])
    @test u(0.1,0.2) ≈ Fun(Legendre(0..1),[0,0,1.])(0.1)
    u=C*Fun(domainspace(C),[zeros(4);1.0])
    @test u(0.1,0.2) ≈ (1-x-y)*Fun(KoornwinderTriangle(0,0,1),[0,1.])(0.1,0.2)
    u=C*Fun(domainspace(C),[zeros(5);1.0])
    @test u(0.1,0.2) ≈ (1-x-y)*Fun(KoornwinderTriangle(0,0,1),[0,0,1.])(0.1,0.2)
    u=C*Fun(domainspace(C),[zeros(6);1.0])
    @test u(0.1,0.2) ≈ Fun(Legendre(0..1),[0,0,0,1.])(0.1)

C=Conversion(DirichletTriangle{1,1,0}(),DirichletTriangle{0,1,0}(),KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(C)
    n,k=0,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ 1
    n,k=1,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*Fun(Jacobi(1,0,0..1),[zeros(n-1);1.0])(x)
    n,k=1,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ y*Fun(KoornwinderTriangle(0,1,0),[zeros(sum(0:(n-1))+n-1);1.0])(x,y)
    n,k=2,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*Fun(Jacobi(1,0,0..1),[zeros(n-1);1.0])(x)
    n,k=2,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*y*Fun(KoornwinderTriangle(1,1,0),[zeros(sum(0:(n-2))+k-1);1.0])(x,y)
    n,k=2,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ y*Fun(KoornwinderTriangle(0,1,0),[zeros(sum(0:(n-1))+n-1);1.0])(x,y)
    n,k=3,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*Fun(Jacobi(1,0,0..1),[zeros(n-1);1.0])(x)
    n,k=3,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*y*Fun(KoornwinderTriangle(1,1,0),[zeros(sum(0:(n-2))+k-1);1.0])(x,y)
    n,k=3,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*y*Fun(KoornwinderTriangle(1,1,0),[zeros(sum(0:(n-2))+k-1);1.0])(x,y)
    n,k=3,3;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ y*Fun(KoornwinderTriangle(0,1,0),[zeros(sum(0:(n-1))+n-1);1.0])(x,y)

C̃=Conversion(DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(C̃)
    @test C[1:20,1:20] ≈ C̃[1:20,1:20]



C=Conversion(DirichletTriangle{1,0,1}(),DirichletTriangle{0,0,1}(),KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(C)
    n,k=0,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ 1
    n,k=1,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*Fun(Jacobi(1,0,0..1),[zeros(n-1);1.0])(x)
    n,k=1,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ (1-x-y)*Fun(KoornwinderTriangle(0,0,1),[zeros(sum(0:(n-1))+n-1);1.0])(x,y)
    n,k=2,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*Fun(Jacobi(1,0,0..1),[zeros(n-1);1.0])(x)
    n,k=2,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*(1-x-y)*Fun(KoornwinderTriangle(1,0,1),[zeros(sum(0:(n-2))+k-1);1.0])(x,y)
    n,k=2,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ (1-x-y)*Fun(KoornwinderTriangle(0,0,1),[zeros(sum(0:(n-1))+n-1);1.0])(x,y)
    n,k=3,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*Fun(Jacobi(1,0,0..1),[zeros(n-1);1.0])(x)
    n,k=3,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*(1-x-y)*Fun(KoornwinderTriangle(1,0,1),[zeros(sum(0:(n-2))+k-1);1.0])(x,y)
    n,k=3,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*(1-x-y)*Fun(KoornwinderTriangle(1,0,1),[zeros(sum(0:(n-2))+k-1);1.0])(x,y)
    n,k=3,3;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ (1-x-y)*Fun(KoornwinderTriangle(0,0,1),[zeros(sum(0:(n-1))+n-1);1.0])(x,y)


C̃=Conversion(DirichletTriangle{1,0,1}(),DirichletTriangle{1,0,0}(),KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(C̃)
    @test C[1:20,1:20] ≈ C̃[1:20,1:20]


C=Conversion(DirichletTriangle{0,1,1}(),DirichletTriangle{0,1,0}(),KoornwinderTriangle(0,0,0))
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
    @test u(x,y) ≈ y*(1-x-y)*Fun(KoornwinderTriangle(0,1,1),[zeros(sum(0:(n-2))+k-2);1.0])(x,y)
    n,k=3,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ (1-x)*Fun(Jacobi(0,1,0..1),[zeros(n-1);1.0])(x)
    n,k=3,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ (1-x-2y)*Fun(Jacobi(0,1,0..1),[zeros(n-1);1.0])(x)
    n,k=3,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ y*(1-x-y)*Fun(KoornwinderTriangle(0,1,1),[zeros(sum(0:(n-2))+k-2);1.0])(x,y)
    n,k=3,3;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ y*(1-x-y)*Fun(KoornwinderTriangle(0,1,1),[zeros(sum(0:(n-2))+k-2);1.0])(x,y)

C̃=Conversion(DirichletTriangle{0,1,1}(),DirichletTriangle{0,0,1}(),KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(C̃)
    @test C[1:20,1:20] ≈ C̃[1:20,1:20]


C=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{0,1,1}(),DirichletTriangle{0,0,1}(),KoornwinderTriangle(0,0,0))
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
    @test u(x,y) ≈ y*(1-x-y)*Fun(KoornwinderTriangle(0,1,1),[zeros(sum(0:(n-2))+n-2);1.0])(x,y)
    n,k=3,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*(1-x)*Fun(Jacobi(1,1,0..1),[zeros(n-2);1.0])(x)
    n,k=3,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*(1-x-2y)*Fun(Jacobi(1,1,0..1),[zeros(n-2);1.0])(x)
    n,k=3,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*y*(1-x-y)*Fun(KoornwinderTriangle(1,1,1),[zeros(sum(0:(n-3))+k-2);1.0])(x,y)
    n,k=3,3;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ y*(1-x-y)*Fun(KoornwinderTriangle(0,1,1),[zeros(sum(0:(n-2))+n-2);1.0])(x,y)
    n,k=4,0;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*(1-x)*Fun(Jacobi(1,1,0..1),[zeros(n-2);1.0])(x)
    n,k=4,1;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*(1-x-2y)*Fun(Jacobi(1,1,0..1),[zeros(n-2);1.0])(x)
    n,k=4,2;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*y*(1-x-y)*Fun(KoornwinderTriangle(1,1,1),[zeros(sum(0:(n-3))+k-2);1.0])(x,y)
    n,k=4,3;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ x*y*(1-x-y)*Fun(KoornwinderTriangle(1,1,1),[zeros(sum(0:(n-3))+k-2);1.0])(x,y)
    n,k=4,4;u=C*Fun(domainspace(C),[zeros(sum(0:n)+k);1.0])
    @test u(x,y) ≈ y*(1-x-y)*Fun(KoornwinderTriangle(0,1,1),[zeros(sum(0:(n-2))+n-2);1.0])(x,y)

C̃=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,0,1}(),DirichletTriangle{0,0,1}(),KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(C̃)
    @test C[1:20,1:20] ≈ C̃[1:20,1:20]

C̃=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,0,1}(),DirichletTriangle{1,0,0}(),KoornwinderTriangle(0,0,0))
    @test C[1:20,1:20] ≈ C̃[1:20,1:20]

C̃=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{0,1,0}(),KoornwinderTriangle(0,0,0))
    @test C[1:20,1:20] ≈ C̃[1:20,1:20]

C̃=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),KoornwinderTriangle(0,0,0))
    @test C[1:20,1:20] ≈ C̃[1:20,1:20]



f=Fun((x,y)->exp(x-0.12*cos(y)),KoornwinderTriangle(0,0,0))

C=Conversion(DirichletTriangle{1,0,0}(),KoornwinderTriangle(0,0,0))
R=Conversion(domainspace(C),Legendre(Vec(0.,0.)..Vec(0.,1.)))
u=C\f
@test (R*u)(0.,0.3) ≈ f(0.,0.3)
@test u(0.1,0.2) ≈ f(0.1,0.2)


C=Conversion(DirichletTriangle{0,1,0}(),KoornwinderTriangle(0,0,0))
R=Conversion(domainspace(C),Legendre(Vec(0.,0.)..Vec(1.,0.)))
u=C\f
@test (R*u)(0.3,0.) ≈ f(0.3,0.)
@test u(0.1,0.2) ≈ f(0.1,0.2)

C=Conversion(DirichletTriangle{0,0,1}(),KoornwinderTriangle(0,0,0))
R=Conversion(domainspace(C),Legendre(Vec(0.,1.)..Vec(1.,0.)))
u=C\f
@test (R*u)(0.3,1-0.3) ≈ f(0.3,1-0.3)
@test u(0.1,0.2) ≈ f(0.1,0.2)

C=Conversion(DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),KoornwinderTriangle(0,0,0))
Rx=Conversion(DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),Legendre(Vec(0.,0.)..Vec(0.,1.)))
Ry=Conversion(DirichletTriangle{1,1,0}(),DirichletTriangle{0,1,0}(),Legendre(Vec(0.,0.)..Vec(1.,0.)))
u=C\f
@test (Rx*u)(0.,0.3) ≈ f(0.,0.3)
@test (Ry*u)(0.3,0.) ≈ f(0.3,0.)
@test u(0.1,0.2) ≈ f(0.1,0.2)

C=Conversion(DirichletTriangle{1,0,1}(),DirichletTriangle{1,0,0}(),KoornwinderTriangle(0,0,0))
Rx=Conversion(DirichletTriangle{1,0,1}(),DirichletTriangle{1,0,0}(),Legendre(Vec(0.,0.)..Vec(0.,1.)))
Rz=Conversion(DirichletTriangle{1,0,1}(),DirichletTriangle{0,0,1}(),Legendre(Vec(0.,1.)..Vec(1.,0.)))
u=C\f
@test (Rx*u)(0.,0.3) ≈ f(0.,0.3)
@test (Rz*u)(0.3,1-0.3) ≈ f(0.3,1-0.3)
@test u(0.1,0.2) ≈ f(0.1,0.2)

C=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),KoornwinderTriangle(0,0,0))
Rx=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),Legendre(Vec(0.,0.)..Vec(0.,1.)))
Ry=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{0,1,0}(),Legendre(Vec(0.,0.)..Vec(1.,0.)))
Rz=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{0,1,1}(),DirichletTriangle{0,0,1}(),Legendre(Vec(0.,1.)..Vec(1.,0.)))
u=C\f
@test (Rx*u)(0.,0.3) ≈ f(0.,0.3)
@test (Ry*u)(0.3,0.) ≈ f(0.3,0.)
@test (Rz*u)(0.3,1-0.3) ≈ f(0.3,1-0.3)
@test u(0.1,0.2) ≈ f(0.1,0.2)

C=Conversion(DirichletTriangle{1,1,1}(),DirichletTriangle{1,1,0}(),DirichletTriangle{1,0,0}(),KoornwinderTriangle(0,0,0))
B=Dirichlet(DirichletTriangle{1,1,1}())
testblockbandedoperator(B)
u=C\f
∂u=B*u
@test ∂u(0.3,0.)≈ f(0.3,0.)
@test ∂u(0.,0.3)≈ f(0.,0.3)
@test ∂u(0.3,1-0.3)≈ f(0.3,1-0.3)

Δ=Laplacian(DirichletTriangle{1,1,1}())
testbandedblockbandedoperator(Δ)



Δ=Laplacian(rangespace(C))
f=Fun((x,y)->real(exp(x+im*y)),rangespace(B))
u=\([B;Δ*C],[f;0.];tolerance=1E-13)

@test (C*u)(0.1,0.2) ≈ real(exp(0.1+0.2im))
@test u(0.1,0.2) ≈ real(exp(0.1+0.2im))

@testset "Triangle Dirichlet Laplacian" begin
    d = Triangle()
    B = Dirichlet(d)
    Δ = Laplacian(domainspace(B))

    L = [B;Δ]

    ∂u = Fun((x,y) -> exp(x)*sin(y), ∂(d))
    Fun([∂u;0],rangespace(L))

    L = [B;Laplacian()]
    Fun([∂u;0],rangespace(L))
end
