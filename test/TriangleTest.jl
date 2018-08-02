using StaticArrays, Plots, BandedMatrices, FastTransforms,
        ApproxFun, MultivariateOrthogonalPolynomials, Compat.Test
    import MultivariateOrthogonalPolynomials: Lowering, ProductTriangle, clenshaw, block, TriangleWeight,plan_evaluate, weight
    import ApproxFun: testbandedblockbandedoperator, Block, BandedBlockBandedMatrix, blockcolrange, blocksize, Vec


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


# Use DuffyMap with

struct DuffyTriangle{S,T} <: Space{Triangle,T}
    space::S
    domain::Triangle
    function DuffyTriangle{S,T}(s::S, d::Triangle) where {S,T}
        @assert domain(s) == Interval(0,1)^2
        new{S,T}(s, d)
    end
end

DuffyTriangle(s::S, d) where S = DuffyTriangle{S,ApproxFun.rangetype(S)}(s, d)
DuffyTriangle() = DuffyTriangle(Chebyshev(0..1)^2, Triangle())
DuffyTriangle(d::Triangle) = DuffyTriangle(Chebyshev()^2, d)

function points(S::DuffyTriangle, N)
    pts = points(S.space, N)
    fromcanonical.(S.domain, iduffy.(pts))
end

plan_transform(S::DuffyTriangle, n::Integer) = TransformPlan(S, plan_transform(S.space,n), Val{false})
plan_transform(S::DuffyTriangle, n::AbstractVector) = TransformPlan(S, plan_transform(S.space,n), Val{false})

evaluate(cfs::AbstractVector, S::DuffyTriangle, x) = evaluate(cfs, S.space, duffy(tocanonical(S.domain,x)))
using FastTransforms
ff = Fun(KoornwinderTriangle(0.0,-0.5,-0.5),[zeros(2); 1.0])
    @time f = Fun((x,y) -> ff(x,y), DuffyTriangle(), 20)
    F = ApproxFun.coefficientmatrix(Fun(space(f).space, coefficients(f)))
    cheb2tri(F,0.0,-0.5,-0.5)

P(1,1,0.1,0.2)/Fun(ff, KoornwinderTriangle(0.0,0.5,0.5))(0.1,0.2)
P(1,1,0.2,0.3)/Fun(ff, KoornwinderTriangle(0.0,0.5,0.5))(0.2,0.3)

P(1,0,0.1,0.2)/ff(0.1,0.2)
P(1,0,0.2,0.3)/ff(0.2,0.3)


ff(0.1,0.2)
ff(0.2,0.3)
Fun(ff, KoornwinderTriangle(0.0,0.5,0.5))(0.1,0.2)
Fun(ff, KoornwinderTriangle(0.0,0.5,0.5))(0.2,0.3)
S1 = space(ff)
S2 = KoornwinderTriangle(0.0,0.5,0.5)
C = Conversion(S, S2)

Jx,Jy = jacobioperators(S1)
testbandedblockbandedoperator(Jy)
Fun(ff, KoornwinderTriangle(0.0,0.5,0.5))(0.1,0.2)
Fun((Jx*ff), KoornwinderTriangle(0.0,0.5,0.5))(0.1,0.2)
Fun((Jy*ff), KoornwinderTriangle(0.0,0.5,0.5))(0.1,0.2)

J2x,J2y = jacobioperators(S2)

using SO
C[1:100,1:100]\(J2x*C)[1:100,1:100] - Jx[1:100,1:100] |> chopm
C[1:100,1:100]\(J2y*C)[1:100,1:100] - Jy[1:100,1:100] |> chopm

S₁ = KoornwinderTriangle(S.α+1,S.β,S.γ,domain(S))
J₁ = Lowering{1}(S₁)*Conversion(S,S₁)
S₂ = KoornwinderTriangle(S.α,S.β+1,S.γ,domain(S))
J₂ = Lowering{2}(S₂)*Conversion(S,S₂)


Fun(ff, S2)(0.1,0.2)
Fun(Conversion(S,S₂)*ff, S2)(0.1,0.2)
Fun(Lowering{2}(S₂)*Conversion(S,S₂)*ff, S2)(0.1,0.2)
using SO


Lowering{2}(S₂)


Fun(Fun((x,y) -> y, S2), S)
ff = (x,y) -> P(2,1,0,0,0,x,y)
    Fun(pad(ProductFun(ff, ProductTriangle(1,1,1)),20,20))(0.1,0.2), ff(0.1,0.2)

ff = (x,y) -> P(1,1,0,-0.5,-0.5,x,y)
    g = Fun(Fun(pad(ProductFun(ff, ProductTriangle(0.0,0.5,0.5)),20,20)),KoornwinderTriangle(0.,0.5,0.5))
    g(0.1,0.2),
        ff(0.1,0.2)

Fun(ff,DuffyTriangle())(0.1,0.2)
ProductFun(ff, ProductTriangle(0.0,0.5,0.5))
ff(0.1,0.2)


Fun(g,KoornwinderTriangle(0.0,-0.5,-0.5))
S1
C = Conversion(S1,S2)
C = Conversion(S1,KoornwinderTriangle(0.0,0.5,-0.5))
@which C[1,1]
@which C[1,1]
g.coefficients
x
r = randn(10)
x,y =0.1,0.2
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
    C = Conversion(KoornwinderTriangle(0.0,-0.5,-0.5),KoornwinderTriangle(0.0,0.5,-0.5))
    testbandedblockbandedoperator(C)
    r = randn(10)
    @test cfseval(0.0,-0.5,-0.5,r,x,y) ≈ cfseval(0.0,0.5,-0.5,[C[k,j] for k=1:10,j=1:10]*r,x,y)
    @test cfseval(0.0,-0.5,-0.5,r,x,y) ≈ cfseval(0.0,0.5,-0.5,C[1:10,1:10]*r,x,y)

    C = Conversion(KoornwinderTriangle(0.0,-0.5,-0.5),KoornwinderTriangle(0.0,-0.5,0.5))
    testbandedblockbandedoperator(C)
    r = randn(10)
    @test cfseval(0.0,-0.5,-0.5,r,x,y) ≈ cfseval(0.0,-0.5,0.5,[C[k,j] for k=1:10,j=1:10]*r,x,y)
    @test cfseval(0.0,-0.5,-0.5,r,x,y) ≈ cfseval(0.0,-0.5,0.5,C[1:10,1:10]*r,x,y)
end



k,j=1,2

K1=domainspace(C);K2=rangespace(C)
α,β,γ = K1.α,K1.β,K1.γ
K = Int(block(K2,k))
J = Int(block(K1,j))
κ=k-blockstart(K2,K)+1
ξ=j-blockstart(K1,J)+1
T = Float64
 J == K   && κ == ξ
 J == K   && κ+1 == ξ
 J == K+1 && κ == ξ
if  J == K == κ == ξ == 1
    one(T)
elseif   J == K   && κ == ξ
    T((K+κ+α+β+γ)/(2K+α+β+γ)*(κ+β+γ)/(2κ+β+γ-1))
elseif J == K   && κ+1 == ξ
    T(-(κ+γ)/(2κ+β+γ+1)*(K-κ)/(2K+α+β+γ))
elseif J == K+1 && κ == ξ
    T(-(K-κ+α+1)/(2K+α+β+γ+2)*(κ+β+γ)/(2κ+β+γ-1))
elseif J == K+1 && κ+1 == ξ
    T((κ+γ)/(2κ+β+γ+1)*(K+κ+β+γ+1)/(2K+α+β+γ+2))
else
    zero(T)
end
(2K+α+β+γ+2)
-(K-κ+α+1)/(2K+α+β+γ+2)*(κ+β+γ)/(2κ+β+γ-1)
(K-κ+α+1)
(κ+β+γ)
(2κ+β+γ-1)

Fun(Fun(pad(ProductFun(ff, ProductTriangle(0.0,0.5,0.5)),20,20)),

ProductFun(ff, ProductTriangle(0.0,-0.5,-0.5))
p = ProductFun(ff, ProductTriangle(1,1,1))

@which Fun(p)
g = Fun(pad(ProductFun(ff, ProductTriangle(1,1,1)),20,20))
g(0.1,0.2)
ff(0.1,0.2)
P(1,1,0,0,0,0.99999,0.0)

P = (n,k,a,b,c,x,y) -> x == 1.0 ? ((1-x))^k*jacobip(n-k,2k+b+c+1,a,1.0)*jacobip(k,c,b,-1.0) :
        ((1-x))^k*jacobip(n-k,2k+b+c+1,a,2x-1)*jacobip(k,c,b,2y/(1-x)-1)


P = (ℓ,m,a,b,c,x,y) -> x == 1.0 ? (2*(1-x))^m*njacobip(ℓ-m,2m+b+c+1,a,1.0)*njacobip(m,c,b,-1.0) :
        (2*(1-x))^m*njacobip(ℓ-m,2m+b+c+1,a,2x-1)*njacobip(m,c,b,2y/(1-x)-1)



Lowering{2}(S₂)*Conversion(S,S₂)*ff

0.1*ff(0.1,0.2)


jacobinorm(n,a,b) = if n ≠ 0
        sqrt((2n+a+b+1))*exp((lgamma(n+a+b+1)+lgamma(n+1)-log(2)*(a+b+1)-lgamma(n+a+1)-lgamma(n+b+1))/2)
    else
        sqrt(exp(lgamma(a+b+2)-log(2)*(a+b+1)-lgamma(a+1)-lgamma(b+1)))
    end
njacobip(n,a,b,x) = jacobinorm(n,a,b) * jacobip(n,a,b,x)

P = (ℓ,m,x,y) -> x == 1.0 ? (2*(1-x))^m*njacobip(ℓ-m,2m,0,1.0)*njacobip(m,-0.5,-0.5,-1.0) :
        (2*(1-x))^m*njacobip(ℓ-m,2m,0,2x-1)*njacobip(m,-0.5,-0.5,2y/(1-x)-1)

f̃ = function(x,y)
        ret = 0.0
        for j=1:size(F,2), k=1:size(F,1)-j+1
            ret += F̌[k,j] * P(k+j-2,j-1,x,y)
        end
        ret
    end



@testset "ProductTriangle constructors" begin
    S = ProductTriangle(1,1,1)
    @test fromcanonical(S, 0,0) == Vec(0,0)
    @test fromcanonical(S, 1,0) == fromcanonical(S, 1,1) == Vec(1,0)
    @test fromcanonical(S, 0,1) == Vec(0,1)
    pf = ProductFun((x,y)->exp(x*cos(y)), S, 40, 40)
    @test pf(0.1,0.2) ≈ exp(0.1*cos(0.2))

    d = Triangle(Vec(2,3),Vec(3,4),Vec(1,6))
    S = ProductTriangle(1,1,1,d)
    @test fromcanonical(S, 0,0) == d.a
    @test fromcanonical(S, 1,0) == fromcanonical(S, 1,1) == d.b
    @test fromcanonical(S, 0,1) == d.c
    pf = ProductFun((x,y)->exp(x*cos(y)), S, 50, 50)
    @test pf(2,4) ≈ exp(2*cos(4))

    pyf=ProductFun((x,y)->y*exp(x*cos(y)),ProductTriangle(1,0,1))
    @test pyf(0.1,0.2) ≈ 0.2exp(0.1*cos(0.2))
end


@testset "KoornwinderTriangle constructors" begin
    f = Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1))
    @test Fun(f,ProductTriangle(1,1,1))(0.1,0.2) ≈ exp(0.1*cos(0.2))
    f=Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(0,0,0))
    @test f(0.1,0.2) ≈ exp(0.1*cos(0.2))
    d = Triangle(Vec(0,0),Vec(3,4),Vec(1,6))
    f = Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1,d))
    @test f(2,4) ≈ exp(2*cos(4))


    K=KoornwinderTriangle(1,1,1)
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
    f = Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1))
    Jx = MultivariateOrthogonalPolynomials.Lowering{1}(space(f))
    testbandedblockbandedoperator(Jx)
    @test Fun(Jx*f,ProductTriangle(0,1,1))(0.1,0.2) ≈ 0.1exp(0.1*cos(0.2))

    Jy = MultivariateOrthogonalPolynomials.Lowering{2}(space(f))
    testbandedblockbandedoperator(Jy)

    @test Jy[3,1] ≈ 1/3

    @test ApproxFun.colstop(Jy,1) == 3
    @test ApproxFun.colstop(Jy,2) == 5
    @test ApproxFun.colstop(Jy,3) == 6
    @test ApproxFun.colstop(Jy,4) == 8


    @test Fun(Jy*f,ProductTriangle(1,0,1))(0.1,0.2) ≈ 0.2exp(0.1*cos(0.2))
    @test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),KoornwinderTriangle(1,0,1))).coefficients) < 1E-11

    Jz = MultivariateOrthogonalPolynomials.Lowering{3}(space(f))
    testbandedblockbandedoperator(Jz)
    @test Fun(Jz*f,ProductTriangle(1,1,0))(0.1,0.2) ≈ (1-0.1-0.2)exp(0.1*cos(0.2))

    @test f(0.1,0.2) ≈ exp(0.1*cos(0.2))

    Jx=Lowering{1}(KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(Jx)
    Jy=Lowering{2}(KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(Jy)
    Jz=Lowering{3}(KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(Jz)

    Jx=MultivariateOrthogonalPolynomials.Lowering{1}(space(f))→space(f)
    testbandedblockbandedoperator(Jx)
    @test norm((Jx*f-Fun((x,y)->x*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-10


    Jy=MultivariateOrthogonalPolynomials.Lowering{2}(space(f))→space(f)
    testbandedblockbandedoperator(Jy)
    @test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-10

    Jz=MultivariateOrthogonalPolynomials.Lowering{3}(space(f))→space(f)
    testbandedblockbandedoperator(Jz)
    @test norm((Jz*f-Fun((x,y)->(1-x-y)*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-10

    for K in (KoornwinderTriangle(2,1,1),KoornwinderTriangle(1,2,1))
        testbandedblockbandedoperator(Lowering{1}(K))
        Jx=(Lowering{1}(K)→K)
        testbandedblockbandedoperator(Jx)
    end

    d = Triangle(Vec(0,0),Vec(3,4),Vec(1,6))
    f = Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1,d))
    Jx,Jy = MultivariateOrthogonalPolynomials.jacobioperators(space(f))
    x,y = fromcanonical(d,0.1,0.2)
    @test (Jx*f)(x,y) ≈ x*f(x,y)
    @test (Jy*f)(x,y) ≈ y*f(x,y)
end

@testset "Triangle Conversion" begin
    C = Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(1,0,0))
    testbandedblockbandedoperator(C)

    C = Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(0,1,0))
    testbandedblockbandedoperator(C)

    C = Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(0,0,1))
    testbandedblockbandedoperator(C)
    C=Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(1,1,1))
    testbandedblockbandedoperator(C)

    Cx = I:KoornwinderTriangle(-1,0,0)→KoornwinderTriangle(0,0,0)
    testbandedblockbandedoperator(Cx)

    Cy = I:KoornwinderTriangle(0,-1,0)→KoornwinderTriangle(0,0,0)
    testbandedblockbandedoperator(Cy)

    Cz = I:KoornwinderTriangle(0,0,-1)→KoornwinderTriangle(0,0,0)
    testbandedblockbandedoperator(Cz)

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


    C=Conversion(KoornwinderTriangle(1,1,1),KoornwinderTriangle(2,1,1))
    testbandedblockbandedoperator(C)

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
end

@testset "Triangle Derivative/Laplacian" begin
    Dx = Derivative(KoornwinderTriangle(1,0,1), [1,0])
    testbandedblockbandedoperator(Dx)

    Dy = Derivative(KoornwinderTriangle(1,0,1), [0,1])
    testbandedblockbandedoperator(Dy)

    Δ = Laplacian(KoornwinderTriangle(1,1,1))
    testbandedblockbandedoperator(Δ)
end


@testset "Triangle derivatives" begin
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

    d = Triangle(Vec(0,0),Vec(3,4),Vec(1,6))
    K = KoornwinderTriangle(1,0,1,d)
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
end

@testset "TriangleWeight" begin
    S=TriangleWeight(1.,1.,1.,KoornwinderTriangle(1,1,1))
    C=Conversion(S,KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(C)
    f=Fun(S,rand(10))
    @test f(0.1,0.2) ≈ (C*f)(0.1,0.2)



    @test (Derivative(S,[1,0])*Fun(S,[1.]))(0.1,0.2) ≈ ((x,y)->y*(1-x-y)-x*y)(0.1,0.2)

    @test (Laplacian(S)*Fun(S,[1.]))(0.1,0.2) ≈ -2*0.1-2*0.2

    S = TriangleWeight(1,1,1,KoornwinderTriangle(1,1,1))


    Dx = Derivative(S,[1,0])
    Dy = Derivative(S,[0,1])

    for k=1:10
        v=[zeros(k);1.0]
        @test (Dy*Fun(S,v))(0.1,0.2) ≈ (Derivative([0,1])*Fun(Fun(S,v),KoornwinderTriangle(1,1,1)))(0.1,0.2)
        @test (Dx*Fun(S,v))(0.1,0.2) ≈ (Derivative([1,0])*Fun(Fun(S,v),KoornwinderTriangle(1,1,1)))(0.1,0.2)
    end

    Δ=Laplacian(S)

    f=Fun(S,rand(3))
    h=0.01
    QR=qrfact(I-h*Δ)
    @time u=\(QR,f;tolerance=1E-7)
    @time g=Fun(f,rangespace(QR))
    @time \(QR,g;tolerance=1E-7)

    # other domain
    S = TriangleWeight(1,1,1,KoornwinderTriangle(1,1,1))
    Dx = Derivative(S,[1,0])
    Dy = Derivative(S,[0,1])
    Div = (Dx → KoornwinderTriangle(0,0,0))  + (Dy → KoornwinderTriangle(0,0,0))
    f = Fun(S,rand(10))

    @test (Div*f)(0.1,0.2) ≈ (Dx*f)(0.1,0.2)+(Dy*f)(0.1,0.2)
    @test maxspace(rangespace(Dx),rangespace(Dy)) == rangespace(Dx + Dy) == KoornwinderTriangle(0,0,0)


    d = Triangle(Vec(2,3),Vec(3,4),Vec(1,6))
    S=TriangleWeight(1.,1.,1.,KoornwinderTriangle(1,1,1,d))
    @test domain(S) ≡ d
    @test maxspace(S,KoornwinderTriangle(0,0,0,d)) == KoornwinderTriangle(0,0,0,d)
    Dx = Derivative(S,[1,0])
    Dy = Derivative(S,[0,1])
    @test maxspace(rangespace(Dx),rangespace(Dy)) == rangespace(Dx + Dy) == KoornwinderTriangle(0,0,0,d)
    @test rangespace(Laplacian(S)) ≡ KoornwinderTriangle(1,1,1,d)
    f = Fun(S, randn(10))
    x,y = fromcanonical(S, 0.1,0.2)
    @test weight(S,x,y) ≈ 0.1*0.2*(1-0.1-0.2)
    g = Fun(f, KoornwinderTriangle(0,0,0,d))
    @test g(x,y) ≈ f(x,y)
    @test (Derivative([1,0])*g)(x,y) ≈ (Derivative([1,0])*f)(x,y)
    @test (Laplacian()*g)(x,y) ≈ (Laplacian()*f)(x,y)

    f=Fun(S,rand(3))
    h=0.01
    Δ = Laplacian(S)
    @test maxspace(rangespace(Δ), S) == KoornwinderTriangle(1,1,1,d)
    QR=qrfact(I-h*Δ)
    @time u=\(QR,f;tolerance=1E-7)
    @time g=Fun(f,rangespace(QR))
    @time \(QR,g;tolerance=1E-7)
end
