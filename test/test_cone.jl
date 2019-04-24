using ApproxFun, MultivariateOrthogonalPolynomials, Test
import MultivariateOrthogonalPolynomials: rectspace, totensor

@testset "DuffyCone" begin
    f = Fun((t,x,y) -> 1, DuffyCone(), 10)
    @test f.coefficients ≈ [1; zeros(ncoefficients(f)-1)]
    @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ 1

    f = Fun((t,x,y) -> t, DuffyCone(), 10)
    @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ sqrt(0.1^2+0.2^2)

    f = Fun((t,x,y) -> x, DuffyCone(), 10)

    @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ 0.1

    f = Fun((t,x,y) -> exp(cos(t*x)*y), DuffyCone(), 1000)
    @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ exp(cos(sqrt(0.1^2+0.2^2)*0.1)*0.2)

    f = Fun((t,x,y) -> exp(cos(t*x)*y), DuffyCone())
    @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ exp(cos(sqrt(0.1^2+0.2^2)*0.1)*0.2)

    m,ℓ = (1,1)
    f = (txy) -> ((t,x,y) = txy;  θ = atan(y,x); Fun(NormalizedJacobi(0,2m+1,Segment(1,0)),[zeros(ℓ);1])(t) * 2^m * t^m * cos(m*θ))
    g = Fun(f, DuffyCone())
    t,x,y = sqrt(0.1^2+0.2^2),0.1,0.2
    @test g(t,x,y) ≈ f((t,x,y))
end

@testset "LegendreConePlan" begin
    m,ℓ = (1,1)
    f = (txy) -> ((t,x,y) = txy;  θ = atan(y,x); Fun(NormalizedJacobi(0,2m+1,Segment(1,0)),[zeros(ℓ);1])(t) * 2^m * t^m * cos(m*θ))
    p = points(LegendreCone(), 10)
    P = plan_transform(LegendreCone(), f.(p))
    @test P.duffyplan*f.(p) ≈ Fun(f, DuffyCone()).coefficients[1:12]
    coefficients(g, LegendreCone())
    g = Fun(f, LegendreCone(), 20)
    t,x,y = sqrt(0.1^2+0.2^2),0.1,0.2
    @test g(t,x,y) ≈ f((t,x,y))
end

@testset "Legendre<>DuffyCone" begin
    a = randn(10)
    F = totensor(rectspace(DuffyCone()), a)


    b = coefficients(a, LegendreCone(), DuffyCone())

    coefficients(b, DuffyCone(), LegendreCone())
end

@testset "LegendreCone" begin
    f = Fun((t,x,y) -> 1, LegendreCone(), 10)
    @test f.coefficients ≈ [1; zeros(ncoefficients(f)-1)]
    @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ 1

    f = Fun((t,x,y) -> t, LegendreCone(), 10)
    @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ sqrt(0.1^2+0.2^2)
    f = Fun((t,x,y) -> 1+t+x+y, LegendreCone(), 10)
    @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ 1+sqrt(0.1^2+0.2^2)+0.1+0.2


    @time Fun((t,x,y) -> 1+t+x+y, LegendreCone(), 1000)
end