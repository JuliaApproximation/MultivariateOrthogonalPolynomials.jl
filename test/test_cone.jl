using ApproxFun, MultivariateOrthogonalPolynomials, Test
import MultivariateOrthogonalPolynomials: rectspace, totensor
import ApproxFunBase: plan_transform

@testset "Conic" begin
    @testset "DuffyConic" begin
        f = Fun((t,x,y) -> 1, DuffyConic(), 10)
        @test f.coefficients ≈ [1; zeros(ncoefficients(f)-1)]
        @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ 1

        f = Fun((t,x,y) -> t, DuffyConic(), 10)
        @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ sqrt(0.1^2+0.2^2)

        f = Fun((t,x,y) -> x, DuffyConic(), 10)

        @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ 0.1

        f = Fun((t,x,y) -> exp(cos(t*x)*y), DuffyConic(), 1000)
        @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ exp(cos(sqrt(0.1^2+0.2^2)*0.1)*0.2)

        f = Fun((t,x,y) -> exp(cos(t*x)*y), DuffyConic())
        @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ exp(cos(sqrt(0.1^2+0.2^2)*0.1)*0.2)

        m,ℓ = (1,1)
        f = (txy) -> ((t,x,y) = txy;  θ = atan(y,x); Fun(NormalizedJacobi(0,2m+1,Segment(1,0)),[zeros(ℓ);1])(t) * 2^m * t^m * cos(m*θ))
        g = Fun(f, DuffyConic())
        t,x,y = sqrt(0.1^2+0.2^2),0.1,0.2
        @test g(t,x,y) ≈ f((t,x,y))
    end

    @testset "LegendreConicPlan" begin
        m,ℓ = (1,1)
        f = (txy) -> ((t,x,y) = txy;  θ = atan(y,x); Fun(NormalizedJacobi(0,2m+1,Segment(1,0)),[zeros(ℓ);1])(t) * 2^m * t^m * cos(m*θ))
        p = points(LegendreConic(), 10)
        P = plan_transform(LegendreConic(), f.(p))
        @test P.duffyplan*f.(p) ≈ Fun(f, DuffyConic()).coefficients[1:12]
        g = Fun(f, LegendreConic(), 20)
        t,x,y = sqrt(0.1^2+0.2^2),0.1,0.2
        @test g(t,x,y) ≈ f((t,x,y))
    end

    @testset "Legendre<>DuffyConic" begin
        a = randn(10)
        F = totensor(rectspace(DuffyConic()), a)


        b = coefficients(a, LegendreConic(), DuffyConic())

        coefficients(b, DuffyConic(), LegendreConic())
    end

    @testset "LegendreConic" begin
        f = Fun((t,x,y) -> 1, LegendreConic(), 10)
        @test f.coefficients ≈ [1; zeros(ncoefficients(f)-1)]
        @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ 1

        f = Fun((t,x,y) -> t, LegendreConic(), 10)
        @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ sqrt(0.1^2+0.2^2)
        f = Fun((t,x,y) -> 1+t+x+y, LegendreConic(), 10)
        @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ 1+sqrt(0.1^2+0.2^2)+0.1+0.2


        @time Fun((t,x,y) -> 1+t+x+y, LegendreConic(), 1000)
    end
end