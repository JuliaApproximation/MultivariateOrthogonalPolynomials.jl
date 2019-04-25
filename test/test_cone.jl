using ApproxFun, MultivariateOrthogonalPolynomials, Test
import MultivariateOrthogonalPolynomials: rectspace, totensor, duffy2legendrecone!, legendre2duffycone!, c_plan_rottriangle, plan_transform

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

@testset "Legendre<>DuffyCone" begin
    for k = 0:10
        a = [zeros(k); 1.0; zeros(5)]
        F = totensor(rectspace(DuffyCone()), a)
        F = pad(F, :, 2size(F,1)-1)
        T = eltype(a)
        P = c_plan_rottriangle(size(F,1), zero(T), zero(T), zero(T))
        @test legendre2duffycone!(P, duffy2legendrecone!(P, copy(F))) ≈ F

        b = coefficients(a, LegendreCone(), DuffyCone())
        @test a ≈ coefficients(b, DuffyCone(), LegendreCone())[1:length(a)]
    end
end


@testset "LegendreConePlan" begin
    for (m,ℓ) in ((0,0), (0,1), (0,2), (1,1), (1,2), (2,2))
        f = (txy) -> ((t,x,y) = txy;  θ = atan(y,x); Fun(NormalizedJacobi(0,2m+1,Segment(1,0)),[zeros(ℓ);1])(t) * 2^m * t^m * cos(m*θ))
        g = Fun(f, LegendreCone())
        t,x,y = sqrt(0.1^2+0.2^2),0.1,0.2
        @test g(t,x,y) ≈ f((t,x,y))
    end
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