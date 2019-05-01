using ApproxFun, MultivariateOrthogonalPolynomials, StaticArrays, FillArrays, Test
import ApproxFunBase: checkpoints
import MultivariateOrthogonalPolynomials: rectspace, totensor, duffy2legendreconic!, legendre2duffyconic!, c_plan_rottriangle, plan_transform

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

    @testset "Legendre<>DuffyConic" begin
        for k = 0:10
            a = [zeros(k); 1.0; zeros(5)]
            F = totensor(rectspace(DuffyConic()), a)
            F = pad(F, :, 2size(F,1)-1)
            T = eltype(a)
            P = c_plan_rottriangle(size(F,1), zero(T), zero(T), zero(T))
            @test legendre2duffyconic!(P, duffy2legendreconic!(P, copy(F))) ≈ F

            b = coefficients(a, LegendreConic(), DuffyConic())
            @test a ≈ coefficients(b, DuffyConic(), LegendreConic())[1:length(a)]
        end
    end

    @testset "LegendreConicPlan" begin
        for (m,ℓ) in ((0,0), (0,1), (0,2), (1,1), (1,2), (2,2))
            f = (txy) -> ((t,x,y) = txy;  θ = atan(y,x); Fun(NormalizedJacobi(0,2m+1,Segment(1,0)),[zeros(ℓ);1])(t) * 2^m * t^m * cos(m*θ))
            g = Fun(f, LegendreConic())
            t,x,y = sqrt(0.1^2+0.2^2),0.1,0.2
            @test g(t,x,y) ≈ f((t,x,y))
        end
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


@testset "Cone" begin
    @testset "rectspace" begin
        rs = rectspace(DuffyCone())
        @test points(rs,10) isa Vector{SVector{3,Float64}}
        @test_broken @inferred(checkpoints(rs))
        @test checkpoints(rs) isa Vector{SVector{3,Float64}}
    end

    @testset "DuffyCone" begin
        p = points(DuffyCone(), 10)
        @test p isa Vector{SVector{3,Float64}}
        P = plan_transform(DuffyCone(), Vector{Float64}(undef, length(p)))
        
        @test P * fill(1.0, length(p)) ≈ [1.2533141373154997; Zeros(164)] ≈ [Fun((x,y) -> 1, ZernikeDisk()).coefficients; Zeros(164)]

        f = Fun((t,x,y) -> 1, DuffyCone(), 10)
        @test f.coefficients ≈ [1.2533141373154997; Zeros(164)]
        @test f(0.3,0.1,0.2) ≈ 1

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

    @testset "Legendre<>DuffyConic" begin
        for k = 0:10
            a = [zeros(k); 1.0; zeros(5)]
            F = totensor(rectspace(DuffyConic()), a)
            F = pad(F, :, 2size(F,1)-1)
            T = eltype(a)
            P = c_plan_rottriangle(size(F,1), zero(T), zero(T), zero(T))
            @test legendre2duffyconic!(P, duffy2legendreconic!(P, copy(F))) ≈ F

            b = coefficients(a, LegendreConic(), DuffyConic())
            @test a ≈ coefficients(b, DuffyConic(), LegendreConic())[1:length(a)]
        end
    end

    @testset "LegendreConicPlan" begin
        for (m,ℓ) in ((0,0), (0,1), (0,2), (1,1), (1,2), (2,2))
            f = (txy) -> ((t,x,y) = txy;  θ = atan(y,x); Fun(NormalizedJacobi(0,2m+1,Segment(1,0)),[zeros(ℓ);1])(t) * 2^m * t^m * cos(m*θ))
            g = Fun(f, LegendreConic())
            t,x,y = sqrt(0.1^2+0.2^2),0.1,0.2
            @test g(t,x,y) ≈ f((t,x,y))
        end
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