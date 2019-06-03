using ApproxFun, MultivariateOrthogonalPolynomials, StaticArrays, FillArrays, Test
import ApproxFunBase: checkpoints
import MultivariateOrthogonalPolynomials: rectspace, totensor, duffy2legendreconic!, legendre2duffyconic!, c_plan_rottriangle, plan_transform,
                        c_execute_tri_hi2lo, c_execute_tri_lo2hi, duffy2legendrecone_column_J!, duffy2legendrecone!, legendre2duffycone!

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

        f = Fun((t,x,y) -> t, DuffyCone(), 10)
        @test f(0.3,0.1,0.2) ≈ 0.3

        f = Fun((t,x,y) -> x, DuffyCone(), 10)
        @test f(0.3,0.1,0.2) ≈ 0.1

        f = Fun((t,x,y) -> y, DuffyCone(), 10)
        @test f(0.3,0.1,0.2) ≈ 0.2

        f = Fun((t,x,y) -> exp(cos(t*x)*y), DuffyCone(), 2000)
        @test f(0.3,0.1,0.2) ≈ exp(cos(0.3*0.1)*0.2)

        f = Fun((t,x,y) -> exp(cos(t*x)*y), DuffyCone())
        @test f(0.3,0.1,0.2) ≈ exp(cos(0.3*0.1)*0.2)
    end

    @testset "TriTransform" begin
        m = 1
        p = Fun(NormalizedJacobi(0,2m+2,0..1), [1])
        q = Fun(x -> p(x)*(1-x)^m * 2^m, NormalizedJacobi(0,2,0..1))
        F = [zeros(2)  q.coefficients]
        F = pad(F, :, 2size(F,1)-1)
        T = Float64
        P = c_plan_rottriangle(size(F,1), zero(T), zero(T), one(T))
        c_execute_tri_lo2hi(P, F)
        @test F ≈ (B = zeros(size(F)); B[1,m+1] = 1; B)
    end

    @testset "Legendre<>DuffyCone" begin
        for (m,k,ℓ) in ((0,0,0), (0,0,1), (0,0,2), (1,0,0), (1,0,1), (1,1,0), (1,1,1), (1,1,2))
            Y = Fun(ZernikeDisk(), [Zeros(sum(1:m)+k); 1])
            f = (txy) -> ((t,x,y) = txy;  θ = atan(y,x); Fun(NormalizedJacobi(0,2m+2,Segment(1,0)),[zeros(ℓ);1])(t) * 2^m * t^m * Y(x/t,y/t))
            g = Fun(f, DuffyCone())
            a = g.coefficients
            F = totensor(rectspace(DuffyCone()), a)
            for j = 1:size(F,2)
                F[:,j] = coefficients(F[:,j], NormalizedJacobi(0,1,Segment(1,0)), NormalizedJacobi(0,2,Segment(1,0)))
            end

            T = eltype(a)
            P = c_plan_rottriangle(size(F,1), zero(T), zero(T), one(T))

            Fc = Matrix{Float64}(undef,size(F,1),size(F,1))
            for J = 1:size(F,2)
                duffy2legendrecone_column_J!(P, Fc, F, J)
            end

            @test F ≈ (B = zeros(size(F)); B[ℓ+1,sum(1:m)+k+1] = 1; B)
        end
        for k = 0:10
            a = [zeros(k); 1.0; zeros(5)]
            F = totensor(rectspace(DuffyCone()), a)
            F = pad(F, :, 2size(F,1)-1)
            
            T = eltype(a)
            P = c_plan_rottriangle(size(F,1), zero(T), zero(T), one(T))
            
            @test legendre2duffycone!(P, duffy2legendrecone!(P, copy(F))) ≈ F

            b = coefficients(a, LegendreCone(), DuffyCone())
            @test a ≈ coefficients(b, DuffyCone(), LegendreCone())[1:length(a)]
        end
    end
    

    @testset "LegendreConePlan" begin
        for (m,k,ℓ) in ((0,0,0), )
            Y = Fun(ZernikeDisk(), [Zeros(sum(1:m)+k); 1])
            f = (txy) -> ((t,x,y) = txy;  θ = atan(y,x); Fun(NormalizedJacobi(0,2m+2,Segment(1,0)),[zeros(ℓ);1])(t) * 2^m * t^m * Y(x,y))
            g = Fun(f, LegendreCone(),20)
            t,x,y = 0.3,0.1,0.2
            @test g(t,x,y) ≈ f((t,x,y))
        end
    end

    @testset "LegendreCone" begin
        f = Fun((t,x,y) -> 1, LegendreCone(), 10)
        @test f.coefficients ≈ [1; zeros(ncoefficients(f)-1)]
        @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ 1

        f = Fun((t,x,y) -> t, LegendreConic(), 10)
        @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ sqrt(0.1^2+0.2^2)
        f = Fun((t,x,y) -> 1+t+x+y, LegendreConic(), 10)
        @test f(sqrt(0.1^2+0.2^2),0.1,0.2) ≈ 1+sqrt(0.1^2+0.2^2)+0.1+0.2


        @time Fun((t,x,y) -> 1+t+x+y, LegendreConic(), 1000)
    end
end