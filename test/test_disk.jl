using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, BlockArrays, FastTransforms, LinearAlgebra, Test, ForwardDiff
import MultivariateOrthogonalPolynomials: DiskTrav, grid
import ClassicalOrthogonalPolynomials: HalfWeighted

function chebydiskeval(c::AbstractMatrix{T}, r, θ) where T
    ret = zero(T)
    for j = 1:2:size(c,2), k=1:size(c,1)
        m,ℓ = j ÷ 2, k-1
        ret += c[k,j] * cos(m*θ) * cos((2ℓ+isodd(m))*acos(r))
    end
    for j = 2:2:size(c,2), k=1:size(c,1)
        m = j ÷ 2; ℓ = k-1
        ret += c[k,j] * sin(m*θ) * cos((2ℓ+isodd(m))*acos(r))
    end
    ret
end

@testset "Disk" begin
    @testset "Evaluation" begin
        r,θ = 0.1, 0.2
        rθ = RadialCoordinate(r,θ)
        xy = SVector(rθ)
        @test Zernike()[rθ,1] ≈ Zernike()[xy,1] ≈ inv(sqrt(π))
        @test Zernike()[rθ,Block(1)] ≈ Zernike()[xy,Block(1)] ≈ [inv(sqrt(π))]
        @test Zernike()[rθ,Block(2)] ≈ [2r/sqrt(π)*sin(θ), 2r/sqrt(π)*cos(θ)]
        @test Zernike()[rθ,Block(3)] ≈ [sqrt(3/π)*(2r^2-1),sqrt(6/π)*r^2*sin(2θ),sqrt(6/π)*r^2*cos(2θ)]
    end

    @testset "DiskTrav" begin
        @test DiskTrav(reshape([1],1,1)) == [1]
        @test DiskTrav([1 2 3]) == 1:3
        @test DiskTrav([1 2 3 5 6;
                        4 0 0 0 0]) == 1:6
        @test DiskTrav([1 2 3 5 6 9 10;
                        4 7 8 0 0 0  0]) == 1:10

        @test DiskTrav([1 2 3 5 6 9 10; 4 7 8 0 0 0  0])[Block(3)] == 4:6

        @test_throws ArgumentError DiskTrav([1 2])
        @test_throws ArgumentError DiskTrav([1 2 3 4])
        @test_throws ArgumentError DiskTrav([1 2 3; 4 5 6])
        @test_throws ArgumentError DiskTrav([1 2 3 4; 5 6 7 8])
    end

    @testset "Transform" begin
        for (a,b) in ((0,0), (0.1, 0.2), (0,1))
            Zn = Zernike(a,b)[:,Block.(Base.OneTo(3))]
            for k = 1:6
                @test factorize(Zn) \ Zernike(a,b)[:,k] ≈ [zeros(k-1); 1; zeros(6-k)]
            end

            Z = Zernike(a,b);
            xy = axes(Z,1); x,y = first.(xy),last.(xy);
            u = Z * (Z \ exp.(x .* cos.(y)))
            @test u[SVector(0.1,0.2)] ≈ exp(0.1cos(0.2))
        end
    end

    @testset "Laplacian" begin
        # u = r^m*f(r^2) * cos(m*θ)
        # u_r = (m*r^(m-1)*f(r^2) + 2r^(m+1)*f'(r^2)) * cos(m*θ)
        # u_rr = (m*(m-1)*r^(m-2)*f(r^2) + (4m+2)*r^m*f'(r^2) + 2r^(m+1)*f''(r^2)) * cos(m*θ)
        # u_rr + u_r/r + u_θθ/r^2 = (4*(m+1)*f'(r^2) + 2r*f''(r^2)) * r^m * cos(m*θ)
        # t = r^2, dt = 2r * dr, 4*(m+1)*f'(t) + 2sqrt(t)*f''(t) = 4 t^(-m) * d/dt * t^(m+1) f'(t)
        # d/ds * (1-s) * P_n^(1,m)(s) = -n*P_n^(0,m+1)(s)
        # use L^6 and L^6'
        # 2t-1 = s, 2dt = ds

        using ClassicalOrthogonalPolynomials
        import ForwardDiff: hessian, derivative

        derivative2 = (f,x) -> hessian(t -> f(t[1]), SVector(x))[1]

        lap = (f,x,y) -> tr(hessian(f, SVector(x,y)))
        lapr = (fr,m,r) ->  derivative2(fr, r) + derivative(fr, r)/r - m^2 * fr(r)/r^2

        ℓ, m, a, b = 6, 2, 0, 1
        x,y = 0.1,0.2
        r = sqrt(x^2+y^2)
        θ = atan(y,x)
        u = xy -> zernikez(ℓ, m, a, b, xy)
        ur = r -> zerniker(ℓ, m, a, b, r)
        @test lap(u, x, y) ≈ lapr(ur,m,r) * (signbit(m) ? sin(-m*θ) : cos(m*θ))
        
        f = t -> sqrt(2^(m+a+b+2-iszero(m))/π) * Normalized(Jacobi{eltype(t)}(b, m+a))[2t-1,(ℓ-m) ÷ 2 + 1]
        ur = r -> r^m*f(r^2)
        @test ur(r) ≈ zerniker(ℓ, m, a, b, r)
        @test f(r^2) ≈ r^(-m) * zerniker(ℓ, m, a, b, r)
        u = xy -> ((x,y) = xy; ur(norm(xy)) * cos(m*atan(y,x)))
        t = r^2; 4*(m+1)*derivative(f,t) + 4t*derivative2(f,t)
        
        @test derivative(ur,r) ≈  m*r^(m-1)*f(r^2) + 2r^(m+1)*derivative(f,r^2)
        @test derivative2(ur,r) ≈ m*(m-1)*r^(m-2)*f(r^2) + (4m+2)*r^m * derivative(f,r^2) + 4r^(m+2)*derivative2(f,r^2)
        @test lapr(ur, m, r) ≈ 4*((m+1) * derivative(f,r^2) + r^2*derivative2(f,r^2)) * r^m
        @test lapr(ur, m, r) ≈ 4*((m+1) * derivative(f,t) + t*derivative2(f,t)) * t^(m/2)
        
        D = Derivative(axes(Chebyshev(),1))
        D1 = Normalized(Jacobi{eltype(t)}(b+1, m+a+1)) \ (D * Normalized(Jacobi{eltype(t)}(b, m+a)))


        D * HalfWeighted{:a}(Normalized(Jacobi(0.1,0.2)))

        ℓ, m, a, b = 6, 2, 0, 1

        f = t -> sqrt(2^(m+b+2-iszero(m))/π) * (1-t) * Normalized(Jacobi{eltype(t)}(b, m))[2t-1,(ℓ-m) ÷ 2 + 1]
        D1 = Normalized(Jacobi(b-1, m+1)) \ (D * (JacobiWeight(1,0) .* Normalized(Jacobi(b, m))))
        @test derivative(f,t) ≈ D1[(ℓ-m) ÷ 2+1,(ℓ-m) ÷ 2 + 1] * sqrt(2^(m+b+2-iszero(m))/π) * Normalized(Jacobi{eltype(t)}(b-1, m+1))[2t-1,(ℓ-m) ÷ 2 + 1]
        @test derivative(f,t) ≈ D1[(ℓ-m) ÷ 2+1,(ℓ-m) ÷ 2 + 1] * r^(-m-1) * zerniker(ℓ+1, m+1, 0, b-1, r)
        
        @test lapr(ur, m, r) ≈ 4D1[(ℓ-m) ÷ 2+1,(ℓ-m) ÷ 2 + 1] * t^(-m/2) * derivative(t -> (r = sqrt(t); t^(m+1) * r^(-m-1) * zerniker(ℓ+1, m+1, 0, b-1, r)), t)

        @test lapr(ur, m, r) ≈ 4D1[(ℓ-m) ÷ 2+1,(ℓ-m) ÷ 2 + 1] * t^(-m/2) * derivative(t -> (r = sqrt(t); t^(m+1) * sqrt(2^(m+b+2-iszero(m))/π) * Normalized(Jacobi{eltype(t)}(b-1, m+1))[2t-1,(ℓ-m) ÷ 2 + 1]), t)

        t^m * derivative(t -> (r = sqrt(t); t^(m+1) * sqrt(2^(m+b+2-iszero(m))/π) * Normalized(Jacobi{eltype(t)}(b-1, m+1))[2t-1,(ℓ-m) ÷ 2 + 1]), t)

        (JacobiWeight(1, m) .* Normalized(Jacobi{eltype(t)}(b, m))) \ ( D * (JacobiWeight(0, m+1) .* Normalized(Jacobi{eltype(t)}(0, m+1))))

        (JacobiWeight(1, m) .* Jacobi{eltype(t)}(b, m)) \ ( D * (JacobiWeight(0, m+1) .* Jacobi{eltype(t)}(0, m+1)))

        Jacobi{eltype(t)}(b, m) \ ( D * (JacobiWeight(0, m+1) .* Jacobi{eltype(t)}(0, m+1)))

    end
end
