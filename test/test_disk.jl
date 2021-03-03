using MultivariateOrthogonalPolynomials, StaticArrays, BlockArrays, FastTransforms, LinearAlgebra, Test
import MultivariateOrthogonalPolynomials: DiskTrav, grid

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
        Zn = Zernike()[:,Block.(Base.OneTo(3))]
        for k = 1:6
            @test factorize(Zn) \ Zernike()[:,k] ≈ [zeros(k-1); 1; zeros(6-k)]
        end

        Z = Zernike();
        xy = axes(Z,1); x,y = first.(xy),last.(xy);
        u = Z * (Z \ exp.(x .* cos.(y)))
        @test u[SVector(0.1,0.2)] ≈ exp(0.1cos(0.2))
    end

    @testset "Laplacian" begin
        # u = r^m*f(r^2) * cos(m*θ)
        # u_r = (m*r^(m-1)*f(r^2) + 2r^(m+1)*f'(r^2)) * cos(m*θ)
        # u_rr = (m*(m-1)*r^(m-2)*f(r^2) + 2*(2m+1)*r^m*f'(r^2) + 4r^(m+2)*f''(r^2)) * cos(m*θ)
        # u_rr + u_r/r + u_θθ/r^2 = (4*(m+1)*f'(r^2) + 4r^2*f''(r^2)) * r^m * cos(m*θ)
        # t = r^2, dt = 2r * dr, 4*(m+1)*f'(t) + 4t*f''(t) = 4 t^(-m) * d/dt * t^(m+1) f'(t)
        # d/ds * (1-s) * P_n^(1,m)(s) = -n*P_n^(0,m+1)(s)
        # use L^6 and L^6'
        # 2t-1 = s, 2dt = ds

    end
end
