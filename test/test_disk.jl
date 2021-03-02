using MultivariateOrthogonalPolynomials, StaticArrays, BlockArrays, Test
import MultivariateOrthogonalPolynomials: DiskTrav

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
        @test Zernike()[rθ,Block(2)] ≈ [2r/π*sin(θ), 2r/π*cos(θ)]
        @test Zernike()[rθ,Block(3)] ≈ [sqrt(3/π)*(2r^2-1),sqrt(6)/π*r^2*sin(2θ),sqrt(6)/π*r^2*cos(2θ)]
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
end