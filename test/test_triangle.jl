using MultivariateOrthogonalPolynomials, StaticArrays, BlockArrays, BlockBandedMatrices, ArrayLayouts, QuasiArrays, Test

@testset "Triangle" begin
    P = JacobiTriangle()
    @test copy(P) ≡ P

    xy = axes(P,1)
    x,y = first.(xy),last.(xy)
    @test xy[SVector(0.1,0.2)] == SVector(0.1,0.2)
    @test x[SVector(0.1,0.2)] == 0.1
    @test y[SVector(0.1,0.2)] == 0.2
    
    @testset "operators" begin
        ∂ˣ = PartialDerivative{1}(xy)
        ∂ʸ = PartialDerivative{2}(xy)

        Dˣ = JacobiTriangle(1,0,1) \ (∂ˣ * P)
        Dʸ = JacobiTriangle(0,1,1) \ (∂ʸ * P)

        M = P'P

        Rx = JacobiTriangle(1,0,0) \ P
        Lx = P \ WeightedTriangle(1,0,0)
        Ry = JacobiTriangle(0,1,0) \ P
        Ly = P \ WeightedTriangle(0,1,0)
        
        ∂ˣ² = (∂ˣ)^2
        ∂ʸ² = (∂ʸ)^2
        @test ∂ˣ² isa ApplyQuasiMatrix{<:Any,typeof(^)}
        @test ∂ʸ² isa ApplyQuasiMatrix{<:Any,typeof(^)}

        Dˣ² = JacobiTriangle(2,0,2) \ (∂ˣ² * P)
        Dˣ² = JacobiTriangle(2,0,2) \ (∂ˣ * (∂ˣ * P))
        Dʸ² = JacobiTriangle(0,2,2) \ (∂ʸ * (∂ʸ * P))

        @testset "jacobi" begin
            x .* P

            X = Lx * Rx
            Y = Ly * Ry

            @test blockbandwidths(M) == subblockbandwidths(M) == (0,0)
            @test blockbandwidths(X) == blockbandwidths(Y) == (1,1)
            @test subblockbandwidths(X) == (0,0)
            @test subblockbandwidths(Y) == (1,1)
            

            @testset "truncations" begin
                N = 100;
                KR,JR = Block.(1:N),Block.(1:N)

                @test Dˣ[KR,JR] isa BandedBlockBandedMatrix
                @test Lx[KR,JR] isa BandedBlockBandedMatrix
                @test Rx[KR,JR] isa BandedBlockBandedMatrix
                @test Ly[KR,JR] isa BandedBlockBandedMatrix
                @test Ry[KR,JR] isa BandedBlockBandedMatrix
                @test X[KR,JR] isa BandedBlockBandedMatrix
                @test Y[KR,JR] isa BandedBlockBandedMatrix
            end
        end
    end
end
