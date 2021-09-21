using MultivariateOrthogonalPolynomials, StaticArrays, BlockArrays, BlockBandedMatrices, ArrayLayouts,
        QuasiArrays, Test, ClassicalOrthogonalPolynomials, BandedMatrices, FastTransforms, LinearAlgebra
import MultivariateOrthogonalPolynomials: dunklxu_raising, dunklxu_lowering, AngularMomentum

@testset "Dunkl-Xu disk" begin
    @testset "basics" begin
        P = DunklXuDisk()
        @test copy(P) ≡ P
        @test P ≠ DunklXuDisk(0.123)

        xy = axes(P,1)
        x,y = first.(xy),last.(xy)
        @test xy[SVector(0.1,0.2)] == SVector(0.1,0.2)
        @test x[SVector(0.1,0.2)] == 0.1
        @test y[SVector(0.1,0.2)] == 0.2
    end

    @testset "operators" begin
        N = 5
        β = 0.123

        P = DunklXuDisk(β)
        Q = DunklXuDisk(β+1)
        WP = WeightedDunklXuDisk(β)
        WQ = WeightedDunklXuDisk(β+1)

        x, y = first.(axes(P, 1)), last.(axes(P, 1))

        L = WP \ WQ
        R = Q \ P

        X = P \ (x .* P)
        Y = P \ (y .* P)

        @test (L * R)[Block.(1:N), Block.(1:N)] ≈ (I - X^2 - Y^2)[Block.(1:N), Block.(1:N)]

        @test (DunklXuDisk() \ WeightedDunklXuDisk(1.0))[Block.(1:N), Block.(1:N)] ≈ (WeightedDunklXuDisk(0.0) \ WeightedDunklXuDisk(1.0))[Block.(1:N), Block.(1:N)]

        ∂x = PartialDerivative{1}(axes(P, 1))
        ∂y = PartialDerivative{2}(axes(P, 1))

        Dx = Q \ (∂x * P)
        Dy = Q \ (∂y * P)

        Mx = Q \ (x .* Q)
        My = Q \ (y .* Q)

        A = (Mx * Dy - My * Dx)[Block.(1:N), Block.(1:N)]

        B = (Q \ P)[Block.(1:N), Block.(1:N)]

        C = B \ A

        @test C ≈ Tridiagonal(C)

        λ = eigvals(Matrix(C))

        @test λ ≈ im*imag(λ)

        ∂θ = AngularMomentum(axes(P, 1))
        A = P \ (∂θ * P)
        A2 = P \ (∂θ^2 * P)

        @test A[Block.(1:N), Block.(1:N)] ≈ C
        @test A2[Block.(1:N), Block.(1:N)] ≈ (A^2)[Block.(1:N), Block.(1:N)] ≈ A[Block.(1:N), Block.(1:N)]^2

        ∂x = PartialDerivative{1}(axes(WQ, 1))
        ∂y = PartialDerivative{2}(axes(WQ, 1))

        wDx = WP \ (∂x * WQ)
        wDy = WP \ (∂y * WQ)

        @testset "truncations" begin
            KR,JR = Block.(1:N),Block.(1:N)

            @test L[KR,JR] isa BandedBlockBandedMatrix
            @test R[KR,JR] isa BandedBlockBandedMatrix
            @test X[KR,JR] isa BandedBlockBandedMatrix
            @test Y[KR,JR] isa BandedBlockBandedMatrix
            @test Dx[KR,JR] isa BandedBlockBandedMatrix
            @test Dy[KR,JR] isa BandedBlockBandedMatrix
        end
    end
end
