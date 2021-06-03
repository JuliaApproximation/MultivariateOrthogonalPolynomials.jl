using MultivariateOrthogonalPolynomials, StaticArrays, BlockArrays, BlockBandedMatrices, ArrayLayouts,
        QuasiArrays, Test, ClassicalOrthogonalPolynomials, BandedMatrices, FastTransforms, LinearAlgebra
import MultivariateOrthogonalPolynomials: dunklxu_raising, dunklxu_lowering

@testset "Dunkl-Xu disk" begin
    @testset "basics" begin
        P = DunklXuDisk()
        @test copy(P) ≡ P

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

        @test (L*R)[Block.(1:N), Block.(1:N)] ≈ (I - X^2 - Y^2)[Block.(1:N), Block.(1:N)]

        ∂x = PartialDerivative{1}(P)
        ∂y = PartialDerivative{2}(P)

        Dx = Q \ (∂x * P)
        Dy = Q \ (∂y * P)

        Mx = Q \ (x .* Q)
        My = Q \ (y .* Q)

        A = Mx[Block.(1:N), Block.(1:N+1)]*Dy[Block.(1:N+1), Block.(1:N)] - My[Block.(1:N), Block.(1:N+1)]*Dx[Block.(1:N+1), Block.(1:N)]

        B = (Q \ P)[Block.(1:N), Block.(1:N)]

        C = B \ A

        @test C ≈ Tridiagonal(C)

        λ = eigvals(Matrix(C))

        @test λ ≈ im*imag(λ)

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
