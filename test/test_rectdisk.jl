using MultivariateOrthogonalPolynomials, StaticArrays, BlockArrays, BlockBandedMatrices, ArrayLayouts, Base64,
        QuasiArrays, Test, ClassicalOrthogonalPolynomials, BandedMatrices, FastTransforms, LinearAlgebra
import MultivariateOrthogonalPolynomials: dunklxu_raising, dunklxu_lowering, AngularMomentum
using ForwardDiff

@testset "Dunkl-Xu disk" begin
    @testset "basics" begin
        P = DunklXuDisk()
        @test copy(P) ≡ P
        @test P ≠ DunklXuDisk(0.123)

        xy = axes(P,1)
        x,y = coordinates(P)
        @test xy[SVector(0.1,0.2)] == SVector(0.1,0.2)
        @test x[SVector(0.1,0.2)] == 0.1
        @test y[SVector(0.1,0.2)] == 0.2

        ρ = sqrt(1-0.1^2)
        @test P[SVector(0.1,0.2),1] ≈ 1
        @test P[SVector(0.1,0.2),Block(2)] ≈ [0.15,0.2]
        @test P[SVector(0.1,0.2),Block(3)] ≈ [jacobip(2,1/2,1/2,0.1),jacobip(1,3/2,3/2,0.1)*ρ*legendrep(1,0.2/ρ),ρ^2*legendrep(2,0.2/ρ)]
    end

    @testset "operators" begin
        N = 5
        β = 0.123

        P = DunklXuDisk(β)
        Q = DunklXuDisk(β+1)
        WP = WeightedDunklXuDisk(β)
        WQ = WeightedDunklXuDisk(β+1)

        @test WP ≠ WQ
        @test WP == WP

        x, y = coordinates(P)
        L = WP \ WQ
        R = Q \ P

        ∂x = Derivative(P, (1,0))
        ∂y = Derivative(P, (0,1))

        Dx = Q \ (∂x * P)
        Dy = Q \ (∂y * P)

        X = P \ (x .* P)
        Y = P \ (y .* P)


        @testset "lowering/raising" begin
            @test WP[SVector(0.1,0.2),Block.(1:6)]'L[Block.(1:6),Block.(1:4)] ≈ WQ[SVector(0.1,0.2),Block.(1:4)]'
            @test Q[SVector(0.1,0.2),Block.(1:4)]'R[Block.(1:4),Block.(1:4)] ≈ P[SVector(0.1,0.2),Block.(1:4)]'

            @test (DunklXuDisk() \ WeightedDunklXuDisk(1.0))[Block.(1:N), Block.(1:N)] ≈ (WeightedDunklXuDisk(0.0) \ WeightedDunklXuDisk(1.0))[Block.(1:N), Block.(1:N)]
        end


        @testset "jacobi" begin
            @test (L * R)[Block.(1:N), Block.(1:N)] ≈ (I - X^2 - Y^2)[Block.(1:N), Block.(1:N)]
            @test P[SVector(0.1,0.2),Block.(1:5)]'X[Block.(1:5),Block.(1:4)] ≈ 0.1P[SVector(0.1,0.2),Block.(1:4)]'
            @test P[SVector(0.1,0.2),Block.(1:5)]'Y[Block.(1:5),Block.(1:4)] ≈ 0.2P[SVector(0.1,0.2),Block.(1:4)]'
        end

        @testset "derivatives" begin
            @test Q[SVector(0.1,0.2),Block.(1:3)]'Dx[Block.(1:3),Block.(1:4)] ≈ [ForwardDiff.gradient(𝐱 -> DunklXuDisk{eltype(𝐱)}(P.β)[𝐱,k], SVector(0.1,0.2))[1] for k=1:10]'
            Mx = Q \ (x .* Q)
            My = Q \ (y .* Q)
    
            A = (Mx * Dy - My * Dx)[Block.(1:N), Block.(1:N)]
        end


        B = (Q \ P)[Block.(1:N), Block.(1:N)]

        C = B \ A

        @test C ≈ Tridiagonal(C)

        λ = eigvals(Matrix(C))

        @test λ ≈ im*imag(λ)

        ∂θ = AngularMomentum(P)
        A = P \ (∂θ * P)
        A2 = P \ (∂θ^2 * P)

        @test A[Block.(1:N), Block.(1:N)] ≈ C
        @test A2[Block.(1:N), Block.(1:N)] ≈ (A^2)[Block.(1:N), Block.(1:N)] ≈ A[Block.(1:N), Block.(1:N)]^2

        ∂x = Derivative(axes(WQ, 1), (1,0))
        ∂y = Derivative(axes(WQ, 1), (0,1))

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

    @testset "show" begin
        @test stringmime("text/plain", DunklXuDisk()) == "DunklXuDisk(0)"
        @test stringmime("text/plain", DunklXuDiskWeight(0)) == "(1-x^2-y^2)^0 on the unit disk"
    end
end
