using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, FillArrays, LazyBandedMatrices, BlockArrays, StaticArrays, Test
using BlockBandedMatrices: _BandedBlockBandedMatrix, _BlockBandedMatrix
using Base: oneto

@testset "Kernels" begin
    @testset "Basic" begin
        P = Legendre()
        T = Chebyshev()
        P² = RectPolynomial(Fill(P, 2))
        T² = RectPolynomial(Fill(T, 2))

        k = (x,y) -> exp(x*cos(y))
        K = kernel(expand(P², splat(k)))
        @test K[0.1,0.2] ≈ k(0.1,0.2)
        @test (K * expand(P, exp))[0.1] ≈ (K*expand(T, exp))[0.1] ≈ 2.5521805183347417

        K = kernel(expand(T², splat(k)))
        @test (K * expand(P, exp))[0.1] ≈ (K*expand(T, exp))[0.1] ≈ 2.5521805183347417


        x = axes(P,1)
        @test (k.(x,x') * expand(P, exp))[0.1] ≈ (k.(x,x') *expand(T, exp))[0.1] ≈ 2.5521805183347417

        y = Inclusion(2..3)
        @test (k.(y, x') * expand(P, exp))[2.1]  ≈ (k.(y, x') * expand(T, exp))[2.1] ≈ 13.80651898993351
    end

    @testset "Extension" begin
        Ex = _BandedBlockBandedMatrix(Ones{Int}((oneto(1),unitblocks(oneto(∞)))), blockedrange(oneto(∞)), (0,0), (0,0))
        # Ey = _BlockBandedMatrix(mortar(OneElement.(oneto(∞), oneto(∞))), oneto(∞), unitblocks(oneto(∞)), (0,0))

        T = ChebyshevT()
        T² = RectPolynomial(Fill(T,2))
        u = T² * (Ex * transform(T, exp))
        @test u[SVector(0.1,0.2)] ≈ exp(0.1)
    end
end