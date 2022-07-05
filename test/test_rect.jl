using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, Test

@testset "RectPolynomial" begin
    @testset "Evaluation" begin
        T = ChebyshevT()
        T² = RectPolynomial(T, T)
        xy = SVector(0.1,0.2)
        @test T²[xy, Block(1)[1]] == T²[xy, 1]
        @test T²[xy, Block(1)] == T²[xy, Block.(1:1)]
        @test T²[xy, Block(2)] == [0.1,0.2]
        @test T²[xy, Block(3)] ≈ [cos(2*acos(0.1)), 0.1*0.2, cos(2*acos(0.2))]

        U = ChebyshevU()
        V = RectPolynomial(T, U)
        @test V[xy, Block(1)[1]] == V[xy, 1]
        @test V[xy, Block(1)] == V[xy, Block.(1:1)]
        @test V[xy, Block(2)] == [0.1,2*0.2]
        @test V[xy, Block(3)] ≈ [cos(2*acos(0.1)), 2*0.1*0.2, sin(3*acos(0.2))/sin(acos(0.2))]
    end

    @testset "Derivatives" begin
        T = ChebyshevT()
        T² = RectPolynomial(T, T)
        xy = axes(T,1)
        D_x,D_y = PartialDerivative{1}(xy),PartialDerivative{2}(xy)
        D_x*T²
        D_y*T²
end