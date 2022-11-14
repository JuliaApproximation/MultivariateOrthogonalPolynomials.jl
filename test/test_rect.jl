using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, LinearAlgebra, BlockArrays, FillArrays, Test
import ClassicalOrthogonalPolynomials: expand

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
        V = KronPolynomial(T, U)
        @test V[xy, Block(1)[1]] == V[xy, 1]
        @test V[xy, Block(1)] == V[xy, Block.(1:1)]
        @test V[xy, Block(2)] == [0.1,2*0.2]
        @test V[xy, Block(3)] ≈ [cos(2*acos(0.1)), 2*0.1*0.2, sin(3*acos(0.2))/sin(acos(0.2))]
    end

    @testset "Transform" begin
        T = ChebyshevT()
        T² = RectPolynomial(Fill(T, 2))
        T²ₙ = T²[:,Block.(Base.OneTo(5))]
        𝐱 = axes(T²ₙ,1)
        x,y = first.(𝐱),last.(𝐱)
        @test T²ₙ \ one.(x) == [1; zeros(14)]
        T² \ x
        f = expand(T², 𝐱 -> ((x,y) = 𝐱; exp(x*cos(y-0.1))))
        @test f[SVector(0.1,0.2)] ≈ exp(0.1*cos(0.1))
        
        U = ChebyshevU()
        U² = RectPolynomial(Fill(U, 2))

        a,b = f.args
        f[SVector(0.1,0.2)]

        a,b = T² , (T² \ broadcast(xy -> ((x,y) = xy; exp(x*cos(y))), xy))
    end

    @testset "Conversion" begin
        T = ChebyshevT()
        U = ChebyshevU()
        T² = RectPolynomial(T, T)
        U² = RectPolynomial(U, U)
        U²\T²
    end

    @testset "Derivatives" begin
        T = ChebyshevT()
        U = ChebyshevU()
        C = Ultraspherical(2)
        T² = RectPolynomial(T, T)
        U² = RectPolynomial(U, U)
        C² = RectPolynomial(C, C)
        xy = axes(T²,1)
        D_x,D_y = PartialDerivative{1}(xy),PartialDerivative{2}(xy)
        D_x*T²
        D_y*T²
        U²\D_x*T²
        U²\D_y*T²

        U²\(D_x + D_y)*T²
        A = C²\D_x^2*T²
        B = C²\D_y^2*T²
        C²\(D_x^2 + D_y^2)*T²
    end

    @testset "PDEs" begin
        Q = Jacobi(1,1)
        W = Weighted(Q)
        P = Legendre()
        W² = RectPolynomial(W, W)
        P² = RectPolynomial(P, P)
        Q² = RectPolynomial(Q, Q)
        xy = axes(W²,1)
        D_x,D_y = PartialDerivative{1}(xy),PartialDerivative{2}(xy)
        Δ = Q²\(D_x^2 + D_y^2)*W²

        K = Block.(1:200); @time L = Δ[K,K]; @time qr(L);
        \(qr(Δ), [1; zeros(∞)]; tolerance=1E-1)
    end
end