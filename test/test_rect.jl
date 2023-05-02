using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, LinearAlgebra, BlockArrays, FillArrays, Test
import ClassicalOrthogonalPolynomials: expand

@testset "RectPolynomial" begin
    @testset "Evaluation" begin
        T = ChebyshevT()
        T² = RectPolynomial(T, T)
        𝐱 = SVector(0.1,0.2)
        @test T²[𝐱, Block(1)[1]] == T²[𝐱, 1]
        @test T²[𝐱, Block(1)] == T²[𝐱, Block.(1:1)]
        @test T²[𝐱, Block(2)] == [0.1,0.2]
        @test T²[𝐱, Block(3)] ≈ [cos(2*acos(0.1)), 0.1*0.2, cos(2*acos(0.2))]

        U = ChebyshevU()
        V = KronPolynomial(T, U)
        @test V[𝐱, Block(1)[1]] == V[𝐱, 1]
        @test V[𝐱, Block(1)] == V[𝐱, Block.(1:1)]
        @test V[𝐱, Block(2)] == [0.1,2*0.2]
        @test V[𝐱, Block(3)] ≈ [cos(2*acos(0.1)), 2*0.1*0.2, sin(3*acos(0.2))/sin(acos(0.2))]
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

        a,b = T² , (T² \ broadcast(𝐱 -> ((x,y) = 𝐱; exp(x*cos(y))), 𝐱))
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
        𝐱 = axes(T²,1)
        D_x,D_y = PartialDerivative{1}(𝐱),PartialDerivative{2}(𝐱)
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
        𝐱 = axes(W²,1)
        D_x,D_y = PartialDerivative{1}(𝐱),PartialDerivative{2}(𝐱)
        Δ = Q²\(D_x^2 + D_y^2)*W²

        K = Block.(1:200); @time L = Δ[K,K]; @time qr(L);
        \(qr(Δ), [1; zeros(∞)]; tolerance=1E-1)
    end

    @testset "Legendre" begin
        P = Legendre()
        P² = RectPolynomial(Fill(P, 2))
        𝐱 = axes(P²,1)
        f = P² / P² \ broadcast(𝐱 -> ((x,y) = 𝐱; exp(x*cos(y))), 𝐱)
        @test f[SVector(0.1,0.2)] ≈ exp(0.1cos(0.2))

        @test (P²[:,Block.(1:100)] \ f) ≈ f.args[2][Block.(1:100)]
    end

    @testset "Weak Laplacian" begin
        W = Weighted(Jacobi(1,1))
        P = Legendre()
        W² = RectPolynomial(Fill(W, 2))
        P² = RectPolynomial(Fill(P, 2))
        𝐱 = axes(P²,1)
        D_x,D_y = PartialDerivative{1}(𝐱),PartialDerivative{2}(𝐱)
        Δ = -((D_x * W²)'*(D_x * W²) + (D_y * W²)'*(D_y * W²))

        f = expand(P² , 𝐱 -> ((x,y) = 𝐱; x^2 + y^2 - 2))

        KR = Block.(Base.OneTo(100))
        @time 𝐜 = Δ[KR,KR] \ (W²'*f)[KR];
        @test W²[SVector(0.1,0.2),KR]'*𝐜 ≈ (1-0.1^2)*(1-0.2^2)/2 

        @test \(Δ, (W²'*f); tolerance=1E-15) ≈ [0.5; zeros(∞)]
    end
end