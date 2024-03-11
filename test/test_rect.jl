using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, LinearAlgebra, BlockArrays, FillArrays, Base64, Test
using ClassicalOrthogonalPolynomials: expand
using MultivariateOrthogonalPolynomials: weaklaplacian
using ContinuumArrays: plotgridvalues

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
        T,U = ChebyshevT(),ChebyshevU()
        T² = RectPolynomial(Fill(T, 2))
        T²ₙ = T²[:,Block.(Base.OneTo(5))]
        𝐱 = axes(T²ₙ,1)
        x,y = first.(𝐱),last.(𝐱)
        @test T²ₙ \ one.(x) == [1; zeros(14)]
        @test (T² \ x)[1:5] ≈[0;1;zeros(3)]

        f = expand(T², splat((x,y) -> exp(x*cos(y-0.1))))
        @test f[SVector(0.1,0.2)] ≈ exp(0.1*cos(0.1))

        U² = RectPolynomial(Fill(U, 2))

        @test f[SVector(0.1,0.2)] ≈ exp(0.1cos(0.1))

        TU = RectPolynomial(T,U)
        x,F = ClassicalOrthogonalPolynomials.plan_grid_transform(TU, Block(5))
        f = expand(TU, splat((x,y) -> exp(x*cos(y-0.1))))
        @test f[SVector(0.1,0.2)] ≈ exp(0.1*cos(0.1))
    end

    @testset "Jacobi matrices" begin
        T = ChebyshevT()
        U = ChebyshevU()
        TU = RectPolynomial(T, U)
        X = jacobimatrix(Val{1}(), TU)
        Y = jacobimatrix(Val{2}(), TU)
        𝐱 = axes(TU, 1)
        x, y = first.(𝐱), last.(𝐱)
        @test_broken TU \ (x .* TU) # Should create X, but it fails
        @test_broken TU \ (y .* TU) # Should create Y, but it fails
        f = expand(TU, splat((x,y) -> exp(x*cos(y-0.1))))
        g = expand(TU, splat((x,y) -> x*exp(x*cos(y-0.1))))
        h = expand(TU, splat((x,y) -> y*exp(x*cos(y-0.1))))
        N = 10
        @test (TU \ (X * (TU \ f)))[Block.(1:N)] ≈ (TU \ g)[Block.(1:N)]
        @test (TU \ (Y * (TU \ f)))[Block.(1:N)] ≈ (TU \ h)[Block.(1:N)]
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

        @test P² == RectPolynomial(Jacobi(0,0), Jacobi(0,0))

        @testset "strong form" begin
            𝐱 = axes(W²,1)
            D_x,D_y = PartialDerivative{1}(𝐱),PartialDerivative{2}(𝐱)
            Δ = Q²\(D_x^2 + D_y^2)*W²

            K = Block.(1:200); @time L = Δ[K,K]; @time qr(L);
            \(qr(Δ), [1; zeros(∞)]; tolerance=1E-1)
        end

        @testset "weakform" begin
            Δ = weaklaplacian(W²)
            c = transform(P², _ -> 1)
            f = expand(P², splat((x,y) -> -2*((1-y^2) + (1-x^2))))
            @test (Δ*c)[Block.(1:5)] ≈ (W²'f)[Block.(1:5)]
        end
    end

    @testset "Legendre" begin
        P = Legendre()
        P² = RectPolynomial(Fill(P, 2))
        𝐱 = axes(P²,1)
        f = P² / P² \ broadcast(𝐱 -> ((x,y) = 𝐱; exp(x*cos(y))), 𝐱)
        @test f[SVector(0.1,0.2)] ≈ exp(0.1cos(0.2))

        @test (P²[:,Block.(1:100)] \ f) ≈ f.args[2][Block.(1:100)]
    end

    @testset "Show" begin
        @test stringmime("text/plain", KronPolynomial(Legendre(), Chebyshev())) == "Legendre() ⊗ ChebyshevT()"
        @test stringmime("text/plain", KronPolynomial(Legendre(), Chebyshev(), Jacobi(1,1))) == "Legendre() ⊗ ChebyshevT() ⊗ Jacobi(1.0, 1.0)"
    end

    @testset "Plot" begin
        P = RectPolynomial(Legendre(),Legendre())
        x,F = plotgridvalues(P[:,1])
        @test x == SVector.(ChebyshevGrid{2}(40), ChebyshevGrid{2}(40)')
        @test F == ones(40,40)
    end
end
