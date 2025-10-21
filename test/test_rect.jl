using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, LinearAlgebra, BlockArrays, FillArrays, Base64, LazyBandedMatrices, ArrayLayouts, Random, StatsBase, Test
using ClassicalOrthogonalPolynomials: expand, coefficients, recurrencecoefficients
using MultivariateOrthogonalPolynomials: weaklaplacian, ClenshawKron
using ContinuumArrays: plotgridvalues, ExpansionLayout, basis, grid
using Base: oneto

Random.seed!(3242)

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
        x,y = coordinates(T²ₙ)
        @test T²ₙ \ one.(x) == [1; zeros(14)]
        @test (T² \ x)[1:5] ≈[0;1;zeros(3)]

        f = splat((x,y) -> exp(x*cos(y-0.1)))
        𝐟 = expand(T², f)
        @test 𝐟[SVector(0.1,0.2)] ≈ f(SVector(0.1,0.2))

        U² = RectPolynomial(Fill(U, 2))
        𝐟 = expand(U², f)
        @test 𝐟[SVector(0.1,0.2)] ≈ f(SVector(0.1,0.2))

        TU = RectPolynomial(T,U)
        𝐟 = expand(TU, f)
        @test 𝐟[SVector(0.1,0.2)] ≈ f(SVector(0.1,0.2))
        
        @testset "matrix" begin
            N = 10
            𝐱 = grid(T², Block(N))
            
            @test T²[𝐱,1] == ones(N,N)
            @test T²[𝐱,2] == first.(𝐱)
            @test T²[𝐱,1:3] == T²[𝐱,Block.(Base.OneTo(2))] == T²[𝐱,[Block(1),Block(2)]] == [ones(N,N) ;;; first.(𝐱) ;;; last.(𝐱)]
            @test T²[𝐱,Block(1)] == [ones(N,N) ;;;]
            @test T²[𝐱,[1 2; 3 4]] ≈ [T²[𝐱,[1,3]] ;;;; T²[𝐱,[2,4]]]
            
            
            F = plan_transform(T², Block(N))
            @test F * f.(𝐱) ≈ transform(T², f)[Block.(1:N)] atol=1E-6
            
            x,y = coordinates(ChebyshevInterval()^2)
            A = [one(x) x y]
            F = plan_transform(T², (Block(N), 3), 1)
            @test F * A[𝐱,:] ≈ [I(3); zeros(52,3)]

            @test T² \ A ≈ [I(3); Zeros(∞,3)]
        end
    end

    @testset "Jacobi matrices" begin
        T = ChebyshevT()
        U = ChebyshevU()
        TU = RectPolynomial(T, U)
        X = jacobimatrix(Val{1}(), TU)
        Y = jacobimatrix(Val{2}(), TU)
        x, y = coordinates(TU)
        N = 10
        KR = Block.(1:N)
        @test (TU \ (x .* TU))[KR,KR] == X[KR,KR]
        @test (TU \ (y .* TU))[KR,KR] == Y[KR,KR]
        f = expand(TU, splat((x,y) -> exp(x*cos(y-0.1))))
        g = expand(TU, splat((x,y) -> x*exp(x*cos(y-0.1))))
        h = expand(TU, splat((x,y) -> y*exp(x*cos(y-0.1))))
        @test (X * (TU \ f))[KR] ≈ (TU \ g)[KR]
        @test (Y * (TU \ f))[KR] ≈ (TU \ h)[KR]
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
        D_x,D_y = Derivative(𝐱,(1,0)),Derivative(𝐱,(0,1))
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
            D_x,D_y = Derivative(𝐱,(1,0)),Derivative(𝐱,(0,1))
            Δ = Q²\((D_x^2 + D_y^2)*W²)

            K = Block.(1:200); @time L = Δ[K,K]; @time qr(L);
            @time \(qr(Δ), [1; zeros(∞)]; tolerance=1E-1)
        end

        @testset "weakform" begin
            Δ = weaklaplacian(W²)
            c = transform(P², _ -> 1)
            f = expand(P², splat((x,y) -> -2*((1-y^2) + (1-x^2))))
            @test (Δ*c)[Block.(1:5)] ≈ (W²'f)[Block.(1:5)]
        end

        @testset "laplacian" begin
            Δ = Q² \ laplacian(W²)
            c = transform(P², _ -> 1)
            f = expand(P², splat((x,y) -> -2*((1-y^2) + (1-x^2))))
            @test (Δ*c)[Block.(1:5)] ≈ (Q² \f)[Block.(1:5)]
            @test laplacian(W² * c)[SVector(0.1,0.2)] ≈ -2*((1-0.2^2) + (1-0.1^2))
            @test abslaplacian(W² * c)[SVector(0.1,0.2)] ≈ 2*((1-0.2^2) + (1-0.1^2))
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
        @test stringmime("text/plain", KronPolynomial(Legendre(), Chebyshev(), Jacobi(1,1))) == "Legendre() ⊗ ChebyshevT() ⊗ Jacobi(1, 1)"
    end

    @testset "Plot" begin
        P = RectPolynomial(Legendre(),Legendre())
        x,F = plotgridvalues(P[:,1])
        @test x == SVector.(ChebyshevGrid{2}(40), ChebyshevGrid{2}(40)')
        @test F == ones(40,40)
    end

    @testset "sum" begin
        for P in (RectPolynomial(Legendre(),Legendre()), RectPolynomial(Legendre(),Chebyshev()))
            p₀ = expand(P, 𝐱 -> 1)
            @test sum(p₀) ≈ 4.0
            f = expand(P, splat((x,y) -> exp(cos(x^2*y))))
            @test sum(f) ≈ 10.546408460894801 # empirical
        end
    end

    @testset "diff" begin
        P = RectPolynomial(Legendre(),Legendre())
        f = expand(P, splat((x,y) -> 1))
        @test diff(f,(1,0))[SVector(0.1,0.2)] == diff(f,(0,1))[SVector(0.1,0.2)] == 0.0
        f = expand(P, splat((x,y) -> x))
        @test diff(f,(1,0))[SVector(0.1,0.2)] ≈ 1.0
        @test diff(f,(0,1))[SVector(0.1,0.2)] == 0.0
        f = expand(P, splat((x,y) -> cos(x*exp(y))))
        @test diff(f,(1,0))[SVector(0.1,0.2)] ≈ -sin(0.1*exp(0.2))*exp(0.2)
        @test diff(f,(0,1))[SVector(0.1,0.2)] ≈ -0.1*sin(0.1*exp(0.2))*exp(0.2)
        @test diff(f,(2,0))[SVector(0.1,0.2)] ≈ -cos(0.1*exp(0.2))*exp(0.4)
        @test diff(f,(1,1))[SVector(0.1,0.2)] ≈ -sin(0.1*exp(0.2))*exp(0.2) - 0.1*cos(0.1*exp(0.2))*exp(0.4)
        @test diff(f,(0,2))[SVector(0.1,0.2)] ≈ -0.1*sin(0.1*exp(0.2))*exp(0.2) - 0.1^2*cos(0.1*exp(0.2))*exp(0.4)
    end

    @testset "KronTrav bug" begin
        W = Weighted(Ultraspherical(3/2))
        D² = diff(W)'diff(W)
        M = W'W
        Dₓ = KronTrav(D²,M)
        @test Dₓ[Block.(1:1),Block.(1:1)] == Dₓ[Block(1,1)]
    end

    @testset "variable coefficients" begin
        T,U = ChebyshevT(), ChebyshevU()
        P = RectPolynomial(T, U)
        x,y = coordinates(P)
        X = P\(x .* P)
        Y = P\(y .* P)

        @test X isa KronTrav
        @test Y isa KronTrav

        a =  (x,y) -> I + x + 2y + 3x^2 +4x*y + 5y^2
        𝐚 = expand(P,splat(a))

        @testset "ClenshawKron" begin
            C = LazyBandedMatrices.paddeddata(LazyBandedMatrices.invdiagtrav(coefficients(𝐚)))

            A = ClenshawKron(C, (recurrencecoefficients(T), recurrencecoefficients(U)), (jacobimatrix(T), jacobimatrix(U)))

            @test copy(A) ≡ A
            @test size(A) == size(X)
            @test summary(A) == "ℵ₀×ℵ₀ ClenshawKron{Float64} with (3, 3) polynomial"

            Ã = a(X,Y)
            for (k,j) in ((Block.(oneto(5)),Block.(oneto(5))), Block.(oneto(5)),Block.(oneto(6)), (Block(2), Block(3)), (4,5),
                        (Block(2)[2], Block(3)[3]), (Block(2)[2], Block(3)))
                @test A[k,j] ≈ Ã[k,j]
            end

            @test A[Block(1,2)] ≈ Ã[Block(1,2)]
            @test A[Block(1,2)][1,2] ≈ Ã[Block(1,2)[1,2]]
        end

        @test P \ (𝐚 .* P) isa ClenshawKron

        @test (𝐚 .* 𝐚)[SVector(0.1,0.2)] ≈ 𝐚[SVector(0.1,0.2)]^2

        𝐛 = expand(RectPolynomial(Legendre(),Ultraspherical(3/2)),splat((x,y) -> cos(x*sin(y))))
        @test (𝐛 .* 𝐚)[SVector(0.1,0.2)] ≈ 𝐚[SVector(0.1,0.2)]𝐛[SVector(0.1,0.2)]

        𝐜 = expand(RectPolynomial(Legendre(),Jacobi(1,0)),splat((x,y) -> cos(x*sin(y))))
    end

    @testset "reshape/vec" begin
        P = RectPolynomial(Legendre(),Chebyshev())
        f = expand(P, splat((x,y) -> cos((x-0.1)*exp(y))))
        F = reshape(f)
        @test F[0.1,0.2] ≈ f[SVector(0.1,0.2)]
        @test vec(F)[SVector(0.1,0.2)] ≈ f[SVector(0.1,0.2)]

        g = F[:,0.2]
        h = F[0.1,:]
        @test MemoryLayout(g) isa ExpansionLayout
        @test MemoryLayout(h) isa ExpansionLayout
        @test g[0.1] ≈ f[SVector(0.1,0.2)]
        @test h[0.2] ≈ f[SVector(0.1,0.2)]

        @test sum(F; dims=1)[1,0.2] ≈ exp(-0.2)*(sin(0.9exp(0.2)) + sin(1.1exp(0.2)))
        # TODO: should be matrix but isn't because of InfiniteArrays/src/reshapedarray.jl:77
        @test_broken sum(F; dims=2)[0.1,1] ≈ 2
        @test sum(F; dims=2)[0.1] ≈ 2
    end

    @testset "sample" begin
        P = RectPolynomial(Legendre(),Legendre())
        f = expand(P, splat((x,y) -> exp(x*cos(y-0.1))))
        F = reshape(f)
        @test sum(F; dims=1)[1,0.2] ≈ 2.346737615950585
        @test sum(F; dims=2)[0.1,1] ≈ 2.1748993079723618
        
        x,y = coordinates(P)
        @test sample(f) isa SVector
        @test sum(sample(f, 100_000))/100_000 ≈ [sum(x .* f)/sum(f),sum(y .* f)/sum(f)] rtol=1E-1
    end

    @testset "qr" begin
        x,y = coordinates(ChebyshevInterval()^2)
        A = [one(x) x y]

        @test A[SVector(0.1,0.2),1] ≈ 1
        @test A[SVector(0.1,0.2),1:3] ≈ A[SVector(0.1,0.2),:] ≈ [1,0.1,0.2]

        P = basis(x)
        P\A


        B[0.1,:]

        P \ A
    end
end
