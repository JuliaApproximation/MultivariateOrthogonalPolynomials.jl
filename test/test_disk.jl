using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, BlockArrays, BandedMatrices, FastTransforms, LinearAlgebra, Test, SpecialFunctions, LazyArrays, InfiniteArrays, Base64
using RecipesBase
import MultivariateOrthogonalPolynomials: ModalTrav, grid, ZernikeTransform, ZernikeITransform, *, ModalInterlace
import ClassicalOrthogonalPolynomials: HalfWeighted, expand
import ForwardDiff: hessian, gradient

@testset "Disk" begin
    @testset "Transform" begin
        N = 5
        T = ZernikeTransform{Float64}(N, 0, 0)
        Ti = ZernikeITransform{Float64}(N, 0, 0)

        v = PseudoBlockArray(randn(sum(1:N)),1:N)
        @test T * (Ti * v) ≈ v


        @test_throws MethodError T * randn(15)
    end
    @testset "Basics" begin
        @test ZernikeWeight(1)[SVector(0.1,0.2)] ≈ (1 - 0.1^2 - 0.2^2)
        @test ZernikeWeight(1) == ZernikeWeight(1)

        @test Zernike() == Zernike()
        @test Zernike(1) ≠ Zernike()
        @test Zernike() ≡ copy(Zernike())

        @test ZernikeWeight() == ZernikeWeight() == ZernikeWeight(0,0) ==
                ZernikeWeight(0) == ZernikeWeight{Float64}() ==
                ZernikeWeight{Float64}(0) == ZernikeWeight{Float64}(0, 0)
        @test ZernikeWeight(1) ≠ ZernikeWeight()
        @test ZernikeWeight() ≡ copy(ZernikeWeight())
    end

    @testset "Evaluation" begin
        r,θ = 0.1, 0.2
        rθ = RadialCoordinate(r,θ)
        xy = SVector(rθ)
        @test Zernike()[rθ,1] ≈ Zernike()[xy,1] ≈ inv(sqrt(π)) ≈ zernikez(0, 0, rθ)
        @test Zernike()[rθ,Block(1)] ≈ Zernike()[xy,Block(1)] ≈ [inv(sqrt(π))]
        @test Zernike()[rθ,Block(2)] ≈ [2r/sqrt(π)*sin(θ), 2r/sqrt(π)*cos(θ)] ≈ [zernikez(1, -1, rθ), zernikez(1, 1, rθ)]
        @test Zernike()[rθ,Block(3)] ≈ [sqrt(3/π)*(2r^2-1),sqrt(6/π)*r^2*sin(2θ),sqrt(6/π)*r^2*cos(2θ)] ≈ [zernikez(2, 0, rθ), zernikez(2, -2, rθ), zernikez(2, 2, rθ)]
        @test Zernike()[rθ,Block(4)] ≈ [zernikez(3, -1, rθ), zernikez(3, 1, rθ), zernikez(3, -3, rθ), zernikez(3, 3, rθ)]

        @test zerniker(5, 0, norm(xy)) ≈ zernikez(5, 0, xy)
    end

    @testset "ModalTrav" begin
        @test ModalTrav(reshape([1],1,1)) == [1]
        @test ModalTrav([1 2 3]) == 1:3
        @test ModalTrav([1 2 3 5 6;
                        4 0 0 0 0]) == 1:6
        @test ModalTrav([1 2 3 5 6 9 10;
                        4 7 8 0 0 0  0]) == 1:10

        @test ModalTrav([1 2 3 5 6 9 10; 4 7 8 0 0 0  0])[Block(3)] == 4:6

        @test_throws ArgumentError ModalTrav([1 2])
        @test_throws ArgumentError ModalTrav([1 2 3 4])
        @test_throws ArgumentError ModalTrav([1 2 3; 4 5 6])
        @test_throws ArgumentError ModalTrav([1 2 3 4; 5 6 7 8])

        for N = 1:10
            v = PseudoBlockArray(1:sum(1:N),1:N)
            if iseven(N)
                @test ModalTrav(v) == [v; zeros(N+1)]
            else
                @test ModalTrav(v) == v
            end
        end
    end

       @testset "Jacobi matrices" begin
        # Setup
        α = 10 * rand()
        Z = Zernike(α)
        xy = axes(Z, 1)
        x, y = first.(xy), last.(xy)
        n = 150

        # X tests
        JX = zeros(n,n)
        for j = 1:n
            JX[1:n,j] = (Z \ (x .* Z[:,j]))[1:n]
        end 
        X = Z \ (x .* Z)
        # The Zernike Jacobi matrices are symmetric for this normalization
        @test issymmetric(X)
        # Consistency with expansion
        @test X[1:150,1:150] ≈ JX
        # Multiplication by x
        f = Z \ (sin.(x.*y) .+ x.^2 .- y)
        xf = Z \ (x.*sin.(x.*y) .+ x.^3 .- x.*y)
        @test X[Block.(1:20),Block.(1:20)]*f[Block.(1:20)] ≈ xf[Block.(1:20)]

        # Y tests
        JY = zeros(n,n)
        for j = 1:n
        JY[1:n,j] = (Z \ (y .* Z[:,j]))[1:n]
        end 
        Y = Z \ (y .* Z)
        # The Zernike Jacobi matrices are symmetric for this normalization
        @test issymmetric(Y)
        # Consistency with expansion
        @test Y[1:150,1:150] ≈ JY
        # Multiplication by y
        f = Z \ (sin.(x.*y) .+ x.^2 .- y)
        yf = Z \ (y.*sin.(x.*y) .+ y .* x.^2 .- y.^2)
        @test Y[Block.(1:20),Block.(1:20)]*f[Block.(1:20)] ≈ yf[Block.(1:20)]
            
        # Multiplication of Jacobi matrices
        @test (X*X)[Block.(1:6),Block.(1:6)] ≈ (X[Block.(1:10),Block.(1:10)]*X[Block.(1:10),Block.(1:10)])[Block.(1:6),Block.(1:6)]
        @test (X*Y)[Block.(1:6),Block.(1:6)] ≈ (X[Block.(1:10),Block.(1:10)]*Y[Block.(1:10),Block.(1:10)])[Block.(1:6),Block.(1:6)]

        # Addition of Jacobi matrices
        @test (X+Y)[Block.(1:6),Block.(1:6)] ≈ (X[Block.(1:6),Block.(1:6)]+Y[Block.(1:6),Block.(1:6)])
        @test (Y+Y)[Block.(1:6),Block.(1:6)] ≈ (Y[Block.(1:6),Block.(1:6)]+Y[Block.(1:6),Block.(1:6)])
            
        # for now, reject non-zero first parameter options
        @test_throws ErrorException("Implement for non-zero first basis parameter.") jacobimatrix(Val(1),Zernike(1,1))  
        @test_throws ErrorException("Implement for non-zero first basis parameter.") jacobimatrix(Val(2),Zernike(1,1))
    end
        
    @testset "Transform" begin
        for (a,b) in ((0,0), (0.1, 0.2), (0,1))
            Zn = Zernike(a,b)[:,Block.(Base.OneTo(3))]
            for k = 1:6
                @test factorize(Zn) \ Zernike(a,b)[:,k] ≈ [zeros(k-1); 1; zeros(6-k)]
            end

            Z = Zernike(a,b);
            xy = axes(Z,1); x,y = first.(xy),last.(xy);
            u = Z * (Z \ exp.(x .* cos.(y)))
            @test u[SVector(0.1,0.2)] ≈ exp(0.1cos(0.2))
        end
    end

    @testset "Laplacian" begin
        # u = r^m*f(r^2) * cos(m*θ)
        # u_r = (m*r^(m-1)*f(r^2) + 2r^(m+1)*f'(r^2)) * cos(m*θ)
        # u_rr = (m*(m-1)*r^(m-2)*f(r^2) + (4m+2)*r^m*f'(r^2) + 2r^(m+1)*f''(r^2)) * cos(m*θ)
        # u_rr + u_r/r + u_θθ/r^2 = (4*(m+1)*f'(r^2) + 2r*f''(r^2)) * r^m * cos(m*θ)
        # t = r^2, dt = 2r * dr, 4*(m+1)*f'(t) + 2sqrt(t)*f''(t) = 4 t^(-m) * d/dt * t^(m+1) f'(t)
        # d/ds * (1-s) * P_n^(1,m)(s) = -n*P_n^(0,m+1)(s)
        # use L^6 and L^6'
        # 2t-1 = s, 2dt = ds


        ℓ, m, b = 6, 2, 1
        x,y = 0.1,0.2
        r = sqrt(x^2+y^2)
        θ = atan(y,x)
        t = r^2
        u = xy -> zernikez(ℓ, m, b, xy)
        ur = r -> zerniker(ℓ, m, b, r)

        f = t -> sqrt(2^(m+b+2-iszero(m))/π) * normalizedjacobip((ℓ-m) ÷ 2, b, m, 2t-1)
        ur = r -> r^m*f(r^2)
        @test ur(r) ≈ zerniker(ℓ, m, b, r)
        @test f(r^2) ≈ r^(-m) * zerniker(ℓ, m, b, r)
        # u = xy -> ((x,y) = xy; ur(norm(xy)) * cos(m*atan(y,x)))
        # t = r^2; 4*(m+1)*derivative(f,t) + 4t*derivative2(f,t)

        # @test derivative(ur,r) ≈  m*r^(m-1)*f(r^2) + 2r^(m+1)*derivative(f,r^2)
        # @test derivative2(ur,r) ≈ m*(m-1)*r^(m-2)*f(r^2) + (4m+2)*r^m * derivative(f,r^2) + 4r^(m+2)*derivative2(f,r^2)
        # @test lapr(ur, m, r) ≈ 4*((m+1) * derivative(f,r^2) + r^2*derivative2(f,r^2)) * r^m
        # @test lapr(ur, m, r) ≈ 4*((m+1) * derivative(f,t) + t*derivative2(f,t)) * t^(m/2)


        ℓ, m, b = 1, 1, 1

        f = t -> sqrt(2^(m+b+2-iszero(m))/π) * (1-t) * normalizedjacobip((ℓ-m) ÷ 2, b, m, 2t-1)
        ur = r -> r^m*f(r^2)
        @test ur(r) ≈ (1-r^2) * zerniker(ℓ, m, b, r)

        D = Derivative(axes(Chebyshev(),1))
        D1 = Normalized(Jacobi(0, m+1)) \ (D * (HalfWeighted{:a}(Normalized(Jacobi(1, m)))))
        D2 = HalfWeighted{:b}(Normalized(Jacobi(1, m))) \ (D * (HalfWeighted{:b}(Normalized(Jacobi(0, m+1)))))

        @test (D1 * D2)[band(0)][1:10] ≈ -((1:∞) .* ((1+m):∞))[1:10]


        ℓ = m = 0; b= 1
        d = -4*((1:∞) .* ((m+1):∞))
        xy = SVector(0.1,0.2)
        # ℓ = m = 0; b= 1
        # u = xy -> (1 - norm(xy)^2) * zernikez(0, 0, 1, xy)
        # @test lap(u, xy...) ≈ Zernike(1)[xy,1] * (-4)

        # u = xy -> (1 - norm(xy)^2) * zernikez(1 , -1, 1, xy)
        # @test lap(u, xy...) ≈ Zernike(1)[xy,2] * (-4) * 1 * 2
        # u = xy -> (1 - norm(xy)^2) * zernikez(1 , 1, 1, xy)
        # @test lap(u, xy...) ≈ Zernike(1)[xy,3] * (-4) * 1 * 2

        # u = xy -> (1 - norm(xy)^2) * zernikez(2 , 0, 1, xy) # eval at 2
        # @test lap(u, xy...) ≈ Zernike(1)[xy,4] * (-4) * 2^2
        # u = xy -> (1 - norm(xy)^2) * zernikez(2 , -2, 1, xy) # eval at 1
        # @test lap(u, xy...) ≈ Zernike(1)[xy,5] * (-4) * 3
        # u = xy -> (1 - norm(xy)^2) * zernikez(2 , 2, 1, xy) # eval at 1
        # @test lap(u, xy...) ≈ Zernike(1)[xy,6] * (-4) * 3

        # u = xy -> (1 - norm(xy)^2) * zernikez(3 , -1, 1, xy) # eval at 2
        # @test lap(u, xy...) ≈ Zernike(1)[xy,7] * (-4) * 2 * 3
        # u = xy -> (1 - norm(xy)^2) * zernikez(3 , 1, 1, xy)
        # @test lap(u, xy...) ≈ Zernike(1)[xy,8] * (-4) * 2 * 3
        # u = xy -> (1 - norm(xy)^2) * zernikez(3 , -3, 1, xy) # eval at 1
        # @test lap(u, xy...) ≈ Zernike(1)[xy,9] * (-4) * 4 * 1
        # u = xy -> (1 - norm(xy)^2) * zernikez(3 , 3, 1, xy) # eval at 1
        # @test lap(u, xy...) ≈ Zernike(1)[xy,10] * (-4) * 4 * 1

        # u = xy -> (1 - norm(xy)^2) * zernikez(4 , 0, 1, xy) # eval at 3
        # @test lap(u, xy...) ≈ Zernike(1)[xy,11] * (-4) * 3^2
        # u = xy -> (1 - norm(xy)^2) * zernikez(4 , -2, 1, xy) # eval at 2
        # @test lap(u, xy...) ≈ Zernike(1)[xy,12] * (-4) * 4 * 2
        # u = xy -> (1 - norm(xy)^2) * zernikez(4 , 2, 1, xy) # eval at 2
        # @test lap(u, xy...) ≈ Zernike(1)[xy,13] * (-4) * 4 * 2
        # u = xy -> (1 - norm(xy)^2) * zernikez(4 , -4, 1, xy) # eval at 1
        # @test lap(u, xy...) ≈ Zernike(1)[xy,14] * (-4) * 5 * 1
        # u = xy -> (1 - norm(xy)^2) * zernikez(4 , 4, 1, xy) # eval at 1
        # @test lap(u, xy...) ≈ Zernike(1)[xy,15] * (-4) * 5 * 1

        WZ = Weighted(Zernike(1)) # Zernike(1) weighted by (1-r^2)
        Δ = Laplacian(axes(WZ,1))
        Δ_Z = Zernike(1) \ (Δ * WZ)
        @test exp.(Δ_Z)[1:10,1:10] == exp.(Δ_Z[1:10,1:10])

        xy = axes(WZ,1)
        x,y = first.(xy),last.(xy)
        u = @. (1 - x^2 - y^2) * exp(x*cos(y))
        Δu = @. (-exp(x*cos(y)) * (4 - x*(-5 + x^2 + y^2)cos(y) + (-1 + x^2 + y^2)cos(y)^2 - 4x*y*sin(y) + x^2*(x^2 + y^2-1)*sin(y)^2))
        @test (WZ * (WZ \ u))[SVector(0.1,0.2)] ≈ u[SVector(0.1,0.2)]
        @test (Δ_Z * (WZ \ u))[1:100]  ≈ (Zernike(1) \ Δu)[1:100]

        @testset "Unweighted" begin
            c = [randn(100); zeros(∞)]
            Z = Zernike()
            Δ = Zernike(2) \ (Laplacian(axes(Z,1)) * Z)
            @test tr(hessian(xy -> (Zernike{eltype(xy)}()*c)[xy], SVector(0.1,0.2))) ≈ (Zernike(2)*(Δ*c))[SVector(0.1,0.2)]

            b = 0.2
            Z = Zernike(b)
            Δ = Zernike(b+2) \ (Laplacian(axes(Z,1)) * Z)
            @test tr(hessian(xy -> (Zernike{eltype(xy)}(b)*c)[xy], SVector(0.1,0.2))) ≈ (Zernike(b+2)*(Δ*c))[SVector(0.1,0.2)]
        end
    end

    @testset "Conversion" begin
        R0 = Normalized(Jacobi(1, 0)) \ Normalized(Jacobi(0, 0))
        R1 = Normalized(Jacobi(1, 1)) \ Normalized(Jacobi(0, 1))
        R2 = Normalized(Jacobi(1, 2)) \ Normalized(Jacobi(0, 2))
        R3 = Normalized(Jacobi(1, 3)) \ Normalized(Jacobi(0, 3))

        xy = SVector(0.1,0.2)
        @test Zernike()[xy,Block(1)[1]] ≈ Zernike(1)[xy,Block(1)[1]]/sqrt(2)

        @test Zernike()[xy,Block(2)[1]] ≈ Zernike(1)[xy,Block(2)[1]]*R1[1,1]/sqrt(2)
        @test Zernike()[xy,Block(2)[2]] ≈ Zernike(1)[xy,Block(2)[2]]*R1[1,1]/sqrt(2)

        @test Zernike()[xy,Block(3)[1]] ≈ R0[1:2,2]'*Zernike(1)[xy,getindex.(Block.(1:2:3),1)]/sqrt(2)
        @test Zernike()[xy,Block(3)[2]] ≈ R2[1,1]*Zernike(1)[xy,Block(3)[2]]/sqrt(2)
        @test Zernike()[xy,Block(3)[3]] ≈ R2[1,1]*Zernike(1)[xy,Block(3)[3]]/sqrt(2)

        @test Zernike()[xy,Block(4)[1]] ≈ R1[1:2,2]'*Zernike(1)[xy,getindex.(Block.(2:2:4),1)]/sqrt(2)
        @test Zernike()[xy,Block(4)[2]] ≈ R1[1:2,2]'*Zernike(1)[xy,getindex.(Block.(2:2:4),2)]/sqrt(2)
        @test Zernike()[xy,Block(4)[3]] ≈ R3[1,1]*Zernike(1)[xy,Block(4)[3]]/sqrt(2)
        @test Zernike()[xy,Block(4)[4]] ≈ R3[1,1]*Zernike(1)[xy,Block(4)[4]]/sqrt(2)

        @test Zernike()[xy,Block(5)[1]] ≈ R0[2:3,3]'*Zernike(1)[xy,getindex.(Block.(3:2:5),1)]/sqrt(2)


        R = Zernike(1) \ Zernike()

        @test R[Block.(Base.OneTo(6)), Block.(Base.OneTo(7))] == R[Block.(1:6), Block.(1:7)]
        @test Zernike()[xy,Block.(1:6)]' ≈ Zernike(1)[xy,Block.(1:6)]'*R[Block.(1:6),Block.(1:6)]

        R = Zernike(2) \ Zernike()
        @test Zernike()[xy,Block.(1:6)]' ≈ Zernike(2)[xy,Block.(1:6)]'*R[Block.(1:6),Block.(1:6)]
    end

    @testset "Lowering" begin
        L0 = Normalized(Jacobi(0, 0)) \ HalfWeighted{:a}(Normalized(Jacobi(1, 0)))
        L1 = Normalized(Jacobi(0, 1)) \ HalfWeighted{:a}(Normalized(Jacobi(1, 1)))
        L2 = Normalized(Jacobi(0, 2)) \ HalfWeighted{:a}(Normalized(Jacobi(1, 2)))
        L3 = Normalized(Jacobi(0, 3)) \ HalfWeighted{:a}(Normalized(Jacobi(1, 3)))

        xy = SVector(0.1,0.2)
        r = norm(xy)
        w = 1 - r^2

        @test w*Zernike(1)[xy,Block(1)[1]] ≈ L0[1:2,1]'*Zernike()[xy,getindex.(Block.(1:2:3),1)] / sqrt(2)

        @test w*Zernike(1)[xy,Block(2)[1]] ≈ L1[1:2,1]'*Zernike()[xy,getindex.(Block.(2:2:4),1)]/sqrt(2)
        @test w*Zernike(1)[xy,Block(2)[2]] ≈ L1[1:2,1]'*Zernike()[xy,getindex.(Block.(2:2:4),2)]/sqrt(2)

        @test w*Zernike(1)[xy,Block(3)[1]] ≈ L0[2:3,2]'*Zernike()[xy,getindex.(Block.(3:2:5),1)]/sqrt(2)
        @test w*Zernike(1)[xy,Block(3)[2]] ≈ L2[1:2,1]'*Zernike()[xy,getindex.(Block.(3:2:5),2)]/sqrt(2)
        @test w*Zernike(1)[xy,Block(3)[3]] ≈ L2[1:2,1]'*Zernike()[xy,getindex.(Block.(3:2:5),3)]/sqrt(2)

        L = Zernike() \ Weighted(Zernike(1))
        @test w*Zernike(1)[xy,Block.(1:5)]' ≈ Zernike()[xy,Block.(1:7)]'*L[Block.(1:7),Block.(1:5)]

        @test exp.(L)[1:10,1:10] == exp.(L[1:10,1:10])

        L = Zernike(1) \ Weighted(Zernike(1))
        @test 2w*Zernike(1)[xy,Block.(1:5)]' ≈ Zernike(1)[xy,Block.(1:7)]'*L[Block.(1:7),Block.(1:5)]

        L = Zernike() \ Weighted(Zernike(2))
        @test w^2*Zernike(2)[xy,Block.(1:5)]' ≈ Zernike()[xy,Block.(1:9)]'*L[Block.(1:9),Block.(1:5)]
    end

    @testset "plotting" begin
        Z = Zernike()
        u = Z * [1; 2; zeros(∞)];
        rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), u);
        g = MultivariateOrthogonalPolynomials.plotgrid(Z[:,1:3])
        @test all(rep[1].args .≈ (first.(g),last.(g),u[g]))

        W = Weighted(Zernike(1))
        u = W * [1; 2; zeros(∞)];
        rep = RecipesBase.apply_recipe(Dict{Symbol, Any}(), u)
        g = MultivariateOrthogonalPolynomials.plotgrid(W[:,1:3])
        @test all(rep[1].args .≈ (first.(g),last.(g),u[g]))
    end

    @testset "partial derivatives" begin
        W = Weighted(Zernike(1))
        Z⁰ = Zernike(0)
        Z¹ = Zernike(1)
        Z² = Zernike(2)

        𝐱 = axes(W,1)
        # ∂ˣ = PartialDerivative{1}(𝐱)
        ∂ʸ = PartialDerivative{2}(𝐱)

        ∂Y⁰ = Z⁰ \ (∂ʸ * W)
        ∂Y¹ = Z¹ \ (∂ʸ * Z⁰)
        ∂Y² = Z² \ (∂ʸ * Z¹)

        B = Block.(1:10); xy = SVector(0.2,0.3)

        for (i,n,m) in zip((1,2,3,4,5,6,14,17), (0,1,1,2,2,2,4,5), (0,-1,1,0,-2,2,-4,1))
            g⁰ =  𝐱 -> gradient(𝐱 -> (1-norm(𝐱)^2)*zernikez(n, m, 1, 𝐱), 𝐱)[2]
            c⁰ = ModalTrav(transform(Z⁰, g⁰)[B])[B]
            @test Z⁰[xy,B]'*∂Y⁰[B,i] ≈ Z⁰[xy, B]' * c⁰

            g¹ =  𝐱 -> gradient(𝐱 -> zernikez(n, m, 0, 𝐱), 𝐱)[2]
            c¹ = ModalTrav(transform(Z¹, g¹)[B])[B]
            @test Z¹[xy,B]'*∂Y¹[B,i] ≈ Z¹[xy, B]' * c¹

            g² =  𝐱 -> gradient(𝐱 -> zernikez(n, m, 1, 𝐱), 𝐱)[2]
            c² = ModalTrav(transform(Z², g²)[B])[B]
            @test Z²[xy,B]'*∂Y²[B,i] ≈ Z²[xy, B]' * c²
        end

    end
end

@testset "Fractional Laplacian on Unit Disk" begin
    @testset "Fractional Laplacian on Disk: (-Δ)^(β) == -Δ when β=1" begin
        WZ = Weighted(Zernike(1.))
        Δ = Laplacian(axes(WZ,1))
        Δ_Z = Zernike(1) \ (Δ * WZ)
        Δfrac = AbsLaplacianPower(axes(WZ,1),1.)
        Δ_Zfrac = Zernike(1) \ (Δfrac * WZ)
        @test Δ_Z[1:100,1:100] ≈ -Δ_Zfrac[1:100,1:100]
    end

    @testset "Fractional Laplacian on Disk: Computing f where (-Δ)^(β) u = f" begin
        @testset "Set 1 - Explicitly known constant f" begin
            # set up basis
            β = 1.34
            Z = Zernike(β)
            WZ = Weighted(Z)
            xy = axes(WZ,1)
            x,y = first.(xy),last.(xy)
            # generate fractional Laplacian
            Δfrac = AbsLaplacianPower(axes(WZ,1),β)
            Δ_Zfrac = Z \ (Δfrac * WZ)
            # define function whose fractional Laplacian is known
            u = @. (1 - x^2 - y^2).^β
            # explicit and computed solutions
            fexplicit0(d,α) = 2^α*gamma(α/2+1)*gamma((d+α)/2)/gamma(d/2) # note that here, α = 2*β
            f = Z*(Δ_Zfrac*(WZ \ u))
            # compare
            @test fexplicit0(2,2*β) ≈ f[(0.1,0.4)] ≈ f[(0.1137,0.001893)] ≈ f[(0.3721,0.3333)]

            # again for different β
            β = 2.11
            Z = Zernike(β)
            WZ = Weighted(Z)
            xy = axes(WZ,1)
            x,y = first.(xy),last.(xy)
            # generate fractional Laplacian
            Δfrac = AbsLaplacianPower(axes(WZ,1),β)
            Δ_Zfrac = Z \ (Δfrac * WZ)
            # define function whose fractional Laplacian is known
            u = @. (1 - x^2 - y^2).^β
            # computed solution
            f = Z*(Δ_Zfrac*(WZ \ u))
            # compare
            @test fexplicit0(2,2*β) ≈ f[(0.14,0.41)] ≈ f[(0.1731,0.091893)] ≈ f[(0.3791,0.333333)]

            # again for different β
            β = 3.14159
            Z = Zernike(β)
            WZ = Weighted(Z)
            xy = axes(WZ,1)
            x,y = first.(xy),last.(xy)
            # generate fractional Laplacian
            Δfrac = AbsLaplacianPower(axes(WZ,1),β)
            Δ_Zfrac = Z \ (Δfrac * WZ)
            # define function whose fractional Laplacian is known
            u = @. (1 - x^2 - y^2).^β
            # computed solution
            f = Z*(Δ_Zfrac*(WZ \ u))
            # compare
            @test fexplicit0(2,2*β) ≈ f[(0.14,0.41)] ≈ f[(0.1837,0.101893)] ≈ f[(0.37222,0.2222)]
        end
        @testset "Set 2 - Explicitly known radially symmetric f" begin
            β = 1.1
            Z = Zernike(β)
            WZ = Weighted(Z)
            xy = axes(WZ,1)
            x,y = first.(xy),last.(xy)
            # generate fractional Laplacian
            Δfrac = AbsLaplacianPower(axes(WZ,1),β)
            Δ_Zfrac = Z \ (Δfrac * WZ)
            # define function whose fractional Laplacian is known
            u = @. (1 - x^2 - y^2).^(β+1)
            # explicit and computed solutions
            fexplicit1(d,α,x) = 2^α*gamma(α/2+2)*gamma((d+α)/2)/gamma(d/2)*(1-(1+α/d)*norm(x)^2) # α = 2*β
            f = Z*(Δ_Zfrac*(WZ \ u))
            # compare
            @test fexplicit1(2,2*β,(0.94,0.01)) ≈ f[(0.94,0.01)]
            @test fexplicit1(2,2*β,(0.14,0.41)) ≈ f[(0.14,0.41)]
            @test fexplicit1(2,2*β,(0.221,0.333)) ≈ f[(0.221,0.333)]

            # again for different β
            β = 2.71999
            Z = Zernike(β)
            WZ = Weighted(Z)
            xy = axes(WZ,1)
            x,y = first.(xy),last.(xy)
            # generate fractional Laplacian
            Δfrac = AbsLaplacianPower(axes(WZ,1),β)
            Δ_Zfrac = Z \ (Δfrac * WZ)
            # define function whose fractional Laplacian is known
            u = @. (1 - x^2 - y^2).^(β+1)
            # explicit and computed solutions
            f = Z*(Δ_Zfrac*(WZ \ u))
            # compare
            @test fexplicit1(2,2*β,(0.94,0.01)) ≈ f[(0.94,0.01)]
            @test fexplicit1(2,2*β,(0.14,0.41)) ≈ f[(0.14,0.41)]
            @test fexplicit1(2,2*β,(0.221,0.333)) ≈ f[(0.221,0.333)]
        end
        @testset "Set 3 - Explicitly known f, not radially symmetric" begin
            # dependence on x
            β = 2.71999
            Z = Zernike(β)
            WZ = Weighted(Z)
            xy = axes(WZ,1)
            x,y = first.(xy),last.(xy)
            # generate fractional Laplacian
            Δfrac = AbsLaplacianPower(axes(WZ,1),β)
            Δ_Zfrac = Z \ (Δfrac * WZ)
            # define function whose fractional Laplacian is known
            u = @. (1 - x^2 - y^2).^(β)*x
            # explicit and computed solutions
            fexplicit2(d,α,x) = 2^α*gamma(α/2+1)*gamma((d+α)/2+1)/gamma(d/2+1)*x[1] # α = 2*β
            f = Z*(Δ_Zfrac*(WZ \ u))
            # compare
            @test fexplicit2(2,2*β,(0.94,0.01)) ≈ f[(0.94,0.01)]
            @test fexplicit2(2,2*β,(0.14,0.41)) ≈ f[(0.14,0.41)]
            @test fexplicit2(2,2*β,(0.221,0.333)) ≈ f[(0.221,0.333)]

            # different β, dependence on y
            β = 1.91239
            Z = Zernike(β)
            WZ = Weighted(Z)
            xy = axes(WZ,1)
            x,y = first.(xy),last.(xy)
            # generate fractional Laplacian
            Δfrac = AbsLaplacianPower(axes(WZ,1),β)
            Δ_Zfrac = Z \ (Δfrac * WZ)
            # define function whose fractional Laplacian is known
            u = @. (1 - x^2 - y^2).^(β)*y
            # explicit and computed solutions
            fexplicit3(d,α,x) = 2^α*gamma(α/2+1)*gamma((d+α)/2+1)/gamma(d/2+1)*x[2] # α = 2*β
            f = Z*(Δ_Zfrac*(WZ \ u))
            # compare
            @test fexplicit3(2,2*β,(0.94,0.01)) ≈ f[(0.94,0.01)]
            @test fexplicit3(2,2*β,(0.14,0.41)) ≈ f[(0.14,0.41)]
            @test fexplicit3(2,2*β,(0.221,0.333)) ≈ f[(0.221,0.333)]
        end
        @testset "Set 4 - Explicitly known f, different non-radially-symmetric example" begin
            # dependence on x
            β = 1.21999
            Z = Zernike(β)
            WZ = Weighted(Z)
            xy = axes(WZ,1)
            x,y = first.(xy),last.(xy)
            # generate fractional Laplacian
            Δfrac = AbsLaplacianPower(axes(WZ,1),β)
            Δ_Zfrac = Z \ (Δfrac * WZ)
            # define function whose fractional Laplacian is known
            u = @. (1 - x^2 - y^2).^(β+1)*x
            # explicit and computed solutions
            fexplicit4(d,α,x) = 2^α*gamma(α/2+2)*gamma((d+α)/2+1)/gamma(d/2+1)*(1-(1+α/(d+2))*norm(x)^2)*x[1] # α = 2*β
            f = Z*(Δ_Zfrac*(WZ \ u))
            # compare
            @test fexplicit4(2,2*β,(0.94,0.01)) ≈ f[(0.94,0.01)]
            @test fexplicit4(2,2*β,(0.14,0.41)) ≈ f[(0.14,0.41)]
            @test fexplicit4(2,2*β,(0.221,0.333)) ≈ f[(0.221,0.333)]

            # different β, dependence on y
            β = 0.141
            Z = Zernike(β)
            WZ = Weighted(Z)
            xy = axes(WZ,1)
            x,y = first.(xy),last.(xy)
            # generate fractional Laplacian
            Δfrac = AbsLaplacianPower(axes(WZ,1),β)
            Δ_Zfrac = Z \ (Δfrac * WZ)
            # define function whose fractional Laplacian is known
            u = @. (1 - x^2 - y^2).^(β+1)*y
            # explicit and computed solutions
            fexplicit5(d,α,x) = 2^α*gamma(α/2+2)*gamma((d+α)/2+1)/gamma(d/2+1)*(1-(1+α/(d+2))*norm(x)^2)*x[2] # α = 2*β
            f = Z*(Δ_Zfrac*(WZ \ u))
            # compare
            @test fexplicit5(2,2*β,(0.94,0.01)) ≈ f[(0.94,0.01)]
            @test fexplicit5(2,2*β,(0.14,0.41)) ≈ f[(0.14,0.41)]
            @test fexplicit5(2,2*β,(0.221,0.333)) ≈ f[(0.221,0.333)]
        end

        @testset "Fractional Poisson equation on Disk: Comparison with explicitly known solutions" begin
            @testset "Set 1 - Radially symmetric solution" begin
                # define basis
                β = 1.1812
                Z = Zernike(β)
                WZ = Weighted(Z)
                xy = axes(WZ,1)
                x,y = first.(xy),last.(xy)
                # generate fractional Laplacian
                Δfrac = AbsLaplacianPower(axes(WZ,1),β)
                Δ_Zfrac = Z \ (Δfrac * WZ)
                # define function whose fractional Laplacian is known
                uexplicit = @. (1 - x^2 - y^2).^(β+1)
                uexplicitcfs = WZ \ uexplicit
                # RHS
                RHS(d,α,x) = 2^α*gamma(α/2+2)*gamma((d+α)/2)/gamma(d/2)*(1-(1+α/d)*norm(x)^2) # α = 2*β
                RHScfs = Z \ @. RHS.(2,2*β,xy)
                # compute solution
                ucomputed = Δ_Zfrac \ RHScfs
                @test uexplicitcfs[1:100] ≈ ucomputed[1:100]
            end
            @testset "Set 2 - Non-radially-symmetric solutions" begin
                # dependence on y
                β = 0.98812
                Z = Zernike(β)
                WZ = Weighted(Z)
                xy = axes(WZ,1)
                x,y = first.(xy),last.(xy)
                # generate fractional Laplacian
                Δfrac = AbsLaplacianPower(axes(WZ,1),β)
                Δ_Zfrac = Z \ (Δfrac * WZ)
                # define function whose fractional Laplacian is known
                uexplicit = @. (1 - x^2 - y^2).^(β+1)*y
                uexplicitcfs = WZ \ uexplicit
                # RHS
                RHS2(d,α,x) = 2^α*gamma(α/2+2)*gamma((d+α)/2+1)/gamma(d/2+1)*(1-(1+α/(d+2))*norm(x)^2)*x[2] # α = 2*β
                RHS2cfs = Z \ @. RHS2.(2,2*β,xy)
                # compute solution
                ucomputed = Δ_Zfrac \ RHS2cfs
                @test uexplicitcfs[1:100] ≈ ucomputed[1:100]

                # different β, dependence on x
                β = 0.506
                Z = Zernike(β)
                WZ = Weighted(Z)
                xy = axes(WZ,1)
                x,y = first.(xy),last.(xy)
                # generate fractional Laplacian
                Δfrac = AbsLaplacianPower(axes(WZ,1),β)
                Δ_Zfrac = Z \ (Δfrac * WZ)
                # define function whose fractional Laplacian is known
                uexplicit = @. (1 - x^2 - y^2).^(β+1)*x
                uexplicitcfs = WZ \ uexplicit
                # RHS
                RHS3(d,α,x) = 2^α*gamma(α/2+2)*gamma((d+α)/2+1)/gamma(d/2+1)*(1-(1+α/(d+2))*norm(x)^2)*x[1] # α = 2*β
                RHS3cfs = Z \ @. RHS3.(2,2*β,xy)
                # compute solution
                ucomputed = Δ_Zfrac \ RHS3cfs
                @test uexplicitcfs[1:100] ≈ ucomputed[1:100]
            end
        end
    end

    @testset "sum" begin
        P = Zernike()
        @test sum(expand(P, 𝐱 -> 1)) ≈ π
        @test sum(expand(P, 𝐱 -> ((x,y) = 𝐱; exp(x*cos(y))))) ≈ 3.4898933353782744
    end

    @testset "Show" begin
        @test stringmime("text/plain", Zernike()) == "Zernike(0.0, 0.0)"
    end
end
