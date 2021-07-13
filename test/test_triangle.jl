using MultivariateOrthogonalPolynomials, StaticArrays, BlockArrays, BlockBandedMatrices, ArrayLayouts,
        QuasiArrays, Test, ClassicalOrthogonalPolynomials, BandedMatrices, FastTransforms, LinearAlgebra
import MultivariateOrthogonalPolynomials: tri_forwardrecurrence, grid, TriangleRecurrenceA, TriangleRecurrenceB, TriangleRecurrenceC, xy_muladd!

@testset "Triangle" begin
    @testset "basics" begin
        P = JacobiTriangle()
        @test copy(P) ≡ P

        xy = axes(P,1)
        x,y = first.(xy),last.(xy)
        @test xy[SVector(0.1,0.2)] == SVector(0.1,0.2)
        @test x[SVector(0.1,0.2)] == 0.1
        @test y[SVector(0.1,0.2)] == 0.2
    end
    p = (n,k,a,b,c,x,y) -> jacobip(n-k,2k+b+c+1,a,2x-1) * (1-x)^k * jacobip(k,c,b,2y/(1-x)-1)

    @testset "evaluation" begin
        @testset "versus explicit" begin
            x,y = xy = SVector(0.1,0.2)
            for (a,b,c) in ((0,0,0), (1,0,0), (0,1,0), (0,0,1), (0.1,0.2,0.3))
                P = JacobiTriangle(a,b,c)
            
                for n = 0:5, k=0:n
                    @test P[xy,Block(n+1)[k+1]] ≈ p(n,k,a,b,c,x,y) atol=1E-13
                end
            end
        end

        @testset "forwardrecurrnce" begin
            P = JacobiTriangle()
            xy = SVector(0.1,0.2)
            P_N = P[xy, Block.(Base.OneTo(10))]
            @test P_N == P[xy,Block.(1:10)]
            for N = 1:10
                @test P[xy, Block(N)] ≈ P_N[Block(N)] ≈ p.(N-1, 0:N-1, 0,0,0, xy...)
                for j=1:N
                    P[xy,Block(N)[j]] ≈ p(N-1,j-1,0,0,0,xy...)
                end
            end
            @test P[xy,1] == 1
            @test P[xy,2] ≈ p(1,0,0,0,0,xy...)
            @test P[xy,3] ≈ p(1,1,0,0,0,xy...)
            @test P[xy,4] ≈ p(2,0,0,0,0,xy...)

            @test P[[SVector(0.1,0.2),SVector(0.2,0.3)],1] ≈ [1,1]
            @test P[[SVector(0.1,0.2),SVector(0.2,0.3)],Block(2)] ≈ [P[SVector(0.1,0.2),2] P[SVector(0.1,0.2),3];
                                                                    P[SVector(0.2,0.3),2] P[SVector(0.2,0.3),3]]
        end
        @testset "function" begin
            P = JacobiTriangle()
            xy = SVector(0.1,0.2)
            c = PseudoBlockVector([1; Zeros(∞)], (axes(P,2),))
            f = P*c
            @test f[xy] == 1.0
            c = PseudoBlockVector([1:3; Zeros(∞)], (axes(P,2),))
            f = P*c
            @test f[xy] ≈ P[xy,1:3]'*(1:3)
            c = PseudoBlockVector([1:6; Zeros(∞)], (axes(P,2),))
            f = P*c
            @test f[xy] ≈ P[xy,1:6]'*(1:6)

            c = PseudoBlockVector([randn(5050); Zeros(∞)], (axes(P,2),))
            f = P*c
            @test f[xy] ≈ P[xy,1:5050]'*c[1:5050]

            c = PseudoBlockVector([1:10; zeros(∞)], (axes(P,2),))
            f = P*c
            xy = SVector(0.1,0.2)
            @test f[xy] ≈ dot(P[xy,1:10],1:10)
            @test f[[xy, xy.+0.1]] ≈ [f[xy], f[xy.+0.1]]
            @test f[permutedims([xy, xy.+0.1])] ≈ [f[xy] f[xy.+0.1]]

            @testset "block structure missing" begin
                f = P * [1:5; zeros(∞)]
                @test f.args[2][Block(2)] == 2:3
                @test f[xy] ≈ P[xy,1:5]'*(1:5)

                f = P * [1:5; Zeros(∞)]
                @test f.args[2][Block(2)] == 2:3
                @test f[xy] ≈ P[xy,1:5]'*(1:5)
            end
        end
    end

    @testset "transform" begin
        @testset "grid" begin
            P = JacobiTriangle()
            N = M = 20
            P_N = P[:,Block.(Base.OneTo(N))]
            x = [sinpi((2N-2n-1)/(4N))^2 for n in 0:N-1]
            w = [sinpi((2M-2m-1)/(4M))^2 for m in 0:M-1]
            g = [SVector(x[n+1], x[N-n]*w[m+1]) for n in 0:N-1, m in 0:M-1]
            @test grid(P_N) ≈ g
        end

        @testset "relation with transform" begin
            P = JacobiTriangle()
            c = PseudoBlockVector([1:10; zeros(∞)], (axes(P,2),))
            f = P*c
            N = 5
            P_N = P[:,Block.(Base.OneTo(N))]
            g = grid(P_N)
            F = f[g]
            Pl = plan_tri2cheb(F, 0, 0, 0)
            PA = plan_tri_analysis(F)
            @time U = Pl\(PA*F)
            @test MultivariateOrthogonalPolynomials.tridenormalize!(U,0,0,0) ≈ [1 3 6 10 0; 2 5 9 0 0; 4 8 0 0 0; 7 0 0 0 0; 0 0 0 0 0]
        end

        @testset "expansions" begin
            P = JacobiTriangle()
            xy = axes(P,1)
            x,y = first.(xy),last.(xy)
            N = 20
            P_N = P[:,Block.(Base.OneTo(N))]
            u = P_N * (P_N \ (exp.(x) .* cos.(y)))
            @test u[SVector(0.1,0.2)] ≈ exp(0.1)*cos(0.2)

            P_n = P[:,1:200]
            u = P_n * (P_n \ (exp.(x) .* cos.(y)))
            @test u[SVector(0.1,0.2)] ≈ exp(0.1)*cos(0.2)

            @time u = P * (P \ (exp.(x) .* cos.(y)))
            @test u[SVector(0.1,0.2)] ≈ exp(0.1)*cos(0.2)
        end
    end

    @testset "operators" begin
        P = JacobiTriangle()
        xy = axes(P,1)

        ∂ˣ = PartialDerivative{1}(xy)
        ∂ʸ = PartialDerivative{2}(xy)

        @test eltype(∂ˣ) == eltype(∂ʸ) == Float64

        Dˣ = JacobiTriangle(1,0,1) \ (∂ˣ * P)
        Dʸ = JacobiTriangle(0,1,1) \ (∂ʸ * P)

        M = P'P
        @test blockbandwidths(M) == subblockbandwidths(M) == (0,0)
        @test M[1,1] ≈ 1/2
        @test M[2,2] ≈ 1/4
        @test M[3,3] ≈ 1/12
        @test M[4,4] ≈ 1/6
        @test M[5,5] ≈ 1/18

        Rx = JacobiTriangle(1,0,0) \ P
        Ry = JacobiTriangle(0,1,0) \ P
        Rz = JacobiTriangle(0,0,1) \ P


        x,y = 0.1,0.2
    
        for n=1:5, k=0:n-1
            @test p(n,k,0,0,0,x,y) ≈ p(n-1,k,1,0,0,x,y) *  Rx[Block(n)[k+1], Block(n+1)[k+1]] + p(n,k,1,0,0,x,y) *  Rx[Block(n+1)[k+1], Block(n+1)[k+1]]
        end
        for n=0:5
            k = n
            @test p(n,k,0,0,0,x,y) ≈ p(n,k,1,0,0,x,y) *  Rx[Block(n+1)[k+1], Block(n+1)[k+1]]
        end

        @test P[SVector(x,y),Block.(1:5)]' ≈ JacobiTriangle(1,0,0)[SVector(x,y),Block.(1:5)]' * Rx[Block.(1:5),Block.(1:5)]
        @test P[SVector(x,y),Block.(1:5)]' ≈ JacobiTriangle(0,1,0)[SVector(x,y),Block.(1:5)]' * Ry[Block.(1:5),Block.(1:5)]
        @test P[SVector(x,y),Block.(1:5)]' ≈ JacobiTriangle(0,0,1)[SVector(x,y),Block.(1:5)]' * Rz[Block.(1:5),Block.(1:5)]


        c = [randn(100); zeros(∞)]
        @test (P*c)[SVector(x,y)] ≈ (JacobiTriangle(1,0,0)*(Rx*c))[SVector(x,y)]
        @test (P*c)[SVector(x,y)] ≈ (JacobiTriangle(0,1,0)*(Ry*c))[SVector(x,y)]
        @test (P*c)[SVector(x,y)] ≈ (JacobiTriangle(0,0,1)*(Rz*c))[SVector(x,y)]

        Lx = P \ WeightedTriangle(1,0,0)
        Ly = P \ WeightedTriangle(0,1,0)
        Lz = P \ WeightedTriangle(0,0,1)

        for n=0:5, k=0:n
            @test x*p(n,k,1,0,0,x,y) ≈ p(n,k,0,0,0,x,y) *  Lx[Block(n+1)[k+1], Block(n+1)[k+1]] + p(n+1,k,0,0,0,x,y) *  Lx[Block(n+2)[k+1], Block(n+1)[k+1]]
        end

        @test WeightedTriangle(1,0,0)[SVector(x,y),Block.(1:5)]' ≈ P[SVector(x,y),Block.(1:6)]' * Lx[Block.(1:6),Block.(1:5)]
        @test WeightedTriangle(0,1,0)[SVector(x,y),Block.(1:5)]' ≈ P[SVector(x,y),Block.(1:6)]' * Ly[Block.(1:6),Block.(1:5)]
        @test WeightedTriangle(0,0,1)[SVector(x,y),Block.(1:5)]' ≈ P[SVector(x,y),Block.(1:6)]' * Lz[Block.(1:6),Block.(1:5)]

        @test (WeightedTriangle(1,0,0)*c)[SVector(x,y)] ≈ (P*(Lx*c))[SVector(x,y)]
        @test (WeightedTriangle(0,1,0)*c)[SVector(x,y)] ≈ (P*(Ly*c))[SVector(x,y)]
        @test (WeightedTriangle(0,0,1)*c)[SVector(x,y)] ≈ (P*(Lz*c))[SVector(x,y)]

        ∂ˣ² = (∂ˣ)^2
        ∂ʸ² = (∂ʸ)^2
        @test ∂ˣ² isa ApplyQuasiMatrix{<:Any,typeof(^)}
        @test ∂ʸ² isa ApplyQuasiMatrix{<:Any,typeof(^)}

        Dˣ² = JacobiTriangle(2,0,2) \ (∂ˣ² * P)
        Dˣ² = JacobiTriangle(2,0,2) \ (∂ˣ * (∂ˣ * P))
        Dʸ² = JacobiTriangle(0,2,2) \ (∂ʸ * (∂ʸ * P))

        @testset "jacobi" begin
            P = JacobiTriangle()
            xy = axes(P,1)
            x,y = first.(xy),last.(xy)
            X = P \ (x .* P)
            Y = P \ (y .* P)

            N = 100
            @test X.args[1][Block.(Base.OneTo(N)), Block.(Base.OneTo(N))] isa BandedBlockBandedMatrix

            @test X[Block(2,2)] isa BandedMatrix

            @test X[Block.(1:5),Block.(1:5)] ≈ (Lx * Rx)[Block.(1:5),Block.(1:5)]
            @test Y[Block.(1:5),Block.(1:5)] ≈ (Ly * Ry)[Block.(1:5),Block.(1:5)]

            @test blockbandwidths(X) == blockbandwidths(Y) == (1,1)
            @test subblockbandwidths(X) == (0,0)
            @test subblockbandwidths(Y) == (1,1)

            @testset "Test with exact" begin
                x,y = 0.1,0.2
                N = 0
                @test [p.(N,0:N,0,0,0,x,y); p.(N+1,0:N+1,0,0,0,x,y)]'* X[Block.(N+1:N+2),1] ≈ x
                @test [p.(N,0:N,0,0,0,x,y); p.(N+1,0:N+1,0,0,0,x,y)]'* Y[Block.(N+1:N+2),1] ≈ y
                for N = 1:5
                    @test [p.(N-1,0:N-1,0,0,0,x,y); p.(N,0:N,0,0,0,x,y); p.(N+1,0:N+1,0,0,0,x,y)]'* X[Block.(N:N+2),Block(N+1)] ≈ x * p.(N,0:N,0,0,0,x,y)'
                    @test [p.(N-1,0:N-1,0,0,0,x,y); p.(N,0:N,0,0,0,x,y); p.(N+1,0:N+1,0,0,0,x,y)]'* Y[Block.(N:N+2),Block(N+1)] ≈ y * p.(N,0:N,0,0,0,x,y)'
                end
            end

            @testset "Recurrence" begin
                @testset "B⁺" begin
                    B⁺ = function(N)
                        Bˣ = X[Block(N+1,N)]'
                        Bʸ = Y[Block(N+1,N)]'
                        B¹ = Bˣ[:,1:N]
                        b₂ = Bʸ[N,N+1]
                        b₁ = if N == 1
                            -b₂ \ [Bʸ[N,N]/Bˣ[N,N]]
                        else
                            -b₂ \ [zeros(N-2); Bʸ[N,N-1]/Bˣ[N-1,N-1]; Bʸ[N,N]/Bˣ[N,N]]
                        end
                        [inv(B¹) zeros(N,N-1) zeros(N);
                            b₁' zeros(1,N-1) inv(b₂)]
                    end

                    # B⁺! = function (w, N, v)
                    #     w[1:N] .= X[Block(N+1,N)][band(0)] .\ view(v,1:N)
                    # end

                    for N = 1:5
                        Bˣ = X[Block(N+1,N)]'; Bʸ = Y[Block(N+1,N)]'; B = [Bˣ; Bʸ]
                        Aˣ = X[Block(N,N)]'; Aʸ = Y[Block(N,N)]'; A = [Aˣ; Aʸ]
                        if N > 1
                            Cˣ = X[Block(N-1,N)]'; Cʸ = Y[Block(N-1,N)]'; C = [Cˣ; Cʸ]
                        end
                        @test B⁺(N) * B ≈ I
                        @test TriangleRecurrenceA(N, X, Y) ≈ B⁺(N)
                        @test TriangleRecurrenceB(N, X, Y) ≈ B⁺(N)*[-Aˣ; -Aʸ]
                        if N > 1
                            @test TriangleRecurrenceC(N, X, Y) ≈ B⁺(N)*[Cˣ; Cʸ]
                        end

                        x,y = 0.1,0.2
                        v = randn(N)
                        w = randn(N+1)
                        @test xy_muladd!((x,y), TriangleRecurrenceA(N,X,Y),  v, 2.0, copy(w)) ≈
                            B⁺(N) * [x*Eye(N); y*Eye(N)]*v + 2w
                        @test xy_muladd!((x,y), TriangleRecurrenceA(N,X,Y)', w, 2.0, copy(v)) ≈
                            (B⁺(N) * [x*Eye(N); y*Eye(N)])'*w + 2v

                        @test mul!(w, TriangleRecurrenceB(N, X, Y), v) ≈ B⁺(N)*[-Aˣ; -Aʸ]*v
                        @test mul!(v, TriangleRecurrenceB(N, X, Y)', w) ≈ (B⁺(N)*[-Aˣ; -Aʸ])'*w
                        @test muladd!(3.0, TriangleRecurrenceB(N, X, Y), v, 2.0, copy(w)) ≈ 3B⁺(N)*[-Aˣ; -Aʸ]*v + 2w
                        @test muladd!(3.0, TriangleRecurrenceB(N, X, Y)', w, 2.0, copy(v)) ≈ 3(B⁺(N)*[-Aˣ; -Aʸ])'*w + 2v


                        if N > 1
                            u = randn(N-1)
                            @test muladd!(3.0, TriangleRecurrenceC(N, X, Y), u, 2.0, copy(w)) ≈
                                3B⁺(N)*[Cˣ; Cʸ]*u + 2w
                            @test muladd!(3.0, TriangleRecurrenceC(N, X, Y)', w, 2.0, copy(u)) ≈
                                3(B⁺(N)*[Cˣ; Cʸ])'*w + 2u

                            # need in-place to minimise buffers in Clenshaw
                            w̃ = copy(w)
                            @test lmul!(TriangleRecurrenceC(N, X, Y)', w̃) === w̃
                            @test w̃ ≈ (B⁺(N)*[Cˣ; Cʸ])'*w
                        end
                    end

                    @testset "comparison with exact" begin
                        x,y = 0.1,0.2

                        N = 1
                        Aˣ = X[Block(N,N)]'
                        Aʸ = Y[Block(N,N)]'
                        P_0 = [1]
                        P_1 = B⁺(1)*[x*I-Aˣ; y*I-Aʸ]*P_0
                        @test P_1 ≈ p.(N,0:N,0,0,0,x,y)

                        N = 2
                        Aˣ = X[Block(N,N)]'; Aʸ = Y[Block(N,N)]'; A = [Aˣ-x*I; Aʸ-y*I]
                        Bˣ = X[Block(N+1,N)]'; Bʸ = Y[Block(N+1,N)]'; B = [Bˣ; Bʸ]
                        Cˣ = X[Block(N-1,N)]'; Cʸ = Y[Block(N-1,N)]'; C = [Cˣ; Cʸ]
                        @test norm(C * p.(N-2,0:N-2,0,0,0,x,y) + A * p.(N-1,0:N-1,0,0,0,x,y) + B * p.(N,0:N,0,0,0,x,y)) ≤ 10eps()

                        P_2 = B⁺(N)*([x*I-Aˣ; y*I-Aʸ]*P_1 - [Cˣ; Cʸ]*P_0 )
                        @test p.(N,0:N,0,0,0,x,y) ≈ P_2

                        N = 3
                        Aˣ = X[Block(N,N)]'; Aʸ = Y[Block(N,N)]'; A = [Aˣ-x*I; Aʸ-y*I]
                        Cˣ = X[Block(N-1,N)]'; Cʸ = Y[Block(N-1,N)]'; C = [Cˣ; Cʸ]
                        P_3 = B⁺(N)*([x*I-Aˣ; y*I-Aʸ]*P_2 - [Cˣ; Cʸ]*P_1 )
                        @test p.(N,0:N,0,0,0,x,y) ≈ P_3

                        @testset "simplify" begin
                            Bˣ = X[Block(N+1,N)]'; Bʸ = Y[Block(N+1,N)]'; B = [Bˣ; Bʸ]

                            b₂ = Bʸ[N,N+1]
                            Bc = -Bʸ[N,N-1]/(b₂*Bˣ[N-1,N-1])*Cˣ[N-1,N-1] + Cʸ[N,N-1]/b₂
                            @test B⁺(N) * [Cˣ; Cʸ] ≈ [Diagonal(Bˣ[band(0)][1:end-1] .\ Cˣ[band(0)]);
                                                        zeros(1, N-1);
                                                      [zeros(1,N-2) Bc]]

                            Ba_1 = -Bʸ[N,N-1]/(b₂*Bˣ[N-1,N-1]) * (x - Aˣ[N-1,N-1]) + (-Aʸ[N,N-1])/b₂
                            Ba_2 = -Bʸ[N,N]/(b₂*Bˣ[N,N]) * (x - Aˣ[N,N]) + (y-Aʸ[N,N])/b₂

                            @test B⁺(N) * [x*I-Aˣ; y*I-Aʸ] ≈ [Diagonal(Bˣ[band(0)] .\ (x .- Aˣ[band(0)]));
                                                             [zeros(1,N-2) Ba_1 Ba_2]]

                            w = Vector{Float64}(undef,N+1)
                            w[1:N-1] .= (Bˣ[band(0)][1:end-1] .\ Cˣ[band(0)] .* P_1)
                            w[N] = 0
                            w[N+1] = Bc * P_1[end]
                            @test w ≈ B⁺(N) * [Cˣ; Cʸ] * P_1
                            w[1:N] .= Bˣ[band(0)] .\ (x .- Aˣ[band(0)]) .* P_2
                            w[N+1] = Ba_1 * P_2[end-1] + Ba_2 * P_2[end]
                            @test w ≈ B⁺(N) * [x*I-Aˣ; y*I-Aʸ] * P_2
                            w[1:N-1] .-= (Bˣ[band(0)][1:end-1] .\ Cˣ[band(0)] .* P_1)
                            w[N+1] -= Bc * P_1[end]
                            @test w ≈ P_3
                        end
                    end
                end

                @testset "forward Recurrence" begin
                    @test tri_forwardrecurrence(1, X, Y, 0.1, 0.2) ≈ [1.0]
                    @test tri_forwardrecurrence(2, X, Y, 0.1, 0.2) ≈ [1.0; p.(1,0:1,0,0,0,0.1,0.2)]
                    @test tri_forwardrecurrence(4, X, Y, 0.1, 0.2)[Block(4)] ≈ p.(3,0:3,0,0,0,0.1,0.2)
                end
            end


            @testset "truncations" begin
                N = 100;
                KR,JR = Block.(1:N),Block.(1:N)

                @test Dˣ[KR,JR] isa BandedBlockBandedMatrix
                @test Lx[KR,JR] isa BandedBlockBandedMatrix
                @test Rx[KR,JR] isa BandedBlockBandedMatrix
                @test Ly[KR,JR] isa BandedBlockBandedMatrix
                @test Ry[KR,JR] isa BandedBlockBandedMatrix
                @test X[KR,JR] isa BandedBlockBandedMatrix
                @test Y[KR,JR] isa BandedBlockBandedMatrix
            end

            @testset "other parameters" begin
                P = JacobiTriangle(1,0,0)
                xy = axes(P,1)
                x,y = first.(xy),last.(xy)
                X = P \ (x .* P)
                Y = P \ (y .* P)

                P_ex = PseudoBlockVector{Float64}(undef, 1:5)
                for n = 0:4, k=0:n
                    P_ex[Block(n+1)[k+1]] = p(n,k,1,0,0,0.1,0.2)
                end
                @test P_ex'*X[Block.(1:5),Block.(1:4)] ≈ 0.1 * P_ex[Block.(1:4)]'
                @test P_ex'*Y[Block.(1:5),Block.(1:4)] ≈ 0.2 * P_ex[Block.(1:4)]'
                @test P[SVector(0.1,0.2),Block.(1:5)]'*X[Block.(1:5),Block.(1:4)] ≈ 0.1 * P[SVector(0.1,0.2),Block.(1:4)]'
            end
        end

        @testset "higher order conversion" begin
            P = JacobiTriangle()
            Q = JacobiTriangle(1,1,1)
            R = Q \ P
            L = P \ Weighted(Q)

            xy = SVector(0.1,0.2)
            @test P[xy,1:10]' ≈ Q[xy,1:10]' * R[1:10,1:10]
        end
    end

    @testset "AngularMomentum" begin
        P = JacobiTriangle()
        P¹ = JacobiTriangle(1,1,1)
        xy = axes(P,1)
        x,y = first.(xy),last.(xy)
        ∂ˣ = PartialDerivative{1}(xy)
        ∂ʸ = PartialDerivative{2}(xy)
        L1 = x .* ∂ʸ
        L2 = y .* ∂ˣ
        L = x .* ∂ʸ - y .* ∂ˣ
        A = P¹ \ (L1 * P)
        B = P¹ \ (L2 * P)
        C = P¹ \ (L * P)
        @test C[1:10,1:10] ≈ A[1:10,1:10] - B[1:10,1:10]
    end
end
