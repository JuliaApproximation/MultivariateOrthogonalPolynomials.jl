using MultivariateOrthogonalPolynomials, StaticArrays, BlockArrays, BlockBandedMatrices, ArrayLayouts, QuasiArrays, Test, OrthogonalPolynomialsQuasi, BandedMatrices
import MultivariateOrthogonalPolynomials: tri_forwardrecurrence


@testset "Triangle" begin
    P = JacobiTriangle()
    @test copy(P) ≡ P

    xy = axes(P,1)
    x,y = first.(xy),last.(xy)
    @test xy[SVector(0.1,0.2)] == SVector(0.1,0.2)
    @test x[SVector(0.1,0.2)] == 0.1
    @test y[SVector(0.1,0.2)] == 0.2

    P̃ = (n,k,x,y) -> Jacobi(2k+1,0)[2x-1,n-k+1] * (1-x)^k * Legendre()[2y/(1-x)-1, k+1]

    @testset "evaluation" begin
        xy = SVector(0.1,0.2)
        P_N = P[xy, Block.(1:10)]
        for N = 1:10
            @test P[xy, Block(N)] ≈ P_N[Block(N)] ≈ P̃.(N-1, 0:N-1, xy...)
            for j=1:N
                P[xy,Block(N)[j]] ≈ P̃(N-1,j-1,xy...)
            end
        end
        @test P[xy,1] == 1
        @test P[xy,2] ≈ P̃(1,0,xy...)
        @test P[xy,3] ≈ P̃(1,1,xy...)
        @test P[xy,4] ≈ P̃(2,0,xy...)

        @test P[[SVector(0.1,0.2),SVector(0.2,0.3)],1] ≈ [1,1]
        @test P[[SVector(0.1,0.2),SVector(0.2,0.3)],Block(2)] ≈ [P[SVector(0.1,0.2),2] P[SVector(0.1,0.2),3];
                                                                 P[SVector(0.2,0.3),2] P[SVector(0.2,0.3),3]]
    end

    @testset "operators" begin
        ∂ˣ = PartialDerivative{1}(xy)
        ∂ʸ = PartialDerivative{2}(xy)

        Dˣ = JacobiTriangle(1,0,1) \ (∂ˣ * P)
        Dʸ = JacobiTriangle(0,1,1) \ (∂ʸ * P)

        M = P'P
        @test blockbandwidths(M) == subblockbandwidths(M) == (0,0)

        Rx = JacobiTriangle(1,0,0) \ P
        Lx = P \ WeightedTriangle(1,0,0)
        Ry = JacobiTriangle(0,1,0) \ P
        Ly = P \ WeightedTriangle(0,1,0)

        ∂ˣ² = (∂ˣ)^2
        ∂ʸ² = (∂ʸ)^2
        @test ∂ˣ² isa ApplyQuasiMatrix{<:Any,typeof(^)}
        @test ∂ʸ² isa ApplyQuasiMatrix{<:Any,typeof(^)}

        Dˣ² = JacobiTriangle(2,0,2) \ (∂ˣ² * P)
        Dˣ² = JacobiTriangle(2,0,2) \ (∂ˣ * (∂ˣ * P))
        Dʸ² = JacobiTriangle(0,2,2) \ (∂ʸ * (∂ʸ * P))

        @testset "jacobi" begin
            X = P \ (x .* P)
            Y = P \ (y .* P)

            @test X[Block(2,2)] isa BandedMatrix

            @test X[Block.(1:5),Block.(1:5)] ≈ (Lx * Rx)[Block.(1:5),Block.(1:5)]
            @test Y[Block.(1:5),Block.(1:5)] ≈ (Ly * Ry)[Block.(1:5),Block.(1:5)]

            @test blockbandwidths(X) == blockbandwidths(Y) == (1,1)
            @test subblockbandwidths(X) == (0,0)
            @test subblockbandwidths(Y) == (1,1)

            @testset "Test with exact" begin
                x,y = 0.1,0.2
                N = 0
                @test [P̃.(N,0:N,x,y); P̃.(N+1,0:N+1,x,y)]'* X[Block.(N+1:N+2),1] ≈ x
                @test [P̃.(N,0:N,x,y); P̃.(N+1,0:N+1,x,y)]'* Y[Block.(N+1:N+2),1] ≈ y
                for N = 1:5
                    @test [P̃.(N-1,0:N-1,x,y); P̃.(N,0:N,x,y); P̃.(N+1,0:N+1,x,y)]'* X[Block.(N:N+2),Block(N+1)] ≈ x * P̃.(N,0:N,x,y)'
                    @test [P̃.(N-1,0:N-1,x,y); P̃.(N,0:N,x,y); P̃.(N+1,0:N+1,x,y)]'* Y[Block.(N:N+2),Block(N+1)] ≈ y * P̃.(N,0:N,x,y)'
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
                        @test B⁺(N) * B ≈ I
                    end

                    @testset "comparison with exact" begin
                        x,y = 0.1,0.2

                        N = 1
                        Aˣ = X[Block(N,N)]'
                        Aʸ = Y[Block(N,N)]'
                        P_0 = [1]
                        P_1 = B⁺(1)*[x*I-Aˣ; y*I-Aʸ]*P_0
                        @test P_1 ≈ P̃.(N,0:N,x,y)

                        N = 2
                        Aˣ = X[Block(N,N)]'; Aʸ = Y[Block(N,N)]'; A = [Aˣ-x*I; Aʸ-y*I]
                        Bˣ = X[Block(N+1,N)]'; Bʸ = Y[Block(N+1,N)]'; B = [Bˣ; Bʸ]
                        Cˣ = X[Block(N-1,N)]'; Cʸ = Y[Block(N-1,N)]'; C = [Cˣ; Cʸ]
                        @test norm(C * P̃.(N-2,0:N-2,x,y) + A * P̃.(N-1,0:N-1,x,y) + B * P̃.(N,0:N,x,y)) ≤ 10eps()

                        P_2 = B⁺(N)*([x*I-Aˣ; y*I-Aʸ]*P_1 - [Cˣ; Cʸ]*P_0 )
                        @test P̃.(N,0:N,x,y) ≈ P_2

                        N = 3
                        Aˣ = X[Block(N,N)]'; Aʸ = Y[Block(N,N)]'; A = [Aˣ-x*I; Aʸ-y*I]
                        Cˣ = X[Block(N-1,N)]'; Cʸ = Y[Block(N-1,N)]'; C = [Cˣ; Cʸ]
                        P_3 = B⁺(N)*([x*I-Aˣ; y*I-Aʸ]*P_2 - [Cˣ; Cʸ]*P_1 )
                        @test P̃.(N,0:N,x,y) ≈ P_3

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
                    @test tri_forwardrecurrence(2, X, Y, 0.1, 0.2) ≈ [1.0; P̃.(1,0:1,0.1,0.2)]
                    @test tri_forwardrecurrence(4, X, Y, 0.1, 0.2)[Block(4)] ≈ P̃.(3,0:3,0.1,0.2)
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
        end
    end
end
