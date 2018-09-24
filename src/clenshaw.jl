####
# Here we implement multivariable clenshaw
#
# It's hard-coded for Triangles right now.
####
struct ClenshawRecurrenceData{T, S}
    space::S
    B̃ˣ::Vector{BandedMatrix{T,Matrix{T}}}
    B̃ʸ::Vector{BandedMatrix{T,Matrix{T}}}
    B::Vector{BandedMatrix{T,Matrix{T}}} # B̃ˣAˣ + B̃ʸAʸ
    C::Vector{BandedMatrix{T,Matrix{T}}} #B̃ˣCˣ + B̃ʸCʸ
end

ClenshawRecurrenceData{T}(sp, N) where T = ClenshawRecurrenceData{T, typeof(sp)}(sp, N)
ClenshawRecurrenceData(sp, N) = ClenshawRecurrenceData{prectype(sp)}(sp, N)

function ClenshawRecurrenceData{T,S}(sp::S, N) where {T,S<:JacobiTriangle}
    B̃ˣ = Vector{BandedMatrix{T,Matrix{T}}}(undef, N-1)
    B̃ʸ = Vector{BandedMatrix{T,Matrix{T}}}(undef, N-1)
    B  = Vector{BandedMatrix{T,Matrix{T}}}(undef, N-1)
    C  = Vector{BandedMatrix{T,Matrix{T}}}(undef, N-2)
    Jx_∞, Jy_∞ = jacobioperators(sp)
    Jx, Jy = BandedBlockBandedMatrix(Jx_∞[Block.(1:N), Block.(1:N-1)]'),
             BandedBlockBandedMatrix(Jy_∞[Block.(1:N), Block.(1:N-1)]')


    for K = 1:N-1
        Ax,Ay = view(Jx,Block(K,K)), view(Jy,Block(K,K))
        Bx,By = view(Jx,Block(K,K+1)), view(Jy,Block(K,K+1))
        b₂ = By[K,K+1]

        B̃ˣ[K] = BandedMatrix{T}(Zeros(K+1,K), (2,0))
        B̃ʸ[K] = BandedMatrix{T}(Zeros(K+1,K), (2,0))
        B[K] = BandedMatrix{T}(undef, (K+1,K), (2,0))

        B̃ˣ[K][band(0)] .= inv.(view(Bx, band(0)))
        B̃ˣ[K][end,end] = - inv(b₂) * By[K,K]/Bx[K,K]
        B̃ʸ[K][end,end] = inv(b₂)

        if K > 1
            Cx,Cy = view(Jx,Block(K,K-1)), view(Jy,Block(K,K-1))
            C[K-1] = BandedMatrix{T}(undef, (K+1,K-1), (2,0))
            B̃ˣ[K][end,end-1] = - inv(b₂) * By[K,K-1]/Bx[K-1,K-1]
            C[K-1] .= Mul(B̃ˣ[K] , Cx)
            C[K-1] .= Mul(B̃ʸ[K] , Cy) .+ C[K-1]
        end

        B[K] .= Mul(B̃ˣ[K] , Ax)
        B[K] .= Mul(B̃ʸ[K] , Ay) .+ B[K]
    end

    ClenshawRecurrenceData{T,S}(sp, B̃ˣ, B̃ʸ, B, C)
end

struct ClenshawRecurrence{T, DATA} <: AbstractBlockMatrix{T}
    data::DATA
    x::T
    y::T
end

ClenshawRecurrence(sp, N, x, y) = ClenshawRecurrence(ClenshawRecurrenceData(sp,N), x, y)

blocksizes(U::ClenshawRecurrence) = BlockSizes(1:(length(U.data.B̃ˣ)+1),1:(length(U.data.B̃ˣ)+1))

blockbandwidths(::ClenshawRecurrence) = (0,2)
subblockbandwidths(::ClenshawRecurrence) = (0,2)

@inline function getblock(U::ClenshawRecurrence{T}, K::Int, J::Int) where T
    J == K && return BandedMatrix(Eye{T}(K,K))
    J == K+1 && return BandedMatrix(transpose(U.data.B[K] - U.x*U.data.B̃ˣ[K] - U.y*U.data.B̃ʸ[K]))
    J == K+2 && return BandedMatrix(transpose(U.data.C[K]))
    return BandedMatrix(Zeros{T}(K,J))
end

@inline function getindex(block_arr::ClenshawRecurrence, blockindex::BlockIndex{2})
    @inbounds block = getblock(block_arr, blockindex.I...)
    @boundscheck checkbounds(block, blockindex.α...)
    @inbounds v = block[blockindex.α...]
    return v
end

function getindex(U::ClenshawRecurrence, i::Int, j::Int)
    @boundscheck checkbounds(U, i...)
    @inbounds v = U[global2blockindex(blocksizes(U), (i, j))]
    return v
end

@time A = ClenshawRecurrence(JacobiTriangle(), 10, 0.1, 0.2);
A.data.B[2]
@time M = BandedBlockBandedMatrix(A)

Jx_∞, Jy_∞ = jacobioperators(sp)
    Jx, Jy = BandedBlockBandedMatrix(Jx_∞[Block.(1:N), Block.(1:N-1)]'),
             BandedBlockBandedMatrix(Jy_∞[Block.(1:N), Block.(1:N-1)]')
             Ax, Ay = Jx[Block(K,K)], Jy[Block(K,K)]
B̃ = [Matrix(A.data.B̃ˣ[2]) Matrix(A.data.B̃ʸ[2])]

B̃ˣ = A.data.B̃ˣ
B̃ʸ = A.data.B̃ʸ

B̃ˣ[K],Ax , B̃ʸ[K],Ay




f = Fun(Triangle(), randn(size(A,1)))
# P= plan_evaluate(f)
@time P(0.1,0.2)

P = PseudoBlockArray([Fun(JacobiTriangle(), [zeros(k-1); 1])(0.1,0.2) for k=1:sum(1:10)], 1:10)

A'*P

N = 10
sp = JacobiTriangle()

x,y = 0.1,0.2
Jx*P - (x*P)[1:45] |> norm

Jy*P - (y*P)[1:45] |> norm


A.data.B̃ˣ


K = 2; B̃*[Matrix(Jx[Block(K,K+1)]); Matrix(Jy[Block(K,K+1)])]


A.data.B[2]
B̃ˣ[2]
A.data.B̃ˣ[2]*Matrix(Jx[Block(K,K)]) + A.data.B̃ʸ[2] * Matrix(Jy[Block(K,K)])
B̃*[Matrix(Jx[Block(K,K)]); Matrix(Jy[Block(K,K)])]
B̃*[Matrix(Jx[Block(K,K)]-x*I); Matrix(Jy[Block(K,K)] - y*I)]

K

B = A.data.B

B[K] .= Mul(B̃ˣ[K] , Ax)
B[K] .= Mul(B̃ʸ[K] , Ay) .+ B[K]
(A')[Block(3,2)]


A.data

B̃*[Matrix(Jx[Block(K,K)]); Matrix(Jy[Block(K,K)])]

B[K]

B̃ˣ[K] * Ax + B̃ʸ[K] * Ay
Matrix(M)'*P

P'*f.coefficients

P'*f.coefficients

UpperTriangular(M)\f.coefficients

v = randn(5050)

@time UpperTriangular(M) \ v

A[Block(3,4)]




@time BandedBlockBandedMatrix(A)

A'*P

(A')

x,y = 0.1,0.2

x*P[Block(1)]

Ax*P[Block(1)] + Bx*P[Block(2)]

Jx[Block.(1:5), Block.(1:5)]'*P






sum(1:5)
