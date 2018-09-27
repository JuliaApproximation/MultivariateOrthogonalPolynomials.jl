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
    Jx, Jy = BandedBlockBandedMatrix(Jx_∞[Block.(1:N), Block.(1:N-1)]),
             BandedBlockBandedMatrix(Jy_∞[Block.(1:N), Block.(1:N-1)])


    for K = 1:N-1
        Ax,Ay = view(Jx,Block(K,K)), view(Jy,Block(K,K))
        Bx,By = view(Jx,Block(K+1,K)), view(Jy,Block(K+1,K))
        b₂ = By[K+1,K]

        B̃ˣ[K] = BandedMatrix{T}(Zeros(K,K+1), (0,2))
        B̃ʸ[K] = BandedMatrix{T}(Zeros(K,K+1), (0,2))
        B[K] = BandedMatrix{T}(undef, (K,K+1), (0,2))

        B̃ˣ[K][band(0)] .= inv.(view(Bx, band(0)))
        B̃ˣ[K][end,end] = - inv(b₂) * By[K,K]/Bx[K,K]
        B̃ʸ[K][end,end] = inv(b₂)

        if K > 1
            Cx,Cy = view(Jx,Block(K-1,K)), view(Jy,Block(K-1,K))
            C[K-1] = BandedMatrix{T}(undef, (K-1,K+1), (0,2))
            B̃ˣ[K][end-1,end] = - inv(b₂) * By[K-1,K]/Bx[K-1,K-1]
            C[K-1] .= Mul(Cx , B̃ˣ[K])
            C[K-1] .= Mul(Cy, B̃ʸ[K]) .+ C[K-1]
        end

        B[K] .= Mul(Ax, B̃ˣ[K])
        B[K] .= Mul(Ay, B̃ʸ[K]) .+ B[K]
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
    J == K+1 && return BandedMatrix((U.data.B[K] - U.x*U.data.B̃ˣ[K] - U.y*U.data.B̃ʸ[K]))
    J == K+2 && return BandedMatrix((U.data.C[K]))
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

function myclenshaw(C, cfs, sp, N)
    Jx_∞, Jy_∞ = jacobioperators(sp)
    Jx, Jy = BandedBlockBandedMatrix(Jx_∞[Block.(1:N), Block.(1:N)]),
             BandedBlockBandedMatrix(Jy_∞[Block.(1:N), Block.(1:N)])

    M = length(C.B)+1

    Q = BandedBlockBandedMatrix(Eye{eltype(Jx)}(size(Jx)), blocksizes(Jx), (0,0), (0,0))
    B2 = Fill(Q,M) .* view(cfs,Block(M))
    B1 = Fill(Q,M-1) .* view(cfs,Block(M-1)) .+ Fill(Jx,M-1) .* (C.B̃ˣ[M-1]*B2) .+ Fill(Jy,M-1) .* (C.B̃ʸ[M-1]*B2) .- C.B[M-1]*B2
    for K = M-2:-1:1
        B1, B2 =  Fill(Q,K) .* view(cfs,Block(K)) .+ Fill(Jx,K) .* (C.B̃ˣ[K]*B1) .+ Fill(Jy,K) .* (C.B̃ʸ[K]*B1) .- C.B[K]*B1 .- C.C[K] * B2 ,  B1
    end
    first(B1)
end


using Juno, Atom, Profile
@time C = ClenshawRecurrenceData(JacobiTriangle(), 3)

@time myclenshaw(C, cfs, sp, N)


sum(1:1000)

N = 200
Jx_∞, Jy_∞ = jacobioperators(sp)
@time Jx, Jy = BandedBlockBandedMatrix(Jx_∞[Block.(1:N), Block.(1:N)]),
         BandedBlockBandedMatrix(Jy_∞[Block.(1:N), Block.(1:N)])

@time Q = BandedBlockBandedMatrix(Eye{eltype(Jx)}(size(Jx)), blocksizes(Jx), (0,0), (0,0))
@time B2 = Fill(Q,M) .* view(cfs,Block(M))
@time B1 = Fill(Q,M-1) .* view(cfs,Block(M-1)) .+ Fill(Jx,M-1) .* (C.B̃ˣ[M-1]*B2) .+ Fill(Jy,M-1) .* (C.B̃ʸ[M-1]*B2) .- C.B[M-1]*B2

@time Jx + Jy

@time Fill(Q,M-1) .* view(cfs,Block(M-1))
@time Fill(Jx,M-1) .* (C.B̃ˣ[M-1]*B2)

@time (C.B̃ˣ[M-1]*B2)

M = length(C.B)+1
@which BandedBlockBandedMatrix(Zeros{eltype(Jx)}(size(Jx)), blocksizes(Jx), (0,0), (0,0))
@time Q = BandedBlockBandedMatrix(Zeros{eltype(Jx)}(size(Jx)), blocksizes(Jx).block_sizes, (0,0), (0,0))

@which BandedBlockBandedMatrix(Zeros{eltype(Jx)}(size(Jx)), blocksizes(Jx).block_sizes, (0,0), (0,0))
@time Q.data[1,:] .= 1

blockbandwidths(Q)





@time Q = BandedBlockBandedMatrix(Eye{eltype(Jx)}(size(Jx)), blocksizes(Jx), (0,0), (0,0))
@time B2 = Fill(Q,M) .* view(cfs,Block(M))
@time A,B = Fill(Q,M-1) .* view(cfs,Block(M-1)) , Fill(Jx,M-1) .* (C.B̃ˣ[M-1]*B2)

@btime B2 = Fill(Q,M) .* view(cfs,Block(M))

@which Q*cfs[1]



C = similar(B[1])

A, B =  A[1] , B[1]
using BenchmarkTools
@btime view(C, Block(200,200)) .= view(B, Block(200,200))
VC, VB = view(C, Block(200,200)) , view(B, Block(200,200))
@btime BandedMatrices._banded_copyto!(VC, VB, L, L)

@btime BandedMatrices.banded_copyto!(VC, VB)
using LazyArrays
L = LazyArrays.MemoryLayout(VC)
LazyArrays.MemoryLayout(VB)
@btime copyto!(BandedMatrices.bandeddata(VC) , BandedMatrices.bandeddata(VB))



bandwidths(view(C, Block(200,200)))

bandwidths(view(B, Block(200,200)))

using BlockBandedMatrices
@which BlockBandedMatrices.blockbanded_copyto!(C, B)
blockbanded_axpy!(one(T), A, dest)

@which broadcast(+,A[1],B[1])

A,B

@time Fill(Q,M-1) .* view(cfs,Block(M-1))

using InteractiveUtils
@time Q*1.0

@time Q.data*1.0

@time 1.0*Q

Fill(Q,M-1) .* view(cfs,Block(M-1))


@time A + B


for K = M-2:-1:1
    B1, B2 =  Fill(Q,K) .* view(cfs,Block(K)) .+ Fill(Jx,K) .* (C.B̃ˣ[K]*B1) .+ Fill(Jy,K) .* (C.B̃ʸ[K]*B1) .- C.B[K]*B1 .- C.C[K] * B2 ,  B1
end
first(B1)




@time Multiplication(Fun(Triangle(), cfs), JacobiTriangle())
Profile.clear()
C.C

B1 = Fill(Q,M) .* view(cfs,Block(M))


Q = BandedBlockBandedMatrix(Eye{eltype(Jx)}(size(Jx)), blocksizes(Jx), (0,0), (0,0))
rhs = PseudoBlockArray(Fill(Q, length(cfs)) .* cfs, 1:M)





cfs[1]*I + cfs[2] * (3Jx - I) .+ cfs[3] * (2Jy + Jx -I)

Fun(Triangle(),[0.1])(0.1,0.2)
Fun(Triangle(),[0,1.])(0.1,0.0)
Fun(Triangle(),[0,0,1.0])(0.1,0.1)

2y + x- 1

3x-1

-0.2 * 2



Fill(Q,K) .* cfs[Block(1)]

C.B̃ˣ[K]*view(cfs,Block(K+1))


c = randn(2,2)

x = [randn(2), randn(2)]


Fill(c,2) .* x
Diagonal([c,c])*x

c*x[1]

c*x



@time C.B̃ˣ[K]*view(cfs,Block(K+1))
@time Jx*0.1




@time M = BandedBlockBandedMatrix(A)

v = randn(55)
Fun(JacobiTriangle(), v)(0.1,0.2)
UpperTriangular(M) \ v

ClenshawRecurrence(JacobiTriangle(), 2, 0.1,0.2)









A.data.B[2]


M * [[1,2],[3,4],[5,6]]

Jx_∞, Jy_∞ = jacobioperators(sp)
    Jx, Jy = BandedBlockBandedMatrix(Jx_∞[Block.(1:N), Block.(1:N-1)]),
             BandedBlockBandedMatrix(Jy_∞[Block.(1:N), Block.(1:N-1)])
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
