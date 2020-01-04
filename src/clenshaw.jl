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

convert(::Type{<:ClenshawRecurrenceData{T}}, C::ClenshawRecurrenceData{<:Any,S}) where {T,S} =
    ClenshawRecurrenceData{T,S}(C.space, convert(Vector{BandedMatrix{T,Matrix{T}}}, C.B̃ˣ),
        convert(Vector{BandedMatrix{T,Matrix{T}}}, C.B̃ʸ), convert(Vector{BandedMatrix{T,Matrix{T}}}, C.B),
        convert(Vector{BandedMatrix{T,Matrix{T}}}, C.C))


ClenshawRecurrenceData{T}(sp, N) where T = ClenshawRecurrenceData{T, typeof(sp)}(sp, N)
ClenshawRecurrenceData(sp, N) = ClenshawRecurrenceData{prectype(sp)}(sp, N)

function ClenshawRecurrenceData{T,S}(sp::S, N) where {T,S<:JacobiTriangle}
    B̃ˣ = Vector{BandedMatrix{T,Matrix{T}}}(undef, N-1)
    B̃ʸ = Vector{BandedMatrix{T,Matrix{T}}}(undef, N-1)
    B  = Vector{BandedMatrix{T,Matrix{T}}}(undef, N-1)
    C  = Vector{BandedMatrix{T,Matrix{T}}}(undef, max(N-2,0))
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
            C[K-1] .= @~ Cx *  B̃ˣ[K]
            C[K-1] .= @~ Cy*B̃ʸ[K] + C[K-1]
        end

        B[K] .= @~ Ax * B̃ˣ[K]
        B[K] .= @~ Ay * B̃ʸ[K] + B[K]
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

using InteractiveUtils


struct ClenshawMultiplication{D,S,T} <: Multiplication{D,S,T}
    data::ClenshawRecurrenceData{T,D}
    cfs::PseudoBlockVector{T,Vector{T}}
    space::S
end

convert(::Type{Operator{T}}, C::ClenshawMultiplication{D,S}) where {D,S,T} =
    ClenshawMultiplication{D,S,T}(convert(ClenshawRecurrenceData{T}, C.data),
                                    convert(AbstractVector{T}, C.cfs), C.space)

function ClenshawMultiplication(f::Fun{D,T}, sp::S) where {D,S,T}
    N = Int(block(space(f), ncoefficients(f)))
    ClenshawMultiplication{D,S,T}(ClenshawRecurrenceData(space(f),N),
                                    PseudoBlockArray(pad(f.coefficients, sum(1:N)),1:N),
                                    sp)
end

Multiplication(f::Fun{JacobiTriangle}, S::JacobiTriangle) = ClenshawMultiplication(f, S)


function Base.getproperty(F::ClenshawMultiplication, d::Symbol)
    if d == :f
        return Fun(F.space, F.cfs)
    else
        getfield(F, d)
    end
end

domainspace(M::ClenshawMultiplication) = M.space
rangespace(M::ClenshawMultiplication) = M.space
isbandedblockbanded(::ClenshawMultiplication) = true
blockbandwidths(M::ClenshawMultiplication) = (blocksize(M.cfs,1)-1,blocksize(M.cfs,1)-1)
subblockbandwidths(M::ClenshawMultiplication) = (blocksize(M.cfs,1)-1,blocksize(M.cfs,1)-1)

function getindex(M::ClenshawMultiplication, k::Int, j::Int)
    K,J = block(M.space,k), block(M.space,j)
    M[K:K,J:J][k-sum(1:Int(K)-1),j-sum(1:Int(J)-1)]
end

function BandedBlockBandedMatrix(V::SubOperator{T,<:ClenshawMultiplication,Tuple{BlockRange1,BlockRange1}}) where T
    KR,JR = parentindices(V)
    M = parent(V)
    sp,cfs,C = M.space, M.cfs, M.data
    N = blocksize(cfs,1)
    JKR = Block.(max(1,min(Int(JR[1]),Int(KR[1]))-(N-1)÷2):max(Int(JR[end]),Int(KR[end]))+(N-1)÷2)

    Jx_∞, Jy_∞ = convert.(Operator{T}, jacobioperators(sp))
    Jx, Jy = Jx_∞[JKR, JKR], Jy_∞[JKR,JKR]

    Q = BandedBlockBandedMatrix(Eye{eltype(Jx)}(size(Jx)...), axes(Jx), (0,0), (0,0))
    B2 = Fill(Q,N) .* view(cfs,Block(N))
    if N == 1
        B1 = B2
    else
        B1 = Fill(Q,N-1) .* view(cfs,Block(N-1)) .+ Fill(Jx,N-1) .* (C.B̃ˣ[N-1]*B2) .+ Fill(Jy,N-1) .* (C.B̃ʸ[N-1]*B2) .- C.B[N-1]*B2
        for K = N-2:-1:1
            B1, B2 =  Fill(Q,K) .* view(cfs,Block(K)) .+ Fill(Jx,K) .* (C.B̃ˣ[K]*B1) .+ Fill(Jy,K) .* (C.B̃ʸ[K]*B1) .- C.B[K]*B1 .- C.C[K] * B2 ,  B1
        end
    end


    # relationship between jkr and kr, jr
    KR2,JR2=Block.(Int.(KR).-Int(JKR[1]).+1),Block.(Int.(JR) .- Int(JKR[1]).+1)
    first(B1)[KR2,JR2]
end
