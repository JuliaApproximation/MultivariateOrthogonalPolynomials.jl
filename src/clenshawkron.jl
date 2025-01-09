
"""
    ClenshawKron(coefficientmatrix, (A₁,A₂), (B₁,B₂), (C₁,C₂), (X₁,X₂)))

represents the multiplication operator of a tensor product of two orthogonal polynomials.  That is,
if `f(x,y) = P(x)*F*Q(y)'` and `a(x,y) = R(x)*A*S(y)'` then

    M = ClenshawKron(A, tuple.(recurrencecoefficients(R), recurrencecoefficients(S)), (jacobimatrix(P), jacobimatrix(Q))
    M * KronTrav(F)

Gives the coefficients of `a(x,y)*f(x,y)`.
"""
struct ClenshawKron{T, Coefs<:AbstractMatrix{T}, Rec<:NTuple{2,NTuple{3,AbstractVector}}, Jac<:NTuple{2,AbstractMatrix}} <: AbstractBandedBlockBandedMatrix{T}
    c::Coefs
    recurrencecoeffients::Rec
    X::Jac
end


copy(M::ClenshawKron) = M
size(M::ClenshawKron) = (ℵ₀, ℵ₀)
axes(M::ClenshawKron) = (blockedrange(oneto(∞)),blockedrange(oneto(∞)))
blockbandwidths(M::ClenshawKron) = (size(M.c,1)-1,size(M.c,1)-1)
subblockbandwidths(M::ClenshawKron) = (size(M.c,2)-1,size(M.c,2)-1)

struct ClenshawKronLayout <: AbstractLazyBandedBlockBandedLayout end
MemoryLayout(::Type{<:ClenshawKron}) = ClenshawKronLayout()

function square_getindex(M::ClenshawKron, N::Block{1})
    # Consider P(x) = L^1_x \ 𝐞_0 
    # So that if a(x,y) = P(x)*c*Q(y)' then we have
    # a(X,Y) = 𝐞_0' * inv(L^1_X') * c * Q(Y)'
    # So we first apply 1D clenshaw to each column

    (A₁,B₁,C₁), (A₂,B₂,C₂) = M.recurrencecoeffients
    X,Y = M.X
    m,n = size(M.c)
    @assert m == n
    # Apply Clenshaw to each column
    g = (a,b,N) -> LazyBandedMatrices.krontrav(a[1:N,1:N], b[1:N,1:N])
    cols = [Clenshaw(M.c[1:m-j+1,j], A₁, B₁, C₁, X) for j=1:n]

    M = m-2+Int(N) # over sample
    Q_Y = forwardrecurrence(n, A₂, B₂, C₂, Y[1:M,1:M])
    +(broadcast((a,b,N) -> LazyBandedMatrices.krontrav(a[1:N,1:N], b[1:N,1:N]), Q_Y, cols, Int(N))...)
end

getindex(M::ClenshawKron, KR::BlockRange{1}, JR::BlockRange{1}) = square_getindex(M, max(maximum(KR), maximum(JR)))[KR,JR]
getindex(M::ClenshawKron, K::Block{1}, J::Block{1}) = square_getindex(M, max(K, J))[K,J]
getindex(M::ClenshawKron, Kk::BlockIndex{1}, Jj::BlockIndex{1}) = M[block(Kk), block(Jj)][blockindex(Kk), blockindex(Jj)]
getindex(M::ClenshawKron, k::Int, j::Int) = M[findblockindex(axes(M,1),k), findblockindex(axes(M,2),j)]

Base.array_summary(io::IO, C::ClenshawKron{T}, inds) where T =
    print(io, Base.dims2string(length.(inds)), " ClenshawKron{$T} with $(size(C.c)) polynomial")

Base.summary(io::IO, C::ClenshawKron) = Base.array_summary(io, C, axes(C))