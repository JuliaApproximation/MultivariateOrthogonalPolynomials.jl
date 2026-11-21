"""
    KronInterlace(A, B, …)

represents the tensor product basis `A ⊗ B ⊗ …` with the columns interlaced by total degree,
so that for `d = 2` the `N`-th block consists of `A[x,N-j+1]*B[y,j]` for `j = 1:N`.
The factors can be general bases. When every factor is an orthogonal polynomial basis the
`MemoryLayout` is `KronOPLayout`, a multivariate orthogonal polynomial layout,
otherwise it is `KronInterlaceLayout`.
"""
struct KronInterlace{d, T, PP} <: Basis{T}
    args::PP
end

KronInterlace{d,T}(a::NTuple{d,Any}) where {d,T} = KronInterlace{d,T,typeof(a)}(a)
KronInterlace{d}(a::NTuple{d,Any}) where d = KronInterlace{d,mapreduce(eltype, promote_type, a)}(a)
KronInterlace(a::NTuple{d,Any}) where d = KronInterlace{d}(a)
KronInterlace{d,T}(a::Vararg{Any,d}) where {d,T} = KronInterlace{d,T}(a)
KronInterlace{d}(a::Vararg{Any,d}) where d = KronInterlace{d}(a)
KronInterlace(a::Vararg{Any,d}) where d = KronInterlace{d}(a)
KronInterlace{d,T}(a::AbstractVector) where {d,T} = KronInterlace{d,T,typeof(a)}(a)
KronInterlace{d}(a::AbstractVector) where d = KronInterlace{d,eltype(eltype(a))}(a)
KronInterlace(a::AbstractVector) = KronInterlace{length(a)}(a)

function show(io::IO, P::KronInterlace)
    for k = 1:length(P.args)
        print(io, "$(P.args[k])")
        k ≠ length(P.args) && print(io, " ⊗ ")
    end
end

==(A::KronInterlace, B::KronInterlace) = length(A.args) == length(B.args) && all(map(==, A.args, B.args))


"""
    KronInterlaceLayout{d}

is the layout of a `KronInterlace` whose factors are general bases.
"""
struct KronInterlaceLayout{d} <: AbstractBasisLayout end

"""
    KronOPLayout{d}

is the layout of a `KronInterlace` whose factors are all orthogonal polynomials,
so that it is a multivariate orthogonal polynomial.
"""
struct KronOPLayout{d} <: AbstractMultivariateOPLayout{d} end

const KronInterlaceLayouts{d} = Union{KronInterlaceLayout{d}, KronOPLayout{d}}

kroninterlacelayout(::Val{d}, ::Tuple{Vararg{AbstractOPLayout}}) where d = KronOPLayout{d}()
kroninterlacelayout(::Val{d}, _) where d = KronInterlaceLayout{d}()

kroninterlace_factorlayouts(::Type{PP}) where PP<:Tuple = map(MemoryLayout, fieldtypes(PP))
kroninterlace_factorlayouts(::Type{PP}) where PP<:AbstractVector = (MemoryLayout(eltype(PP)),)
MemoryLayout(::Type{<:KronInterlace{d,<:Any,PP}}) where {d,PP} = kroninterlacelayout(Val(d), kroninterlace_factorlayouts(PP))

const RectPolynomial{T, PP} = KronInterlace{2, T, PP}

copy(P::KronInterlace) = P

axes(P::KronInterlace) = (Inclusion(cartesianproduct(map(domain, axes.(P.args, 1))...)), _krontrav_axes(axes.(P.args, 2)...))


normalized(P::KronInterlace) = KronInterlace(map(normalized, P.args))

function getindex(P::RectPolynomial{T}, xy::StaticVector{2}, Jj::BlockIndex{1})::T where T
    a,b = P.args
    J,j = Int(block(Jj)),blockindex(Jj)
    x,y = xy
    a[x,J-j+1]b[y,j]
end
getindex(P::RectPolynomial, xy::StaticVector{2}, B::Block{1}) = [P[xy, B[j]] for j=1:Int(B)]
function getindex(P::RectPolynomial, xy::StaticVector{2}, JR::BlockOneTo)
    A,B = P.args
    x,y = xy
    N = size(JR,1)
    DiagTrav(A[x,OneTo(N)] .* B[y,OneTo(N)]')
end

# reduce other indexing to blocks
_getindex(::Type{Tuple{IND1,IND2}}, P::KronInterlace, (𝐱,J)::Tuple{IND1,Block{1}}) where {IND1,IND2} = P[𝐱, Block.(OneTo(Int(J)))][J]
_getindex(::Type{Tuple{IND1,IND2}}, P::KronInterlace, (𝐱,JR)::Tuple{IND1,BlockRange{1}}) where {IND1,IND2} = P[𝐱, Block.(OneTo(Int(maximum(JR))))][JR]
_getindex(::Type{Tuple{IND1,IND2}}, P::KronInterlace, (𝐱,Jj)::Tuple{IND1,BlockIndex{1}}) where {IND1,IND2} = P[𝐱, block(Jj)][blockindex(Jj)]
_getindex(::Type{Tuple{IND1,IND2}}, P::KronInterlace, (𝐱,j)::Tuple{IND1,IND2}) where {IND1,IND2} = P[𝐱, findblockindex(axes(P,2), j)]
_getindex(::Type{Tuple{IND1,IND2}}, P::KronInterlace, (𝐱,jr)::Tuple{IND1,AbstractArray{IND2}}) where {IND1,IND2} = P[𝐱, Block.(OneTo(Int(findblock(axes(P,2), maximum(jr)))))][jr]

# multiplication by coordinates is only closed for OPs
for (Inc, f, d) in ((:FirstInclusion, :first, 1), (:LastInclusion, :last, 2))
    @eval function layout_broadcasted(::Tuple{BroadcastLayout{typeof($f)},KronOPLayout{2}}, ::typeof(*), x::$Inc, P)
        axes(x,1) == axes(P,1) || throw(DimensionMismatch())
        P*jacobimatrix(Val($d), P)
    end
end

# Actually Jxᵀ
function jacobimatrix(::Val{1}, P::RectPolynomial)
    A,B = P.args
    X = jacobimatrix(A)
    KronTrav(Eye{eltype(X)}(∞), X)
end
# Actually Jyᵀ
function jacobimatrix(::Val{2}, P::RectPolynomial)
    A,B = P.args
    Y = jacobimatrix(B)
    KronTrav(Y, Eye{eltype(Y)}(∞))
end
function diff(P::KronInterlace{N}, order::NTuple{N,Int}; dims...) where N
    diffs = map(diff, P.args, order)
    RectPolynomial(map(basis, diffs)...) * KronTrav(reverse(map(coefficients, diffs))...)
end


function weaklaplacian(P::RectPolynomial)
    A,B = P.args
    Δ_A,Δ_B = weaklaplacian(A), weaklaplacian(B)
    M_A,M_B = grammatrix(A), grammatrix(B)
    KronTrav(M_B,Δ_A) + KronTrav(Δ_B,M_A)
end

function \(P::RectPolynomial, Q::RectPolynomial)
    PA,PB = P.args
    QA,QB = Q.args
    krontrav(PB\QB, PA\QA)
end

@simplify function *(Ac::QuasiAdjoint{<:Any,<:RectPolynomial}, B::RectPolynomial)
    PA,PB = Ac'.args
    QA,QB = B.args
    KronTrav(PB'QB, PA'QA)
end

simplifiable(::typeof(*), ::QuasiAdjoint{T,<:KronInterlace{d}}, ::AbstractQuasiVector) where {d,T} = Val(true)
simplifiable(::typeof(*), ::QuasiAdjoint{T,<:AbstractQuasiVector}, ::KronInterlace{d}) where {d,T} = Val(true)
function mul(Ac::QuasiAdjoint{T,<:KronInterlace{d}}, B::AbstractQuasiVector) where {d,T}
    P = KronInterlace{d, T}(map(_ -> Legendre{eltype(Ac)}(), parent(Ac).args))
    (Ac * P) * (P \ B)
end

mul(B::QuasiAdjoint{T,<:AbstractQuasiVector}, A::KronInterlace{d}) where {d,T} = (A'B')'


grammatrix(P::KronInterlace) = KronTrav(reverse(grammatrix.(P.args))...)
    

basis_axes(d::Inclusion{<:Any,<:ProductDomain}, v) = KronInterlace(map(d -> basis(Inclusion(d)),components(d.domain))...)
basis_axes(d::Inclusion{<:Any,<:DomainSets.FixedIntervalProduct{N,T,D}}, v) where {N,T,D} = KronInterlace(Fill(basis(Inclusion(D())), N))

struct TensorPlan{T, Plans}
    plans::Plans
end

TensorPlan(P...) = TensorPlan{mapreduce(eltype,promote_type,P), typeof(P)}(P)

function *(A::TensorPlan, B::AbstractArray)
    for p in A.plans
        B = p*B
    end
    B
end

function plan_transform(P::KronInterlace{d,<:Any,<:Fill}, (B,)::Tuple{Block{1}}, dims=1:1) where d
    @assert only(dims) == 1

    T = first(P.args)
    @assert d == 2
    ApplyPlan(diagtrav, plan_transform(T, tuple(Fill(Int(B),d)...)))
end

function plan_transform(P::KronInterlace{d,<:Any,<:Fill}, (B,n)::Tuple{Block{1},Int}, dims=1:1) where d
    @assert only(dims) == 1

    T = first(P.args)
    @assert d == 2
    ApplyPlan(A -> diagtrav(A; dims=1:d), plan_transform(T, tuple(Fill(Int(B),d)...,n), 1:d))
end

function grid(P::RectPolynomial, B::Block{1})
    x,y = grid.(P.args, Int(B))
    SVector.(x, y')
end

function plotgrid(P::RectPolynomial, B::Block{1})
    x,y = plotgrid.(P.args, Int(B))
    SVector.(x, y')
end


function plan_transform(P::KronInterlace{d}, B::Tuple{Block{1}}, dims=1:1) where d
    @assert dims == 1 || dims == 1:1 || dims == (1,)
    @assert d == 2
    N = Int(B[1])
    Fx,Fy = plan_transform(P.args[1], (N,N), 1),plan_transform(P.args[2], (N,N), 2)
    ApplyPlan(diagtrav, TensorPlan(Fx,Fy))
end

grid(P::KronInterlace, n::Integer) = grid(P, findblock(axes(P,2), n))
plotgrid(P::KronInterlace, n::Integer) = plotgrid(P, findblock(axes(P,2), n))
plan_transform(P::KronInterlace, Bs::NTuple{N,Int}, dims=ntuple(identity,Val(N))) where N = plan_transform(P, findblock.(Ref(axes(P,2)), Bs), dims)
transform_ldiv(V::SubQuasiArray{<:Any,2,<:KronInterlace,<:Tuple{Inclusion,BlockSlice{BlockOneTo}}}, B::AbstractQuasiArray, _) = factorize(V) \ B

applylayout(::Type{typeof(*)}, ::Lay, ::DiagTravLayout) where Lay <: AbstractBasisLayout = ExpansionLayout{Lay}()
ContinuumArrays._mul_plotgrid(::Tuple{Any,DiagTravLayout{<:PaddedLayout}}, (P,c)::NTuple{2,Any}) = plotgrid(P, maximum(blockcolsupport(c)))

pad(C::DiagTrav, ::BlockedOneTo{Int,RangeCumsum{Int,OneToInf{Int}}}) = DiagTrav(pad(C.array, ∞, ∞))

QuasiArrays.mul(A::BivariateOrthogonalPolynomial, b::DiagTrav) = ApplyQuasiArray(*, A, b)
# make sure block structure matches
QuasiArrays.mul(A::KronInterlace, b::AbstractVector) = ApplyQuasiArray(*, A, BlockedVector(b, (axes(A,2),)))
QuasiArrays.mul(A::KronInterlace, b::DiagTrav) = ApplyQuasiArray(*, A, b)


#########
# evaluation
########

function Base.unsafe_getindex(f::Mul{KronOPLayout{2},<:DiagTravLayout{<:PaddedLayout}}, 𝐱::SVector)
    P,c = f.A, f.B
    A,B = P.args
    x,y = 𝐱
    clenshaw(vec(clenshaw(paddeddata(c.array), recurrencecoefficients(A)..., x; dims=1)), recurrencecoefficients(B)..., y)
end

Base.@propagate_inbounds function getindex(f::Mul{KronOPLayout{2},<:DiagTravLayout{<:PaddedLayout}}, x::SVector, j...)
    @inbounds checkbounds(ApplyQuasiArray(*,f.A,f.B), x, j...)
    Base.unsafe_getindex(f, x, j...)
end

## Special Legendre case

function transform_ldiv(K::KronInterlace{d,V,<:Fill{<:Legendre}}, f::AbstractQuasiVector) where {d,V}
    T = KronInterlace{d}(Fill(ChebyshevT{V}(), size(K.args)...))
    dat = invdiagtrav(T \ f)
    DiagTrav(pad(FastTransforms.th_cheb2leg(paddeddata(dat)), axes(dat)...))
end


## sum

function Base._sum(P::RectPolynomial, dims::Int)
    @assert dims == 1
    KronTrav(reverse(sum.(P.args; dims=1))...)
end

## multiplication

function layout_broadcasted(::Tuple{ExpansionLayout{KronOPLayout{2}},KronOPLayout{2}}, ::typeof(*), a, P)
    axes(a,1) == axes(P,1) || throw(DimensionMismatch())

    A,B = basis(a).args
    T,U = P.args

    C = paddeddata(invdiagtrav(coefficients(a)))
        
    P * ClenshawKron(C, (recurrencecoefficients(A), recurrencecoefficients(B)), (jacobimatrix(T), jacobimatrix(U)))
end


broadcastbasis(::typeof(+), A::KronInterlace, B::KronInterlace) =  KronInterlace(broadcastbasis.(+, A.args, B.args)...)



####
# reshape/vec
####

function reshape_layout(::ExpansionLayout{<:KronInterlaceLayouts{2}}, f)
    A,B = basis(f).args
    A*invdiagtrav(coefficients(f))*B'
end

# TODO: generalise
function vec_layout(::KronExpansionLayout{OPLayout, OPLayout}, f)
    A,C,Bc = arguments(f)
    RectPolynomial(A, Bc') * DiagTrav(C)
end
