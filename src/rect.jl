"""
    KronPolynomial(A, B, C...)

represents the kronecker product of `A`, `B`, …. In particular, if `K = KronPolynomial(A,B)` and `U` is an infinite
matrix of coefficients we have
``
K[SVector(x,y),:]'DiagTrav(U) == A[x,:]'U*B[y,:]
``
"""

struct KronPolynomial{d, T, PP} <: MultivariateOrthogonalPolynomial{d, T}
    args::PP
end

KronPolynomial{d,T}(a::Vararg{Any,d}) where {d,T} = KronPolynomial{d,T,typeof(a)}(a)
KronPolynomial{d}(a::Vararg{Any,d}) where d = KronPolynomial{d,mapreduce(eltype, promote_type, a)}(a...)
KronPolynomial(a::Vararg{Any,d}) where d = KronPolynomial{d}(a...)
KronPolynomial{d,T}(a::AbstractVector) where {d,T} = KronPolynomial{d,T,typeof(a)}(a)
KronPolynomial{d}(a::AbstractVector) where d = KronPolynomial{d,eltype(eltype(a))}(a)
KronPolynomial(a::AbstractVector) = KronPolynomial{length(a)}(a)

function show(io::IO, P::KronPolynomial)
    for k = 1:length(P.args)
        print(io, "$(P.args[k])")
        k ≠ length(P.args) && print(io, " ⊗ ")
    end
end

==(A::KronPolynomial, B::KronPolynomial) = length(A.args) == length(B.args) && all(map(==, A.args, B.args))


struct KronOPLayout{d} <: AbstractMultivariateOPLayout{d} end
MemoryLayout(::Type{<:KronPolynomial{d}}) where d = KronOPLayout{d}()

const RectPolynomial{T, PP} = KronPolynomial{2, T, PP}



axes(P::KronPolynomial) = (Inclusion(×(map(domain, axes.(P.args, 1))...)), _krontrav_axes(axes.(P.args, 2)...))
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
function diff(P::KronPolynomial{N}, order::NTuple{N,Int}; dims...) where N
    diffs = diff.(P.args, order)
    RectPolynomial(basis.(diffs)...) * KronTrav(coefficients.(diffs)...)
end


function weaklaplacian(P::RectPolynomial)
    A,B = P.args
    Δ_A,Δ_B = weaklaplacian(A), weaklaplacian(B)
    M_A,M_B = grammatrix(A), grammatrix(B)
    KronTrav(Δ_A,M_B) + KronTrav(M_A,Δ_B)
end

function \(P::RectPolynomial, Q::RectPolynomial)
    PA,PB = P.args
    QA,QB = Q.args
    krontrav(PA\QA, PB\QB)
end

@simplify function *(Ac::QuasiAdjoint{<:Any,<:RectPolynomial}, B::RectPolynomial)
    PA,PB = Ac'.args
    QA,QB = B.args
    KronTrav(PA'QA, PB'QB)
end


struct ApplyPlan{T, F, Pl}
    f::F
    plan::Pl
end

ApplyPlan(f, P) = ApplyPlan{eltype(P), typeof(f), typeof(P)}(f, P)

*(A::ApplyPlan, B::AbstractArray) = A.f(A.plan*B)

basis_axes(d::Inclusion{<:Any,<:ProductDomain}, v) = KronPolynomial(map(d -> basis(Inclusion(d)),components(d.domain))...)

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

function checkpoints(P::RectPolynomial)
    x,y = checkpoints.(P.args)
    SVector.(x, y')
end

function plan_transform(P::KronPolynomial{d,<:Any,<:Fill}, B::Tuple{Block{1}}, dims=1:1) where d
    @assert dims == 1

    T = first(P.args)
    @assert d == 2
    ApplyPlan(DiagTrav, plan_transform(T, tuple(Fill(Int(B[1]),d)...)))
end

function grid(P::RectPolynomial, B::Block{1})
    x,y = grid.(P.args, Int(B))
    SVector.(x, y')
end

function plotgrid(P::RectPolynomial, B::Block{1})
    x,y = plotgrid.(P.args, Int(B))
    SVector.(x, y')
end


function plan_transform(P::KronPolynomial{d}, B::Tuple{Block{1}}, dims=1:1) where d
    @assert dims == 1 || dims == 1:1 || dims == (1,)
    @assert d == 2
    N = Int(B[1])
    Fx,Fy = plan_transform(P.args[1], (N,N), 1),plan_transform(P.args[2], (N,N), 2)
    ApplyPlan(DiagTrav, TensorPlan(Fx,Fy))
end

applylayout(::Type{typeof(*)}, ::Lay, ::DiagTravLayout) where Lay <: AbstractBasisLayout = ExpansionLayout{Lay}()
ContinuumArrays._mul_plotgrid(::Tuple{Any,DiagTravLayout{<:PaddedLayout}}, (P,c)::NTuple{2,Any}) = plotgrid(P, maximum(blockcolsupport(c)))

pad(C::DiagTrav, ::BlockedOneTo{Int,RangeCumsum{Int,OneToInf{Int}}}) = DiagTrav(pad(C.array, ∞, ∞))

QuasiArrays.mul(A::BivariateOrthogonalPolynomial, b::DiagTrav) = ApplyQuasiArray(*, A, b)

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

function transform_ldiv(K::KronPolynomial{d,V,<:Fill{<:Legendre}}, f::Union{AbstractQuasiVector,AbstractQuasiMatrix}) where {d,V}
    T = KronPolynomial{d}(Fill(ChebyshevT{V}(), size(K.args)...))
    dat = (T \ f).array
    DiagTrav(pad(FastTransforms.th_cheb2leg(paddeddata(dat)), axes(dat)...))
end


## sum

function Base._sum(P::RectPolynomial, dims)
    @assert dims == 1
    KronTrav(sum.(P.args; dims=1)...)
end

## multiplication

function layout_broadcasted(::Tuple{ExpansionLayout{KronOPLayout{2}},KronOPLayout{2}}, ::typeof(*), a, P)
    axes(a,1) == axes(P,1) || throw(DimensionMismatch())

    A,B = basis(a).args
    T,U = P.args

    C = paddeddata(invdiagtrav(coefficients(a)))
        
    P * ClenshawKron(C, (recurrencecoefficients(A), recurrencecoefficients(B)), (jacobimatrix(T), jacobimatrix(U)))
end


broadcastbasis(::typeof(+), A::KronPolynomial, B::KronPolynomial) =  KronPolynomial(broadcastbasis.(+, A.args, B.args)...)