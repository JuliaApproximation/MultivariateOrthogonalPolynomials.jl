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
@simplify function *(Dx::PartialDerivative{1}, P::RectPolynomial)
    A,B = P.args
    U,M = (Derivative(axes(A,1))*A).args
    # We want I ⊗ D² as A ⊗ B means B * X * A'
    RectPolynomial(U,B) * KronTrav(Eye{eltype(M)}(∞), M)
end

@simplify function *(Dx::PartialDerivative{2}, P::RectPolynomial)
    A,B = P.args
    U,M = (Derivative(axes(B,1))*B).args
    RectPolynomial(A,U) * KronTrav(M, Eye{eltype(M)}(∞))
end

function \(P::RectPolynomial, Q::RectPolynomial)
    PA,PB = P.args
    QA,QB = Q.args
    KronTrav(PA\QA, PB\QB)
end

struct ApplyPlan{T, F, Pl}
    f::F
    plan::Pl
end

ApplyPlan(f, P) = ApplyPlan{eltype(P), typeof(f), typeof(P)}(f, P)

*(A::ApplyPlan, B::AbstractArray) = A.f(A.plan*B)

basis_axes(d::Inclusion{<:Any,<:ProductDomain}, v) = KronPolynomial(map(d -> basis(Inclusion(d)),components(d.domain))...)

function checkpoints(P::RectPolynomial)
    x,y = checkpoints.(P.args)
    SVector.(x, y')
end

function plan_grid_transform(P::KronPolynomial{d,<:Any,<:Fill}, B::Tuple{Block{1}}, dims=1:1) where d
    @assert dims == 1

    T = first(P.args)
    x, F = plan_grid_transform(T, Array{eltype(P)}(undef, Fill(Int(B[1]),d)...))
    @assert d == 2
    x̃ = Vector(x)
    SVector.(x̃, x̃'), ApplyPlan(DiagTrav, F)
end

function plotgrid(P::RectPolynomial, B::Block{1})
    x,y = plotgrid.(P.args, Int(B))
    SVector.(x, y')
end


applylayout(::Type{typeof(*)}, ::Lay, ::DiagTravLayout) where Lay <: AbstractBasisLayout = ExpansionLayout{Lay}()
ContinuumArrays._mul_plotgrid(::Tuple{Any,DiagTravLayout{<:PaddedLayout}}, (P,c)) = plotgrid(P, maximum(blockcolsupport(c)))

pad(C::DiagTrav, ::BlockedUnitRange{RangeCumsum{Int,OneToInf{Int}}}) = DiagTrav(pad(C.array, ∞, ∞))

QuasiArrays.mul(A::BivariateOrthogonalPolynomial, b::DiagTrav) = ApplyQuasiArray(*, A, b)

function Base.unsafe_getindex(f::Mul{MultivariateOPLayout{2},<:DiagTravLayout{<:PaddedLayout}}, 𝐱::SVector)
    P,c = f.A, f.B
    A,B = P.args
    x,y = 𝐱
    clenshaw(clenshaw(paddeddata(c.array), recurrencecoefficients(A)..., x), recurrencecoefficients(B)..., y)
end

Base.@propagate_inbounds function getindex(f::Mul{MultivariateOPLayout{2},<:DiagTravLayout{<:PaddedLayout}}, x::SVector, j...)
    @inbounds checkbounds(ApplyQuasiArray(*,f.A,f.B), x, j...)
    Base.unsafe_getindex(f, x, j...)
end

## Special Legendre case

function transform_ldiv(K::KronPolynomial{d,V,<:Fill{<:Legendre}}, f::Union{AbstractQuasiVector,AbstractQuasiMatrix}) where {d,V}
    T = KronPolynomial{d}(Fill(ChebyshevT{V}(), size(K.args)...))
    dat = (T \ f).array
    DiagTrav(pad(FastTransforms.th_cheb2leg(paddeddata(dat)), axes(dat)...))
end