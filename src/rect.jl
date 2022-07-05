struct KronPolynomial{d, T, PP} <: MultivariateOrthogonalPolynomial{d, T}
    args::PP
end

KronPolynomial{d,T}(a::Vararg{Any,d}) where {d,T} = KronPolynomial{d,T,typeof(a)}(a)
KronPolynomial{d}(a::Vararg{Any,d}) where d = KronPolynomial{d,mapreduce(eltype, promote_type, a)}(a...)
KronPolynomial(a::Vararg{Any,d}) where d = KronPolynomial{d}(a...)

const RectPolynomial{T, PP} = KronPolynomial{2, T, PP}


axes(P::KronPolynomial) = (Inclusion(×(map(domain, axes.(P.args, 1))...)), _krontrav_axes(tuple.(axes.(P.args, 2))...)...)
function getindex(P::RectPolynomial{T}, xy::StaticVector{2}, Jj::BlockIndex{1})::T where T
    a,b = P.args
    J,j = Int(block(Jj)),blockindex(Jj)
    x,y = xy
    a[x,J-j+1]b[y,j]
end
getindex(P::RectPolynomial, xy::StaticVector{2}, B::Block{1}) = [P[xy, B[j]] for j=1:Int(B)]
getindex(P::RectPolynomial, xy::StaticVector{2}, JR::BlockOneTo) = mortar([P[xy,Block(J)] for J = 1:Int(JR[end])])

@simplify function *(Dx::PartialDerivative{1}, P::RectPolynomial)
    A,B = P.args
    U,M = (Derivative(axes(A,1))*A).args
    RectPolynomial(U,B) * KronTrav(M, Eye{eltype(M)}(∞))
end

@simplify function *(Dx::PartialDerivative{2}, P::RectPolynomial)
    A,B = P.args
    U,M = (Derivative(axes(B,1))*B).args
    RectPolynomial(A,U) * KronTrav(Eye{eltype(M)}(∞), M)
end