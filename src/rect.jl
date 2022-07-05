struct KronPolynomial{T, N, PP} <: MultivariateOrthogonalPolynomial{T, N}
    args::PP
end

KronPolynomial{T,N}(a::Vararg{Any,N}) where {T,N} = KronPolynomial{T,N,typeof(a)}(a)
KronPolynomial{T}(a::Vararg{Any,N}) where {T,N} = KronPolynomial{T,N}(a...)
KronPolynomial(a...) = KronPolynomial{mapreduce(eltype, promote_type, a)}(a...)

axes(P::KronPolynomial) = (Inclusion(×(map(domain, axes.(P.args, 1))...)), _krontrav_axes(axes.(P.args, 2)...))
function getindex(P::KronPolynomial{<:Any,2}, xy::StaticVector{2}, Jj::Block{1})
    a,b = P.args
    J,j = Int(block(Jj)),blockindex(Jj)
    x,y = xy
    a[x,J-j+1]b[y,j]
end