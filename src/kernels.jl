const Kernel{T,F,D1,V,D2} = BroadcastQuasiMatrix{T,F,Tuple{D1,QuasiAdjoint{V,Inclusion{V,D2}}}}

"""
    kernel(f)

returns `K::AbstractQuasiMatrix` such that `K[x,y] == f[SVector(x,y)]`.
"""
function kernel(f::AbstractQuasiVector{<:Real})
    P², c = f.args
    P,Q = P².args
    P * c.array * Q'
end

@simplify function *(K::Kernel, Q::OrthogonalPolynomial)
    x,y = axes(K)
    Px,Py = legendre(x),legendre(y)
    kernel(expand(RectPolynomial(Px, Py), splat(K.f))) * Q
end

