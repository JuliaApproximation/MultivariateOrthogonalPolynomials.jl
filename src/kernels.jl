const Kernel{T,F,D1,V,D2} = BroadcastQuasiMatrix{T,F,Tuple{D1,QuasiAdjoint{V,Inclusion{V,D2}}}}

@simplify function *(K::Kernel, Q::OrthogonalPolynomial)
    x,y = axes(K)
    Px,Py = legendre(x),legendre(y)
    Px * transform(RectPolynomial(Px, Py), splat(K.f)).array * (Py'Q)
end