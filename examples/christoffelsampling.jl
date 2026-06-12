using ClassicalOrthogonalPolynomials, MultivariateOrthogonalPolynomials, StatsBase, Plots


function christoffel(A)
    Q,R = qr(A)
    n = size(A,2)
    sum(expand(Q[:,k] .^2) for k=1:n)/n
end

x,y = coordinates(ChebyshevInterval() ^ 2)
n = 3
A = hcat([@.(cos(k*x)cos(j*y)) for k=0:n, j=0:n]...)
K = christoffel(A)
