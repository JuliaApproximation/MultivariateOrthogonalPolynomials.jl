using MultivariateOrthogonalPolynomials, FastTransforms, BlockBandedMatrices, Plots
plotly()

Z = Zernike(1)
W = Weighted(Z)
xy = axes(Z,1); x,y = first.(xy),last.(xy)
Δ = Z \ (Laplacian(xy) * W)
S = Z \ W

KR = Block.(Base.OneTo(5)); @time S.args[1][KR,KR];
@time aB = S.args[1].ops[1]; kr = 1:500; 
@time aB[kr,kr];
bc = Base.broadcasted(view(aB,kr,kr))
@time copy(bc)
@ent aB.args[1][kr,kr] ./ aB.args[2]
aB.f
@time aB[kr,kr]

AssertionErro
@time aB.args[1][kr];
@time aB[kr,kr];
@
using ArrayLayouts
c = aB[kr,kr]

@time copyto!(c, bc);
@time aB[kr,kr];
@time aB.args[2][kr,kr];
k = 2
f = @.(cos(x*exp(y)))
F = factorize(Δ + k^2 * S)
c = Z \ f
F \ c

u = W * ((Δ + k^2 * S) \ (Z \ f))



