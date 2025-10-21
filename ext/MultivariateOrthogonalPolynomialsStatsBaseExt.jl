module MultivariateOrthogonalPolynomialsStatsBaseExt
using MultivariateOrthogonalPolynomials, StatsBase
using MultivariateOrthogonalPolynomials.StaticArrays
using MultivariateOrthogonalPolynomials.QuasiArrays
using MultivariateOrthogonalPolynomials: ExpansionLayout, KronOPLayout
import MultivariateOrthogonalPolynomials.QuasiArrays: sample_layout

function sample_layout(::ExpansionLayout{KronOPLayout{2}}, f::AbstractQuasiVector, n::Integer)
    F = reshape(f)
    x = sample(sum(F; dims=2), n)
    # x = sample(F[:,y]) # TODO: this should work
    y = [sample(F[x,:]) for x in x]
    map(SVector, x, y)
end

sample_layout(lay::ExpansionLayout{KronOPLayout{2}}, f::AbstractQuasiVector) = only(sample_layout(lay, f, 1))

end