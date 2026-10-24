module MultivariateOrthogonalPolynomialsRecipesBaseExt
using MultivariateOrthogonalPolynomials, RecipesBase
using MultivariateOrthogonalPolynomials: rectplotgridvalues
using MultivariateOrthogonalPolynomials.QuasiArrays: ApplyQuasiVector
using MultivariateOrthogonalPolynomials.ContinuumArrays: plotgridvalues, _split_svec

# contour plots and heatmaps require a rectilinear grid
@recipe function f(u::ApplyQuasiVector{<:Any,typeof(*),<:Tuple{JacobiTriangle,AbstractVector}})
    if get(plotattributes, :seriestype, nothing) in (:contour, :contourf, :heatmap)
        rectplotgridvalues(u)
    else
        x,v = plotgridvalues(u)
        tuple(_split_svec(x)..., v)
    end
end

end
