using MultivariateOrthogonalPolynomials, StaticArrays, BlockArrays, Test


function chebydiskeval(c::AbstractMatrix{T}, r, θ) where T
    ret = zero(T)
    for j = 1:2:size(c,2), k=1:size(c,1)
        m,ℓ = j ÷ 2, k-1
        ret += c[k,j] * cos(m*θ) * cos((2ℓ+isodd(m))*acos(r))
    end
    for j = 2:2:size(c,2), k=1:size(c,1)
        m = j ÷ 2; ℓ = k-1
        ret += c[k,j] * sin(m*θ) * cos((2ℓ+isodd(m))*acos(r))
    end
    ret
end

@testset "Disk" begin
    r,θ = 0.1, 0.2
    rθ = RadialCoordinate(r,θ)
    xy = SVector(rθ)
    @test Zernike()[rθ,1] ≈ Zernike()[xy,1] ≈ inv(sqrt(π))
    @test Zernike()[rθ,Block(1)] ≈ Zernike()[xy,Block(1)] ≈ [inv(sqrt(π))]
    @test Zernike()[rθ,Block(2)] ≈ [2r/π*sin(θ), 2r/π*cos(θ)]
    @test Zernike()[rθ,Block(3)] ≈ [sqrt(3/π)*(2r^2-1),sqrt(6)/π*r^2*sin(2θ),sqrt(6)/π*r^2*cos(2θ)]
end