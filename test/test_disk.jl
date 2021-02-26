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
    @test Zernike()[SVector(0.1,0.2),1] ≈ inv(sqrt(π))

    Zernike()[SVector(0.1,0.2),Block.(1:5)]

    Zernike()[SVector(0.1,0.2),Block(2)]
    Zernike()[SVector(0.1,0.2),Block(3)]
end