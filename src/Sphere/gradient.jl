# This file calculates the surface gradient of a scalar field.

function gradient!(U::Matrix{T}, ∇θU::Matrix{T}, ∇φU::Matrix{T}) where T
    @assert size(U) == size(∇θU) == size(∇φU)
    N, M = size(U)

    # The first column is easy.
    @inbounds @simd for ℓ = 1:N-1
        ∇θU[ℓ, 1] = -sqrt(T(ℓ*(ℓ+1)))*U[ℓ+1, 1]
        ∇φU[ℓ, 1] = 0
    end
    ∇θU[N, 1] = 0
    ∇φU[N, 1] = 0

    # Next, we differentiate with respect to φ, which preserves the order. It swaps sines and cosines in longitude, though.
    @inbounds for m = 1:M÷2
        @simd for ℓ = 1:N+1-m
            ∇φU[ℓ, 2m] = -m*U[ℓ, 2m+1]
            ∇φU[ℓ, 2m+1] = m*U[ℓ, 2m]
        end
    end

    # Then, we differentiate with respect to θ, which preserves the order but divides by sin(θ).

    @inbounds for m = 1:M÷2
        ℓ = 1
        bℓ = -(ℓ+m+1)*sqrt(T(ℓ*(ℓ+2m))/T((2ℓ+2m-1)*(2ℓ+2m+1)))
        ∇θU[ℓ, 2m] = bℓ*U[ℓ+1, 2m]
        ∇θU[ℓ, 2m+1] = bℓ*U[ℓ+1, 2m+1]
        @simd for ℓ = 2:N-m
            aℓ = (ℓ+m-2)*sqrt(T((ℓ-1)*(ℓ+2m-1))/T((2ℓ+2m-3)*(2ℓ+2m-1)))
            bℓ = -(ℓ+m+1)*sqrt(T(ℓ*(ℓ+2m))/T((2ℓ+2m-1)*(2ℓ+2m+1)))
            ∇θU[ℓ, 2m] = aℓ*U[ℓ-1, 2m] + bℓ*U[ℓ+1, 2m]
            ∇θU[ℓ, 2m+1] = aℓ*U[ℓ-1, 2m+1] + bℓ*U[ℓ+1, 2m+1]
        end
        ℓ = N-m+1
        aℓ = (ℓ+m-2)*sqrt(T((ℓ-1)*(ℓ+2m-1))/T((2ℓ+2m-3)*(2ℓ+2m-1)))
        ∇θU[ℓ, 2m] = aℓ*U[ℓ-1, 2m]
        ∇θU[ℓ, 2m+1] = aℓ*U[ℓ-1, 2m+1]
    end

    # Finally, we divide by sin(θ), which can be done by decrementing the order P_ℓ^m ↘ P_ℓ^{m-1}.
    @inbounds for m = 1:M÷2
        ℓ = N+1-m
        aℓ = sqrt(T((ℓ+2m-2)*(ℓ+2m-1))/T((2ℓ+2m-3)*(2ℓ+2m-1)))
        ∇θU[ℓ, 2m] = ∇θU[ℓ, 2m]/aℓ
        ∇θU[ℓ, 2m+1] = ∇θU[ℓ, 2m+1]/aℓ
        ∇φU[ℓ, 2m] = ∇φU[ℓ, 2m]/aℓ
        ∇φU[ℓ, 2m+1] = ∇φU[ℓ, 2m+1]/aℓ
        ℓ = N+1-m-1
        if ℓ > 0
            aℓ = sqrt(T((ℓ+2m-2)*(ℓ+2m-1))/T((2ℓ+2m-3)*(2ℓ+2m-1)))
            ∇θU[ℓ, 2m] = ∇θU[ℓ, 2m]/aℓ
            ∇θU[ℓ, 2m+1] = ∇θU[ℓ, 2m+1]/aℓ
            ∇φU[ℓ, 2m] = ∇φU[ℓ, 2m]/aℓ
            ∇φU[ℓ, 2m+1] = ∇φU[ℓ, 2m+1]/aℓ
        end
        @simd for ℓ = N+1-m-2:-1:1
            aℓ = sqrt(T((ℓ+2m-2)*(ℓ+2m-1))/T((2ℓ+2m-3)*(2ℓ+2m-1)))
            bℓ = -sqrt(T(ℓ*(ℓ+1))/T((2ℓ+2m-1)*(2ℓ+2m+1)))
            ∇θU[ℓ, 2m] = (∇θU[ℓ, 2m] - bℓ*∇θU[ℓ+2, 2m])/aℓ
            ∇θU[ℓ, 2m+1] = (∇θU[ℓ, 2m+1] - bℓ*∇θU[ℓ+2, 2m+1])/aℓ
            ∇φU[ℓ, 2m] = (∇φU[ℓ, 2m] - bℓ*∇φU[ℓ+2, 2m])/aℓ
            ∇φU[ℓ, 2m+1] = (∇φU[ℓ, 2m+1] - bℓ*∇φU[ℓ+2, 2m+1])/aℓ
        end
    end

    ∇θU
end

function curl!(U::Matrix, U1::Matrix, U2::Matrix)
    gradient!(U, U2, U1)
    N, M = size(U)
    @inbounds for j = 1:M
        for i = 1:N
            U1[i,j] = -U1[i,j]
        end
    end
    U1
end


function partial_gradient!(U::Matrix{T}, ∇θU::Matrix{T}, ∇φU::Matrix{T}) where T
    @assert size(U) == size(∇θU) == size(∇φU)
    N, M = size(U)

    # The first column is easy.
    @inbounds @simd for ℓ = 1:N-1
        ∇θU[ℓ, 1] = -sqrt(T(ℓ*(ℓ+1)))*U[ℓ+1, 1]
        ∇φU[ℓ, 1] = 0
    end
    ∇θU[N, 1] = 0
    ∇φU[N, 1] = 0

    # Next, we differentiate with respect to φ, which preserves the order. It swaps sines and cosines in longitude, though.
    @inbounds for m = 1:M÷2
        @simd for ℓ = 1:N+1-m
            ∇φU[ℓ, 2m] = -m*U[ℓ, 2m+1]
            ∇φU[ℓ, 2m+1] = m*U[ℓ, 2m]
        end
    end

    # Then, we differentiate with respect to θ, which preserves the order but divides by sin(θ).

    @inbounds for m = 1:M÷2
        ℓ = 1
        bℓ = -(ℓ+m+1)*sqrt(T(ℓ*(ℓ+2m))/T((2ℓ+2m-1)*(2ℓ+2m+1)))
        ∇θU[ℓ, 2m] = bℓ*U[ℓ+1, 2m]
        ∇θU[ℓ, 2m+1] = bℓ*U[ℓ+1, 2m+1]
        @simd for ℓ = 2:N-m
            aℓ = (ℓ+m-2)*sqrt(T((ℓ-1)*(ℓ+2m-1))/T((2ℓ+2m-3)*(2ℓ+2m-1)))
            bℓ = -(ℓ+m+1)*sqrt(T(ℓ*(ℓ+2m))/T((2ℓ+2m-1)*(2ℓ+2m+1)))
            ∇θU[ℓ, 2m] = aℓ*U[ℓ-1, 2m] + bℓ*U[ℓ+1, 2m]
            ∇θU[ℓ, 2m+1] = aℓ*U[ℓ-1, 2m+1] + bℓ*U[ℓ+1, 2m+1]
        end
        ℓ = N-m+1
        aℓ = (ℓ+m-2)*sqrt(T((ℓ-1)*(ℓ+2m-1))/T((2ℓ+2m-3)*(2ℓ+2m-1)))
        ∇θU[ℓ, 2m] = aℓ*U[ℓ-1, 2m]
        ∇θU[ℓ, 2m+1] = aℓ*U[ℓ-1, 2m+1]
    end

    ∇θU
end
