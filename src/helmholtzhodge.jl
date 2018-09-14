include("gradient.jl")

using BandedMatrices

import BandedMatrices: BandedQ

# Store QR factorizations required to apply the Helmholtz-Hodge decomposition.
struct HelmholtzHodge{T}
    Q::Vector{BandedQ{T}}
    R::Vector{BandedMatrix{T, Matrix{T}}}
    X::Vector{T}
end

function HelmholtzHodge(::Type{T}, N::Int) where T
    Q = Vector{BandedQ{T}}(N)
    R = Vector{BandedMatrix{T, Matrix{T}}}(N)
    for m = 1:N
        Q[m], R[m] = qr(helmholtzhodgeconversion(T, N, m))
    end
    HelmholtzHodge(Q, R, zeros(T, 2N+2))
end

function helmholtzhodgeconversion(::Type{T}, N::Int, m::Int) where T
    A = BandedMatrix(Zeros{T}(2N+4-2m, 2N+2-2m), (2, 2))
    for ℓ = 1:N+1-m
        A[2ℓ, 2ℓ-1] = m
        A[2ℓ-1, 2ℓ] = m
        cst = (m+ℓ-1)*sqrt(T(ℓ*(ℓ+2m))/T((2ℓ+2m-1)*(2ℓ+2m+1)))
        A[2ℓ+2, 2ℓ] = cst
        A[2ℓ+1, 2ℓ-1] = cst
    end
    for ℓ = 1:N-m
        cst = -(m+ℓ+1)*sqrt(T(ℓ*(ℓ+2m))/T((2ℓ+2m-1)*(2ℓ+2m+1)))
        A[2ℓ-1, 2ℓ+1] = cst
        A[2ℓ, 2ℓ+2] = cst
    end
    A
end

# This function works in-place on the data stored in the factorization.
function solvehelmholtzhodge!(HH::HelmholtzHodge{T}, m::Int) where T
    Q = HH.Q[m]
    R = HH.R[m]
    X = HH.X

    # Step 1: apply Q'

    H=Q.H
    m=Q.m

    M=size(H,1)
    x=pointer(X)
    h=pointer(H)
    st=stride(H,2)
    sz=sizeof(T)

    for k=1:min(size(H,2),m-M+1)
        wp=h+sz*st*(k-1)
        xp=x+sz*(k-1)

        dt=BandedMatrices.dot(M,wp,1,xp,1)
        BandedMatrices.axpy!(M,-2*dt,wp,1,xp,1)
    end

    for k=m-M+2:size(H,2)
        p=k-m+M-1

        wp=h+sz*st*(k-1)
        xp=x+sz*(k-1)

        dt=BandedMatrices.dot(M-p,wp,1,xp,1)
        BandedMatrices.axpy!(M-p,-2*dt,wp,1,xp,1)
    end

    # Step 2: backsolve with (square) R

    BandedMatrices.tbsv!('U', 'N', 'N', size(R.data, 2), R.u, pointer(R.data), size(R.data, 1), pointer(X), 1)

    X
end


function helmholtzhodge!(HH::HelmholtzHodge{T}, U1, U2, V1, V2) where T
	N, M = size(V1)

	# U1 is for e_theta and U2 is for e_phi.
    # The first columns are easy.
    U1[1, 1] = 0
    U2[1, 1] = 0
    @inbounds @simd for ℓ = 1:N-1
        U1[ℓ+1, 1] = -V1[ℓ, 1]/sqrt(T(ℓ*(ℓ+1)))
        U2[ℓ+1, 1] = -V2[ℓ, 1]/sqrt(T(ℓ*(ℓ+1)))
    end

    # First, we multiply by sin(θ), which can be done by incrementing the order P_ℓ^{m-1} ↗ P_ℓ^m.
	@inbounds for m = 1:M÷2
        @simd for ℓ = 1:N-1-m
            aℓ = sqrt(T((ℓ+2m-2)*(ℓ+2m-1))/T((2ℓ+2m-3)*(2ℓ+2m-1)))
            bℓ = -sqrt(T(ℓ*(ℓ+1))/T((2ℓ+2m-1)*(2ℓ+2m+1)))
            V1[ℓ, 2m] = aℓ*V1[ℓ, 2m] + bℓ*V1[ℓ+2, 2m]
            V1[ℓ, 2m+1] = aℓ*V1[ℓ, 2m+1] + bℓ*V1[ℓ+2, 2m+1]
			V2[ℓ, 2m] = aℓ*V2[ℓ, 2m] + bℓ*V2[ℓ+2, 2m]
            V2[ℓ, 2m+1] = aℓ*V2[ℓ, 2m+1] + bℓ*V2[ℓ+2, 2m+1]
        end
        ℓ = N-m
        if ℓ > 0
            aℓ = sqrt(T((ℓ+2m-2)*(ℓ+2m-1))/T((2ℓ+2m-3)*(2ℓ+2m-1)))
            V1[ℓ, 2m] = aℓ*V1[ℓ, 2m]
            V1[ℓ, 2m+1] = aℓ*V1[ℓ, 2m+1]
    		V2[ℓ, 2m] = aℓ*V2[ℓ, 2m]
            V2[ℓ, 2m+1] = aℓ*V2[ℓ, 2m+1]
        end
		ℓ = N+1-m
        aℓ = sqrt(T((ℓ+2m-2)*(ℓ+2m-1))/T((2ℓ+2m-3)*(2ℓ+2m-1)))
        V1[ℓ, 2m] = aℓ*V1[ℓ, 2m]
        V1[ℓ, 2m+1] = aℓ*V1[ℓ, 2m+1]
		V2[ℓ, 2m] = aℓ*V2[ℓ, 2m]
        V2[ℓ, 2m+1] = aℓ*V2[ℓ, 2m+1]
    end

    # Next, we solve the banded linear systems.
    for m = 1:M÷2
        readin1!(HH.X, V1, V2, N, m)
        solvehelmholtzhodge!(HH, m)
        writeout1!(HH.X, U1, U2, N, m)
        readin2!(HH.X, V1, V2, N, m)
        solvehelmholtzhodge!(HH, m)
        writeout2!(HH.X, U1, U2, N, m)
    end

    U1
end

function readin1!(X, V1, V2, N, m)
	X[1] = V1[1, 2m]
	X[2N+2-2m] = V2[N+1-m, 2m+1]
	@inbounds for ℓ = 1:N-m
		X[2ℓ] = V2[ℓ, 2m+1]
		X[2ℓ+1] = V1[ℓ+1, 2m]
	end
    X[2N+3-2m] = X[2N+4-2m] = 0
end

function readin2!(X, V1, V2, N, m)
	X[1] = V1[1, 2m+1]
	X[2N+2-2m] = -V2[N+1-m, 2m]
	@inbounds for ℓ = 1:N-m
		X[2ℓ] = -V2[ℓ, 2m]
		X[2ℓ+1] = V1[ℓ+1, 2m+1]
	end
    X[2N+3-2m] = X[2N+4-2m] = 0
end

function writeout1!(X, U1, U2, N, m)
	U1[1, 2m] = X[1]
    U2[1, 2m+1] = X[N+1-m]
	@inbounds for ℓ = 1:N-1-m
		U1[ℓ+1, 2m] = X[2ℓ+1]
		U2[ℓ, 2m+1] = X[2ℓ]
	end
    U2[N-m, 2m+1] = X[2N-2m]
end

function writeout2!(X, U1, U2, N, m)
	U1[1, 2m+1] = X[1]
    U2[1, 2m] = X[N+1-m]
	@inbounds for ℓ = 1:N-1-m
		U1[ℓ+1, 2m+1] = X[2ℓ+1]
		U2[ℓ, 2m] = -X[2ℓ]
	end
    U2[N-m, 2m] = -X[2N-2m]
end
