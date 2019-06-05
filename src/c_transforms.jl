const libfasttransforms = homedir() * "/Projects/FastTransforms/libfasttransforms"

struct ft_plan_struct end
const PlanPtr = Ptr{ft_plan_struct}

function c_plan_tri2cheb end
function c_tri2cheb end
function c_cheb2tri end
function c_plan_disk2cxf end
function c_disk2cxf end
function c_cxf2disk end

if Libdl.find_library(libfasttransforms) ≡ libfasttransforms
    ft_set_threads(n::Int) = ccall((:omp_set_num_threads, libfasttransforms), Nothing, (Int, ), n)
    ft_set_threads(Sys.CPU_THREADS)

    c_plan_sph2fourier(n::Int) = ccall((:ft_plan_sph2fourier, libfasttransforms), PlanPtr, (Int, ), n)
    fc_sph2fourier(P::PlanPtr, A::Matrix{Float64}) = ccall((:ft_execute_sph2fourier, libfasttransforms), Nothing, (PlanPtr, Ref{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))

    c_plan_rottriangle(n::Int, α::Float64, β::Float64, γ::Float64) = ccall((:ft_plan_rottriangle, libfasttransforms), PlanPtr, (Int, Float64, Float64, Float64), n, α, β, γ)
    c_execute_tri_hi2lo(P::PlanPtr, A::Matrix{Float64}) = ccall((:ft_execute_tri_hi2lo, libfasttransforms), Nothing, (PlanPtr, Ref{Float64}, Int), P, A, size(A, 2))
    c_execute_tri_lo2hi(P::PlanPtr, A::Matrix{Float64}) = ccall((:ft_execute_tri_lo2hi, libfasttransforms), Nothing, (PlanPtr, Ref{Float64}, Int), P, A, size(A, 2))

    c_plan_tri2cheb(n::Int, α::Float64, β::Float64, γ::Float64) = ccall((:ft_plan_tri2cheb, libfasttransforms), PlanPtr, (Int, Float64, Float64, Float64), n, α, β, γ)
    c_tri2cheb(P::PlanPtr, A::Matrix{Float64}) = ccall((:ft_execute_tri2cheb, libfasttransforms), Nothing, (PlanPtr, Ref{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
    c_cheb2tri(P::PlanPtr, A::Matrix{Float64}) = ccall((:ft_execute_cheb2tri, libfasttransforms), Nothing, (PlanPtr, Ref{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))

    c_plan_disk2cxf(n::Int) = ccall((:ft_plan_disk2cxf, libfasttransforms), PlanPtr, (Int,), n)
    c_disk2cxf(P::PlanPtr, A::Matrix{Float64}) = ccall((:ft_execute_disk2cxf, libfasttransforms), Nothing, (PlanPtr, Ref{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
    c_cxf2disk(P::PlanPtr, A::Matrix{Float64}) = ccall((:ft_execute_cxf2disk, libfasttransforms), Nothing, (PlanPtr, Ref{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
else
    @warn "Cannot load FastTransforms Dylib"
end

struct CTri2ChebPlan
    plan::PlanPtr
    n::Int
    α::Float64
    β::Float64
    γ::Float64
end

CTri2ChebPlan(n::Int, α::Float64, β::Float64, γ::Float64) = 
    CTri2ChebPlan(c_plan_tri2cheb(n, α, β, γ), n, α, β, γ)

function *(C::CTri2ChebPlan, A::Matrix{Float64})
    size(A,1) == size(A,2) == C.n || throw(ArgumentError(A))
    B = copy(A)
    c_tri2cheb(C.plan, B)
    B
end

function \(C::CTri2ChebPlan, A::Matrix{Float64})
    size(A,1) == size(A,2) == C.n || throw(ArgumentError(A))
    B = copy(A)
    c_cheb2tri(C.plan, B)
    B
end


struct CDisk2CxfPlan
    plan::PlanPtr
    n::Int
end

CDisk2CxfPlan(n::Int) = CDisk2CxfPlan(c_plan_disk2cxf(n), n)

function *(C::CDisk2CxfPlan, A::Matrix{Float64})
    (size(A,1) == C.n && size(A,2) == 4C.n-3) || throw(DimensionMismatch(""))
    B = copy(A)
    c_disk2cxf(C.plan, B)
    B
end

function \(C::CDisk2CxfPlan, A::Matrix{Float64})
    (size(A,1) == C.n && size(A,2) == 4C.n-3) || throw(DimensionMismatch(""))
    B = copy(A)
    c_cxf2disk(C.plan, B)
    B
end
