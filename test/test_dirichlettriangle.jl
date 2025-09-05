using MultivariateOrthogonalPolynomials, HarmonicOrthogonalPolynomials, SpecialFunctions, ClassicalOrthogonalPolynomials, ContinuumArrays, StaticArrays, LinearAlgebra, BlockArrays, ArrayLayouts, HCubature
using BlockArrays: block, blockindex, BlockIndex
using HarmonicOrthogonalPolynomials: BlockOneTo
using MultivariateOrthogonalPolynomials: UnitTriangle, TriangleWeight

using Test

@testset "Ultraspherical Ladders" begin
    for λ in (-1 / 2, 1 / 2, 2, 3 / 2, 5 / 2)
        for n in 1:10
            x = 0.3
            U = Ultraspherical(λ)
            L1 = (λ, x, n, u) -> diff(u)[x, n+1]
            L2 = (λ, x, n, u) -> 2(2λ - 1) * (λ - 1) * x * u[x, n+1] - 2(λ - 1) * (1 - x^2) * diff(u)[x, n+1]
            @test L1(λ, x, n, U) ≈ 2λ * Ultraspherical(λ + 1)[x, n] atol = 1e-9
            @test L2(λ, x, n, U) ≈ (n + 1) * (n + 2λ - 1) * Ultraspherical(λ - 1)[x, n+2] atol = 1e-9

            Ū = Ultraspherical(λ)[affine(0 .. 1, -1 .. 1), :]
            L1 = (λ, x, n, u) -> diff(u)[x, n+1]
            L2 = (λ, x, n, u) -> 2(2λ - 1) * (λ - 1) * (2x - 1) * u[x, n+1] - 4(λ - 1) * x * (1 - x) * diff(u)[x, n+1]
            @test L1(λ, x, n, Ū) ≈ 4λ * Ultraspherical(λ + 1)[affine(0 .. 1, -1 .. 1), :][x, n] atol = 1e-9
            @test L2(λ, x, n, Ū) ≈ (n + 1) * (n + 2λ - 1) * Ultraspherical(λ - 1)[affine(0 .. 1, -1 .. 1), :][x, n+2] atol = 1e-9
        end
    end
    λ = -1 / 2
    for n in 1:10
        x = 0.3
        U = Ultraspherical(-1 / 2)
        L1 = (x, n, u) -> diff(u)[x, n+1]
        L2 = (x, n, u) -> 6 * x * u[x, n+1] + 3(1 - x^2) * diff(u)[x, n+1]
        @test L1(x, n, U) ≈ -Ultraspherical(1 / 2)[x, n] atol = 1e-9
        @test L2(x, n, U) ≈ (n + 1) * (n - 2) * Ultraspherical(-3 / 2)[x, n+2] atol = 1e-9

        Ū = Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]
        L1 = (x, n, u) -> diff(u)[x, n+1]
        L2 = (x, n, u) -> 6(2x - 1) * u[x, n+1] + 6 * x * (1 - x) * diff(u)[x, n+1]
        @test L1(x, n, Ū) ≈ -2 * Ultraspherical(1 / 2)[affine(0 .. 1, -1 .. 1), :][x, n] atol = 1e-9
        @test L2(x, n, Ū) ≈ (n + 1) * (n - 2) * Ultraspherical(-3 / 2)[affine(0 .. 1, -1 .. 1), :][x, n+2] atol = 1e-9
    end
end

@testset "Jacobi with one negative parameter" begin
    x = 0.3
    for α in [0:10; -1//2:5//2]
        m = 1:9
        lhs = Jacobi(α, -1)[x, m.+1]
        @test Jacobi(α, -1)[x, 1] == 1
        rhs = (1 + x) .* (m .+ α) ./ (2m) .* Jacobi(α, 1)[x, m]
        @test lhs ≈ rhs
        m = 1:9
        lhs = Jacobi(-1, α)[x, m.+1]
        @test Jacobi(-1, α)[x, 1] == 1
        rhs = -(1 - x) .* (m .+ α) ./ (2m) .* Jacobi(1, α)[x, m]
        @test lhs ≈ rhs
    end
end

@testset "Relating Ultraspherical and Jacobi" begin
    scale = (n, λ) -> 2^(1 - 2λ) * sqrt(π) * gamma(n + 2λ) / (gamma(λ) * gamma(n + λ + 1 / 2))
    λ = 3 / 2
    x = 0.3
    @test Ultraspherical(λ)[x, 1:11] ≈ scale.(0:10, λ) .* Jacobi(λ - 1 / 2, λ - 1 / 2)[x, 1:11]
    scale = (n, λ) -> gamma(λ) * gamma(n + λ + 1 / 2) / (2^(1 - 2λ) * sqrt(π) * gamma(n + 2λ))
    λ = 3 / 2
    x = 0.4
    @test Jacobi(λ - 1 / 2, λ - 1 / 2)[x, 1:11] ≈ scale.(0:10, λ) .* Ultraspherical(λ)[x, 1:11]
    @test Ultraspherical(1 / 2)[x, 1:11] ≈ Jacobi(0, 0)[x, 1:11]
end

struct DirichletTriangle{T} <: BivariateOrthogonalPolynomial{T} end
DirichletTriangle() = DirichletTriangle{Float64}()
function Base.getindex(P::DirichletTriangle{T}, xy::StaticVector{2}, B::BlockIndex{1})::T where {T}
    x, y = xy
    n, k = Int(block(B)), blockindex(B)
    @assert n ≥ k ≥ 1
    a = b = c = -1
    a1, b1 = 2(k - 1) + b + c + 1, a
    J2 = (1 - x)^(k - 1)
    J3 = Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]
    if a1 == b1 == -1
        J1 = J3
        return J1[x, n-k+1] * J2 * J3[y/(1-x), k]
    else
        J1 = Jacobi(a1, b1)[affine(0 .. 1, -1 .. 1), :]
        return J1[x, n-k+1] * J2 * J3[y/(1-x), k]
    end
end
function Base.getindex(P::DirichletTriangle, xy::StaticVector{2}, B::Block{1})
    return [P[xy, B[j]] for j in 1:Int(B)]
end
function Base.getindex(P::DirichletTriangle, xy::StaticVector{2}, JR::BlockOneTo)
    return mortar([P[xy, Block(J)] for J in 1:Int(JR[end])])
end
function Base.getindex(P::JacobiTriangle{T}, xy::SVector{2}, B::BlockIndex{1})::T where {T}
    x, y = xy
    n, k = Int(block(B)), blockindex(B)
    @assert n ≥ k ≥ 1
    a, b, c = P.a, P.b, P.c 
    a1, b1 = 2(k - 1) + b + c + 1, a
    J2 = (1 - x)^(k - 1)
    if b == c == -1 
        J3 = Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]
    else
        J3 = Jacobi(c, b)[affine(0 .. 1, -1 .. 1), :]
    end
    if a1 == b1 == -1 
        J1 = J3  
    else 
        J1 = Jacobi(a1, b1)[affine(0 .. 1, -1 .. 1), :]
    end
    return J1[x, n-k+1] * J2 * J3[y/(1-x), k]
end
Base.axes(::DirichletTriangle{T}) where {T} = (Inclusion(UnitTriangle{T}()), blockedrange(oneto(∞)))
Base.copy(P::DirichletTriangle) = P
Base.show(io::IO, P::DirichletTriangle) = Base.summary(io, P)
Base.summary(io::IO, P::DirichletTriangle) = print(io, "DirichletTriangle")
ClassicalOrthogonalPolynomials.orthogonalityweight(P::DirichletTriangle) = TriangleWeight(-1, -1, -1)

@testset "DirichletTriangle definition" begin
    dirichlet_triangle_def(x, y, n, k) = (k > 0 ? Jacobi(2k - 1, -1)[affine(0 .. 1, -1 .. 1), :][x, n-k+1] : Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :][x, n+1]) * (1 - x)^k * Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :][y/(1-x), k+1]
    P = DirichletTriangle()
    @test eltype(P) == Float64
    @test P[SVector(0.2, 0.3), 1] ≈ dirichlet_triangle_def(0.2, 0.3, 0, 0)
    @test P[SVector(0.2, 0.3), 2] ≈ dirichlet_triangle_def(0.2, 0.3, 1, 0)
    @test P[SVector(0.2, 0.3), 3] ≈ dirichlet_triangle_def(0.2, 0.3, 1, 1)
    @test P[SVector(0.2, 0.3), 4] ≈ dirichlet_triangle_def(0.2, 0.3, 2, 0)
    @test P[SVector(0.2, 0.3), 5] ≈ dirichlet_triangle_def(0.2, 0.3, 2, 1)
    @test P[SVector(0.2, 0.3), 6] ≈ dirichlet_triangle_def(0.2, 0.3, 2, 2)
    @test P[SVector(0.2, 0.3), Block(5)[3]] ≈ dirichlet_triangle_def(0.2, 0.3, 4, 2)
    @test P[SVector(0.2, 0.3), Block(5)] ≈ [dirichlet_triangle_def(0.2, 0.3, 4, k) for k in 0:4]
    @test P[SVector(0.2, 0.3), Block.(1:5)] ≈ [dirichlet_triangle_def.(0.2, 0.3, 0, 0);
        dirichlet_triangle_def.(0.2, 0.3, 1, 0:1);
        dirichlet_triangle_def.(0.2, 0.3, 2, 0:2);
        dirichlet_triangle_def.(0.2, 0.3, 3, 0:3);
        dirichlet_triangle_def.(0.2, 0.3, 4, 0:4)]
    @test P[SVector(0.2, 0.3), Block.(2:4)] ≈ P[SVector(0.2, 0.3), Block.(1:4)][Block.(2:4)]
    @test axes(P) == axes(JacobiTriangle())
    @test copy(P) === P
    @test ClassicalOrthogonalPolynomials.orthogonalityweight(P) == ClassicalOrthogonalPolynomials.orthogonalityweight(JacobiTriangle(-1, -1, -1))
end

function integrate_triangle(f, a, b, c)
    # Integrates f(x, y) on the triangle {0 < x < 1, 0 < y < 1, 0 < y < 1 - x},
    # weighted by x^a y^b (1 - x - y)^c 
    g = let a = a, b = b, c = c, f = f
        ((x, y),) -> f(x, y) * x^a * y^b * (1 - x - y)^c
    end
    h = let g = g
        ((u, v),) -> g((u, (1 - u) * v)) * (1 - u)
    end
    return HCubature.hcubature(h, [0, 0], [1, 1], atol=1e-9)[1]
end

@test integrate_triangle((x, y) -> x * y + 2exp(5(x + y)), 1, 3, 1) ≈ 131exp(5) / 156250 + 215209 / 118125000 atol = 1e-6
@test integrate_triangle(Returns(1.0), 0, 0, 0) ≈ 1 / 2

get_abc(P) = P isa DirichletTriangle ? (-1, -1, -1) : (P.a, P.b, P.c)

function approx_dy(P, n, k, x, y)
    k == 0 && return zero(eltype(P))
    a, b, c = get_abc(P)
    if P isa DirichletTriangle
        return -2 * JacobiTriangle(-1, 0, 0)[SVector(x, y), Block(n)[k]]
    else
        return (k + b + c + 1) * JacobiTriangle(a, b + 1, c + 1)[SVector(x, y), Block(n)[k]]
    end
end

function finite_difference_dy(P, n, k, x, y)
    g = let P = P, n = n, k = k
        (xx, yy) -> P[SVector(xx, yy), Block(n + 1)[k+1]]
    end
    # Approximate dg/dy 
    h = sqrt(eps(eltype(P))) * max(abs(y), 1)
    return (g(x, y + h) - g(x, y - h)) / (2h)
end

function expand_dy(P, n, k)
    # Expands the y-derivative of P(a, b, c) in the P(a, b+1, c+1) basis 
    a, b, c = get_abc(P)
    Q = JacobiTriangle(a, b + 1, c + 1)
    cfs = BlockArray{eltype(P)}(undef, 1:7)
    deriv = let P = P, n = n, k = k
        ((x, y),) -> approx_dy(P, n, k, x, y)
    end
    for n in blockaxes(cfs, 1)
        for k in 1:Int(n)
            integrand_num = let deriv = deriv, Q = Q
                (x, y) -> deriv((x, y),) * Q[SVector(x, y), n[k]]
            end
            integrand_den = let Q = Q
                (x, y) -> Q[SVector(x, y), Block(n)[k]]^2
            end
            num = integrate_triangle(integrand_num, get_abc(Q)...)
            den = integrate_triangle(integrand_den, get_abc(Q)...)
            cfs[n[k]] = num / den
        end
    end
    return cfs
end

#=
@testset "Derivative testing" begin
    P = JacobiTriangle(2, 1, 3)
    n, k = 5, 3
    x, y = 0.3, 0.25612
    @test approx_dy(P, n, k, x, y) ≈ finite_difference_dy(P, n, k, x, y) atol = 1e-6
    @test approx_dy(P, n, k, x, y) ≈ diff(P, Val((0, 1)))[SVector(x, y), Block(n + 1)[k+1]]
    P = DirichletTriangle()
    @test approx_dy(P, n, k, x, y) ≈ finite_difference_dy(P, n, k, x, y) atol = 1e-6

    P = JacobiTriangle(2, 1, 3)
    coeffs = zeros(100, 100)
    for n in 0:3
        for k in 0:n
            @show n, k
            idx = axes(P, 2)[Block(n + 1)[k+1]]
            cfs = expand_dy(P, n, k)
            coeffs[eachindex(cfs), idx] = cfs
        end
    end
    midx = axes(P, 2)[Block(4)[4]]
    coeffs2 = copy(coeffs)
    coeffs2[abs.(coeffs2).<1e-5] .= 0
    coeffs3 = diff(P, Val((0, 1))).args[2][1:100, 1:midx]
    coeffs2 = coeffs2[1:100, 1:midx]
    @test coeffs2 ≈ coeffs3 atol = 1e-4

    P = DirichletTriangle()
    coeffs = zeros(100, 100)
    for n in 0:3
        for k in 0:n
            @show n, k
            idx = axes(P, 2)[Block(n + 1)[k+1]]
            cfs = expand_dy(P, n, k)
            coeffs[eachindex(cfs), idx] = cfs
        end
    end
    midx = axes(P, 2)[Block(4)[4]]
    for k in 1:midx
        expn = JacobiTriangle(-1, 0, 0)[:, 1:100] * coeffs[:, k]
        nk = findblockindex(axes(P, 2), k)
        n, k = Int(block(nk)), blockindex(nk)
        @test_broken expn[SVector(0.2, 0.3)] ≈ approx_dy(P, n - 1, k - 1, 0.2, 0.3)
    end
end
=#

@testset "Some conversion testing" begin
    x = 0.21231243111555555555555555555555
    U = Ultraspherical(-1 / 2)
    V = Ultraspherical(1 / 2)
    n = 5
    @test x * V[x, n-1] ≈ (n - 2) / (2n - 3) * V[x, n-2] + (n - 1) / (2n - 3) * V[x, n]
    @test x * U[x, n+1] ≈ (n - 2) / (2n - 1) * U[x, n] + (n + 1) / (2n - 1) * U[x, n+2]
    R = Jacobi(-1, 0)
    for n in 2:10
        @test U[x, n+1] ≈ -2 / (2n - 1) * (R[x, n] + R[x, n+1])
    end
    R = Jacobi(0, -1)
    for n in 2:10
        @test U[x, n+1] ≈ 2 / (2n - 1) * (R[x, n] - R[x, n+1])
    end
    n, α, β = 5, 3, 2
    x = 0.7771
    @test (2 * n + α + β + 1) * Jacobi(α, β)[x, n+1] ≈ (n + α + β + 1) * Jacobi(α + 1, β)[x, n+1] - (n + β) * Jacobi(α + 1, β)[x, n]
    @test (n + α / 2 + β / 2 + 1) * (1 - x) * Jacobi(α + 1, β)[x, n+1] ≈ -(n + 1) * Jacobi(α, β)[x, n+2] + (n + α + 1) * Jacobi(α, β)[x, n+1]
    @test (1 - x) * Jacobi(α + 1, β)[affine(0 .. 1, -1 .. 1), :][x, n+1] ≈ (n + α + 1) / (2n + α + β + 2) * Jacobi(α, β)[affine(0 .. 1, -1 .. 1), :][x, n+1] - (n + 1) / (2n + α + β + 2) * Jacobi(α, β)[affine(0 .. 1, -1 .. 1), :][x, n+2]
    @test (2 * n + α + β + 1) * Jacobi(α, β)[affine(0 .. 1, -1 .. 1), :][x, n+1] ≈ (n + α + β + 1) * Jacobi(α + 1, β)[affine(0 .. 1, -1 .. 1), :][x, n+1] - (n + β) * Jacobi(α + 1, β)[affine(0 .. 1, -1 .. 1), :][x, n]
end

## Ultraspherical ladders redux
# Unshifted
L1 = (x, n, u) -> diff(u)[x, n+1]
@test L1(0.3, 5, Ultraspherical(-1 / 2)) ≈ -Ultraspherical(1 / 2)[0.3, 5]
L2 = (x, n, u) -> begin
    if n == 0
        u[x, n+1]
    else
        -1 / 2 * (n - 1) * u[x, n+1] - 1 / 2 * (1 + x) * diff(u)[x, n+1]
    end
end
@test L2(0.3, 0, Ultraspherical(-1 / 2)) ≈ Jacobi(0, -1)[0.3, 1]
@test L2(0.3, 1, Ultraspherical(-1 / 2)) ≈ Jacobi(0, -1)[0.3, 2]
@test L2(0.3, 2, Ultraspherical(-1 / 2)) ≈ Jacobi(0, -1)[0.3, 3]
@test L2(0.3, 16, Ultraspherical(-1 / 2)) ≈ Jacobi(0, -1)[0.3, 17]
L3 = (x, n, u) -> begin
    if n == 0
        u[x, n+1]
    else
        -1 / 2 * (n - 1) * u[x, n+1] + 1 / 2 * (1 - x) * diff(u)[x, n+1]
    end
end
@test L3(0.3, 0, Ultraspherical(-1 / 2)) ≈ Jacobi(-1, 0)[0.3, 1]
@test L3(0.3, 1, Ultraspherical(-1 / 2)) ≈ Jacobi(-1, 0)[0.3, 2]
@test L3(0.3, 2, Ultraspherical(-1 / 2)) ≈ Jacobi(-1, 0)[0.3, 3]
@test L3(0.3, 16, Ultraspherical(-1 / 2)) ≈ Jacobi(-1, 0)[0.3, 17]
L4d = (x, n, u) -> begin
    if n == 0
        (1 + x) * diff(u)[x, n+1]
    elseif n == 1
        u[x, n+1] - (1 + x) * diff(u)[x, n+1]
    else
        1 / 2 * (n * u[x, n+1] - (1 + x) * diff(u)[x, n+1])
    end
end
@test L4d(0.3, 0, Ultraspherical(-1 / 2)) ≈ 0
@test L4d(0.3, 1, Ultraspherical(-1 / 2)) ≈ Jacobi(0, -1)[0.3, 1]
@test L4d(0.3, 2, Ultraspherical(-1 / 2)) ≈ Jacobi(0, -1)[0.3, 2]
@test L4d(0.3, 16, Ultraspherical(-1 / 2)) ≈ Jacobi(0, -1)[0.3, 16]
L5d = (x, n, u) -> begin
    if n == 0
        (1 - x) * diff(u)[x, n+1]
    elseif n == 1
        -u[x, n+1] - (1 - x) * diff(u)[x, n+1]
    else
        -1 / 2 * (n * u[x, n+1] + (1 - x) * diff(u)[x, n+1])
    end
end
@test L5d(0.3, 0, Ultraspherical(-1 / 2)) ≈ 0
@test L5d(0.3, 1, Ultraspherical(-1 / 2)) ≈ Jacobi(-1, 0)[0.3, 1]
@test L5d(0.3, 2, Ultraspherical(-1 / 2)) ≈ Jacobi(-1, 0)[0.3, 2]
@test L5d(0.3, 16, Ultraspherical(-1 / 2)) ≈ Jacobi(-1, 0)[0.3, 16]

# Shifted
L1 = (x, n, u) -> diff(u)[x, n+1]
@test L1(0.3, 5, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ -2Ultraspherical(1 / 2)[affine(0 .. 1, -1 .. 1), :][0.3, 5]
L2 = (x, n, u) -> begin
    if n == 0
        u[x, n+1]
    else
        -1 / 2 * (n - 1) * u[x, n+1] - 1 / 2 * x * diff(u)[x, n+1]
    end
end
@test L2(0.3, 0, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(0, -1)[affine(0 .. 1, -1 .. 1), :][0.3, 1]
@test L2(0.3, 1, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(0, -1)[affine(0 .. 1, -1 .. 1), :][0.3, 2]
@test L2(0.3, 2, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(0, -1)[affine(0 .. 1, -1 .. 1), :][0.3, 3]
@test L2(0.3, 16, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(0, -1)[affine(0 .. 1, -1 .. 1), :][0.3, 17]
L3 = (x, n, u) -> begin
    if n == 0
        u[x, n+1]
    else
        -1 / 2 * (n - 1) * u[x, n+1] + 1 / 2 * (1 - x) * diff(u)[x, n+1]
    end
end
@test L3(0.3, 0, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(-1, 0)[affine(0 .. 1, -1 .. 1), :][0.3, 1]
@test L3(0.3, 1, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(-1, 0)[affine(0 .. 1, -1 .. 1), :][0.3, 2]
@test L3(0.3, 2, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(-1, 0)[affine(0 .. 1, -1 .. 1), :][0.3, 3]
@test L3(0.3, 16, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(-1, 0)[affine(0 .. 1, -1 .. 1), :][0.3, 17]
L4d = (x, n, u) -> begin
    if n == 0
        x * diff(u)[x, n+1]
    elseif n == 1
        u[x, n+1] - x * diff(u)[x, n+1]
    else
        1 / 2 * (n * u[x, n+1] - x * diff(u)[x, n+1])
    end
end
@test L4d(0.3, 0, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ 0
@test L4d(0.3, 1, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(0, -1)[affine(0 .. 1, -1 .. 1), :][0.3, 1]
@test L4d(0.3, 2, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(0, -1)[affine(0 .. 1, -1 .. 1), :][0.3, 2]
@test L4d(0.3, 16, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(0, -1)[affine(0 .. 1, -1 .. 1), :][0.3, 16]
L5d = (x, n, u) -> begin
    if n == 0
        (1 - x) * diff(u)[x, n+1]
    elseif n == 1
        -u[x, n+1] - (1 - x) * diff(u)[x, n+1]
    else
        -1 / 2 * (n * u[x, n+1] + (1 - x) * diff(u)[x, n+1])
    end
end
@test L5d(0.3, 0, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ 0
@test L5d(0.3, 1, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(-1, 0)[affine(0 .. 1, -1 .. 1), :][0.3, 1]
@test L5d(0.3, 2, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(-1, 0)[affine(0 .. 1, -1 .. 1), :][0.3, 2]
@test L5d(0.3, 16, Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :]) ≈ Jacobi(-1, 0)[affine(0 .. 1, -1 .. 1), :][0.3, 16]

x = 0.3
n = 5
k = 3
P = Jacobi(2k - 1, -1)[affine(0 .. 1, -1 .. 1), :]
P2 = Jacobi(2k, 0)[affine(0 .. 1, -1 .. 1), :]
lhs = (1 - x) * diff(P)[x, n-k+1]
rhs = 1 / 2 * (n + k - 1) * ((k^2 - n^2) / (n * (2 * n - 1)) * P2[x, n-k+1] + (1 + k^2 / (n * (n - 1))) * P2[x, n-k-1+1] + 1 / 2 * ((2k^2 + n - 1) / ((2n - 1) * (n - 1)) - 1) * P2[x, n-k-2+1])
@test lhs ≈ rhs

function approx_dx(P, n, k, x, y)
    k == 0 && return zero(eltype(P))
    a, b, c = get_abc(P)
    if P isa DirichletTriangle
        return (-k * Jacobi(2k - 1, -1)[affine(0 .. 1, -1 .. 1), :][x, n-k+1] + (1 - x) * (n + k - 1) * Jacobi(2k, 0)[affine(0 .. 1, -1 .. 1), :][x, n-k-1+1]) * (1 - x)^(k - 1) * Ultraspherical(-1 / 2)[affine(0 .. 1, -1 .. 1), :][y/(1-x), k+1] - 2Jacobi(2k - 1, -1)[affine(0 .. 1, -1 .. 1), :][x, n-k+1] * (1 - x)^(k - 1) * Ultraspherical(1 / 2)[affine(0 .. 1, -1 .. 1), :][y/(1-x), k]
    else
        val = (n + k + a + b + c + 2) * (k + b + c + 1) * JacobiTriangle(a + 1, b, c + 1)[SVector(x, y), Block(n)[k+1]] + (k + b) * (n + k + b + c + 1) * JacobiTriangle(a + 1, b, c + 1)[SVector(x, y), Block(n)[k]]
        return val / (2k + b + c + 1)
    end
end

function finite_difference_dx(P, n, k, x, y)
    @show 2k+b+c+1, n+k+a+b+c+2,k+b+c+1,k+b,n+k+b+c+1
    g = let P = P, n = n, k = k
        (xx, yy) -> P[SVector(xx, yy), Block(n + 1)[k + 1]]
    end
    # Approximate dg/dx
    h = sqrt(eps(eltype(P))) * max(abs(x), 1)
    return (g(x + h, y) - g(x - h, y)) / (2h)
end

P = JacobiTriangle(2, 3, 1)
dP = diff(P, Val((1, 0)))
@test dP[SVector(0.2, 0.3), Block(1)[1]] ≈ finite_difference_dx(P, 0, 0, 0.2, 0.3)
@test dP[SVector(0.2, 0.3), Block(2)[1]] ≈ finite_difference_dx(P, 1, 0, 0.2, 0.3)
@test dP[SVector(0.2, 0.3), Block(2)[2]] ≈ finite_difference_dx(P, 1, 1, 0.2, 0.3)
@test dP[SVector(0.2, 0.3), Block(18)[12]] ≈ finite_difference_dx(P, 17, 11, 0.2, 0.3)
@test dP[SVector(0.2, 0.3), Block(18)[12]] ≈ approx_dx(P, 17, 11, 0.2, 0.3)
P = JacobiTriangle(-1, -1, -1)
dP = diff(P, Val((1, 0)))
@test -2dP[SVector(0.2, 0.3), Block(5)[3]] ≈ finite_difference_dx(P, 4, 2, 0.2, 0.3)
@test -dP[SVector(0.2, 0.3), Block(5)[4]] ≈ finite_difference_dx(P, 4, 3, 0.2, 0.3)
@test -2dP[SVector(0.2, 0.3), Block(5)[5]]/3 ≈ finite_difference_dx(P, 4, 4, 0.2, 0.3)
@test -dP[SVector(0.2, 0.3), Block(6)[1]]/2 ≈ finite_difference_dx(P, 5, 0, 0.2, 0.3)
@test -dP[SVector(0.2, 0.3), Block(6)[2]] ≈ finite_difference_dx(P, 5, 1, 0.2, 0.3)
@test -2dP[SVector(0.2, 0.3), Block(6)[3]] ≈ finite_difference_dx(P, 5, 2, 0.2, 0.3)
@test -dP[SVector(0.2, 0.3), Block(6)[4]] ≈ finite_difference_dx(P, 5, 3, 0.2, 0.3)
@test -2dP[SVector(0.2, 0.3), Block(6)[5]]/3 ≈ finite_difference_dx(P, 5, 4, 0.2, 0.3)
@test -dP[SVector(0.2, 0.3), Block(6)[6]]/2 ≈ finite_difference_dx(P, 5, 5, 0.2, 0.3)



dP[SVector(0.2, 0.3), Block(5)[3]] 