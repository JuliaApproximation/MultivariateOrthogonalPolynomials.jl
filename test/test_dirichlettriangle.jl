using MultivariateOrthogonalPolynomials, StaticArrays, BlockArrays, BlockBandedMatrices, ArrayLayouts,
    QuasiArrays, Test, ClassicalOrthogonalPolynomials, BandedMatrices, FastTransforms, LinearAlgebra, ContinuumArrays
import MultivariateOrthogonalPolynomials: tri_forwardrecurrence, grid, TriangleRecurrenceA, TriangleRecurrenceB, TriangleRecurrenceC, xy_muladd!, ExpansionLayout, Triangle, ApplyBandedBlockBandedLayout

# Basic checks 
a, b, c = true, false, true
P = DirichletTriangle(a, b, c)
@test P == P == DirichletTriangle(1, 0, 1) == DirichletTriangle(1.0, 0.0, 1.0)
@test DirichletTriangle(false, true, false) ≠ DirichletTriangle(true, false, false)
@test axes(P) == axes(JacobiTriangle(a, b, c))
@test copy(P) == P
@test JacobiTriangle(P) == JacobiTriangle(1.0, 0.0, 1.0)

## Evaluation checks 
const Pₙᵃᵇf = (n, a, b, x) -> jacobip(n, a, b, x) # Jacobi polynomials 
const P̃ₙᵃᵇf = (n, a, b, x) -> Pₙᵃᵇf(n, a, b, 2 * x - 1) # rescaled Jacobi polynomials on [0, 1]
const Pₙₖf = (n, k, x, y) -> P̃ₙᵃᵇf(n - k, 2k + 1, 0, x) * (1 - x)^k * P̃ₙᵃᵇf(k, 0, 0, y / (1 - x))
const Pₙₖᵃᵇᶜf = (n, k, a, b, c, x, y) -> P̃ₙᵃᵇf(n - k, 2k + b + c + 1, a, x) * (1 - x)^k * P̃ₙᵃᵇf(k, c, b, y / (1 - x)) # Jacobi polynomials on the unit simplex
const to_idx = (n, k) -> n * (n + 1) ÷ 2 + k + 1 # lexicographical ordering to linear indexing
function to_nk(j) # linear indexing to lexicographical ordering
    n = 0
    while to_idx(n + 1, 0) ≤ j
        n += 1
    end
    k = j - to_idx(n, 0)
    return n, k
    # could also just put this in terms of rounding to the lowest triangular number, but this works fine
end
function Qₙₖᵃᵇᶜf(n, k, a, b, c, x, y)
    z = 1 - x - y
    if a == 1 && b == 0 && c == 0
        if n == k == 0
            return Pₙₖf(0, 0, x, y)
        elseif n ≠ k
            return x * Pₙₖᵃᵇᶜf(n - 1, k, 1, 0, 0, x, y)
        else
            return Pₙₖᵃᵇᶜf(n, n, 1, 0, 0, x, y)
        end
    elseif a == 0 && b == 1 && c == 0
        if k == 0
            return P̃ₙᵃᵇf(n, 0, 0, x)
        else
            return y * Pₙₖᵃᵇᶜf(n - 1, k - 1, 0, 1, 0, x, y)
        end
    elseif a == 0 && b === 0 && c == 1
        if k == 0
            return P̃ₙᵃᵇf(n, 0, 0, x)
        else
            return z * Pₙₖᵃᵇᶜf(n - 1, k - 1, 0, 0, 1, x, y)
        end
    elseif a == 1 && b == 1 && c == 0
        if n == k == 0
            return 1.0
        elseif k == 0
            return x * P̃ₙᵃᵇf(n - 1, 0, 1, x)
        elseif n ≠ k
            return x * y * Pₙₖᵃᵇᶜf(n - 2, k - 1, 1, 1, 0, x, y)
        else
            return y * Pₙₖᵃᵇᶜf(n - 1, n - 1, 0, 1, 0, x, y)
        end
    elseif a == 1 && b == 0 && c == 1
        if n == k == 0
            return 1.0
        elseif k == 0
            return x * P̃ₙᵃᵇf(n - 1, 0, 1, x)
        elseif n ≠ k
            return x * z * Pₙₖᵃᵇᶜf(n - 2, k - 1, 1, 0, 1, x, y)
        else
            return z * Pₙₖᵃᵇᶜf(n - 1, n - 1, 0, 0, 1, x, y)
        end
    elseif a == 0 && b == 1 && c == 1
        if n == k == 0
            return 1.0
        elseif k == 0
            return (1 - x) * Pₙₖf(n - 1, 0, x, y)
        elseif k == 1
            return (1 - x - 2y) * Pₙₖf(n - 1, 0, x, y)
        else
            return y * z * Pₙₖᵃᵇᶜ(n - 2, k - 2, 0, 1, 1, x, y)
        end
    elseif a == 1 && b == 1 && c == 1
        if n == k == 0
            return 1.0
        elseif n == 1 && k == 0
            return 1 - 2x
        elseif n == k == 1
            return 1 - x - 2y
        elseif k == 0
            return x * (1 - x) * Pₙₖᵃᵇᶜf(n - 2, 0, 1, 0, 0, x, y)
        elseif k == 1
            return x * (1 - x - 2y) * Pₙₖᵃᵇᶜf(n - 2, 0, 1, 0, 0, x, y)
        elseif n ≠ k
            return x * y * z * Pₙₖᵃᵇᶜf(n - 3, k - 2, 1, 1, 1, x, y)
        else
            return y * z * Pₙₖᵅᵇᶜf(n - 2, n - 2, 0, 1, 1, x, y)
        end
    else
        throw(ArgumentError("Invalid values for a, b, c"))
    end
end
for (a, b, c) in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    P = DirichletTriangle(a, b, c)
    xy = SVector(0.2, 0.3)
    vals = P[xy, BlockRange(1:10)]
    Qvals = [Qₙₖᵃᵇᶜf(to_nk(i)..., a, b, c, xy...) for i in 1:to_idx(10 - 1, 10 - 1)]
    @test vals ≈ Qvals
end

a, b, c = 0, 1, 0
P = DirichletTriangle(a, b, c)
xy = SVector(0.2, 0.3)
vals = P[xy, 17]
vals ≈ Qₙₖᵃᵇᶜf(to_nk(17)..., a, b, c, xy[1], xy[2])

P[xy, 1], Qₙₖᵃᵇᶜf(to_nk(1)..., a, b, c, xy[1], xy[2])
P[xy, 2], Qₙₖᵃᵇᶜf(to_nk(2)..., a, b, c, xy[1], xy[2])

P[xy, 3], Qₙₖᵃᵇᶜf(to_nk(3)..., a, b, c, xy[1], xy[2])


JR = findblockindex(axes(P, 2), 3) |> BlockArrays.block |> x -> Block.(Base.OneTo(Int(x)))
a, b, c = P.a, P.b, P.c
n = length(JR)
x, y = xy
PJ = JacobiTriangle(P)
jacobi_vals = PJ[xy, JR]
dirichlet_vals = copy(jacobi_vals)
P̃ = Jacobi(0.0, 0.0)[2x-1, 1:n]
prev_block = view(jacobi_vals, Block(1))
mult = b ? y : z
i = 2
block = view(jacobi_vals, Block(i))
dirichlet_block = view(dirichlet_vals, Block(i))