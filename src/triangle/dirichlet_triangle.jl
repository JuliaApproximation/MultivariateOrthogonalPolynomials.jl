# Expansions in OPs orthogonal to x^ay^b(1-x-y)^c, augmented 
# with additional polynomials so that they span all polynomials of degree 
# less than or equal to N, where a, b, c are binary values.
struct DirichletTriangle{T} <: BivariateOrthogonalPolynomial{T}
    a::Bool
    b::Bool
    c::Bool
    function DirichletTriangle{T}(a, b, c) where {T}
        a, b, c = convert.(Bool, (a, b, c))
        nedges = count((a, b, c))
        iszero(nedges) && throw(ArgumentError("DirichletTriangle must have at least one edge."))
        return new{T}(a, b, c)
    end
    # Intentionally not defining DirichletTriangle()
end
DirichletTriangle(a, b, c) = DirichletTriangle{float(Bool)}(a, b, c) # float(Bool) = Float64 
==(T1::DirichletTriangle, T2::DirichletTriangle) = T1.a == T2.a && T1.b == T2.b && T1.c == T2.c

axes(P::DirichletTriangle{T}) where {T} = (Inclusion(UnitTriangle{T}()), blockedrange(oneto(∞)))

copy(P::DirichletTriangle) = P

show(io::IO, P::DirichletTriangle) = summary(io, P)
summary(io::IO, P::DirichletTriangle) = print(io, "DirichletTriangle(", P.a ? 1 : 0, ", ", P.b ? 1 : 0, ", ", P.c ? 1 : 0, ")")

_nedges(P::DirichletTriangle) = count((P.a, P.b, P.c))

JacobiTriangle(P::DirichletTriangle{T}) where {T} = JacobiTriangle{T}(1.0P.a, 1.0P.b, 1.0P.c)

function getindex(P::DirichletTriangle, xy::SVector{2}, JR::BlockOneTo)
    if _nedges(P) == 1
        return _getindex_dirichlet_1(P, xy, JR)
    elseif _nedges(P) == 2
        return _getindex_dirichlet_2(P, xy, JR)
    else # if _nedges(P) == 3
        return _getindex_dirichlet_3(P, xy, JR)
    end
end

function _getindex_dirichlet_1(P::DirichletTriangle, xy::SVector{2}, JR::BlockOneTo)
    a, b, _ = P.a, P.b, P.c
    n = length(JR)
    x, y = xy
    z = 1 - x - y
    PJ = JacobiTriangle(P)
    jacobi_vals = PJ[xy, JR]
    dirichlet_vals = copy(jacobi_vals)
    if a
        prev_block = view(jacobi_vals, Block(1))
        for i in 2:n
            block = view(jacobi_vals, Block(i))
            dirichlet_block = view(dirichlet_vals, Block(i))
            for k in eachindex(prev_block)
                dirichlet_block[k] = x * prev_block[k]
            end
            prev_block = block
        end
        dirichlet_vals[1] = JacobiTriangle(0, 0, 0)[xy, 1]
    else
        P̃ = Jacobi(0.0, 0.0)[2x-1, 1:n] # == Jacobi(0.0, 0.0)[affine(-1..1, 0..1), :][x, 1:n] 
        prev_block = view(jacobi_vals, Block(1))
        mult = b ? y : z
        for i in 2:n
            block = view(jacobi_vals, Block(i))
            dirichlet_block = view(dirichlet_vals, Block(i))
            for k in 2:lastindex(dirichlet_block)
                dirichlet_block[k] = mult * prev_block[k-1]
            end
            dirichlet_block[1] = P̃[i]
            prev_block = block
        end
    end
    return dirichlet_vals
end

function _getindex_dirichlet_2(P::DirichletTriangle, xy::SVector{2}, JR::BlockOneTo)
    a, b, c = P.a, P.b, P.c
    n = length(JR)
    x, y = xy
    z = 1 - x - y
    PJ = JacobiTriangle(P)
    jacobi_vals = PJ[xy, JR]
    dirichlet_vals = copy(jacobi_vals)
    dirichlet_vals[1] = 1.0
    if a && b
        P̃ = Jacobi(0.0, 1.0)[2x-1, 1:n]
        PJ_one = JacobiTriangle(0.0, 1.0, 0.0)
        jacobi_one_vals = PJ_one[xy, JR]
        for i in 2:n
            dirichlet_block = view(dirichlet_vals, Block(i))
            dirichlet_block[1] = x * P̃[i-1]
            dirichlet_block[end] = y * view(jacobi_one_vals, Block(i - 1))[end]
            for k in 2:(i-1)
                dirichlet_block[k] = x * y * view(jacobi_vals, Block(i - 2))[k-1]
            end
        end
    elseif a && c
        P̃ = Jacobi(0.0, 1.0)[2x-1, 1:n]
        PJ_one = JacobiTriangle(0.0, 0.0, 1.0)
        jacobi_one_vals = PJ_one[xy, JR]
        for i in 2:n
            dirichlet_block = view(dirichlet_vals, Block(i))
            dirichlet_block[1] = x * P̃[i-1]
            dirichlet_block[end] = z * view(jacobi_one_vals, Block(i - 1))[end]
            for k in 2:(i-1)
                dirichlet_block[k] = x * z * view(jacobi_vals, Block(i - 2))[k-1]
            end
        end
    else
        P̃ = Jacobi(1.0, 0.0)[2x-1, 1:n]
        for i in 2:n
            dirichlet_block = view(dirichlet_vals, Block(i))
            dirichlet_block[1] = (1 - x) * P̃[i-1]
            dirichlet_block[2] = (1 - x - 2y) * P̃[i-1]
            for k in 3:i
                dirichlet_block[k] = y * z * view(jacobi_vals, Block(i - 2))[k-2]
            end
        end
    end
    return dirichlet_vals
end

function _getindex_dirichlet_3(P::DirichletTriangle, xy::SVector{2}, JR::BlockOneTo)
    n = length(JR)
    x, y = xy 
    z = 1 - x - y
    dirichlet_vals = PseudoBlockVector(zeros(n * (n - 1) ÷ 2 + n), 1:n)
    dirichlet_vals[1] = 1.0
    n ==1 && return dirichlet_vals
    dirichlet_vals[2] = 1.0 - 2.0x
    dirichlet_vals[3] = 1.0 - x - 2.0y
    P¹¹_vals = Jacobi(1.0, 1.0)[x, 1:n]
    P¹¹¹_vals = JacobiTriangle(1.0, 1.0, 1.0)[xy, JR]
    P⁰¹¹_vals = JacobiTriangle(0.0, 1.0, 1.0)[xy, JR]
    for i in 3:n
        dirichlet_block = view(dirichlet_vals, Block(i))
        dirichlet_block[1] = x * (1 - x) * P¹¹_vals[i-2]
        dirichlet_block[2] = x * (1.0 - x - 2.0y) * P¹¹_vals[i-2]
        dirichlet_block[end] = y * z * view(P⁰¹¹_vals, Block(i - 2))[end]
        for k in 3:(i-1)
            dirichlet_block[k] = x * y * z * view(P¹¹¹_vals, Block(i - 3))[k-2]
        end
    end
    return dirichlet_vals
end