# module MultivariateOrthogonalPolynomialsMakieExt

using GLMakie
using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays
import Makie: mesh, mesh!
using ContinuumArrays: plotgridvalues

export contourf, contourf

contourf(f::Fun; kwds...) = _mesh(meshdata(f)...; shading=false, kwds...)
contourf!(s, f::Fun; kwds...) = _mesh!(s,  meshdata(f)...; shading=false, kwds...)


function _mesh(p, T, v; resolution=(400,400), kwds...)
    T_mat = Array{Int}(undef, length(T), 3)
    for k = 1:length(T)
        T_mat[k,:] .= T[k]
    end
    s = Scene(resolution=resolution)
    mesh!(s, [first.(p) last.(p)], T_mat; color=v, kwds...)
end


function _surface(p, T, v; resolution=(400,400), kwds...)
    T_mat = Array{Int}(undef, length(T), 3)
    for k = 1:length(T)
        T_mat[k,:] .= T[k]
    end
    # s = Scene(resolution=resolution)
    mesh(first.(p), last.(p), vec(v), T_mat; kwds...)
end



function _mesh!(s, p, T, v; kwds...)
    T_mat = Array{Int}(undef, length(T), 3)
    for k = 1:length(T)
        T_mat[k,:] .= T[k]
    end
    mesh!(s, [first.(p) last.(p)], T_mat; color=v, kwds...)
end

function meshdata(f::Fun{<:PiecewiseSpace})
    pTv = MultivariateTriangle.meshdata.(components(f))
    p = vcat(first.(pTv)...)
    T = pTv[1][2]
    cs = length(pTv[1][1])
    for k = 2:length(pTv)
        append!(T, (t -> (cs.+t)).(pTv[k][2]))
        cs += length(pTv[k][1])
    end

    v = vcat(last.(pTv)...)

    p, T, v
end

function meshdata(f::Fun{<:TensorSpace{<:Tuple{<:Chebyshev,<:Chebyshev}}})
    p = points(f)
    v = values(f)
    n = length(p)
    T = Vector{NTuple{3,Int}}()
    d_x,d_y = factors(domain(f))
    a_x,b_x = endpoints(d_x)
    a_y,b_y = endpoints(d_y)
    if iseven(_padua_length(n))
        l = floor(Int, (1+sqrt(1+8n))/4)

        push!(p, Vec(b_x,b_y))
        push!(p, Vec(a_x,b_y))

        push!(v, f(b_x,b_y))
        push!(v, f(a_x,b_y))

        for p = 0:l-2
            for k = (2p*l)+1:(2p*l)+l-1
                push!(T, (k+1, k, l+k+1))
            end
            for k = (2p*l)+1:(2p*l)+l-1
                push!(T, (k, l+k, l+k+1))
            end
            for k = (2p*l)+l+1:(2p*l)+2l-1
                push!(T, (k+1, k, l+k))
            end
            for k = (2p*l)+l+2:(2p*l)+2l
                push!(T, (k, k+l-1, l+k))
            end
        end
        for p=0:l-3
            push!(T, ((2p+1)*l+1, (2p+2)*l+1, (2p+3)*l+1))
        end
        for p =0:l-2
            push!(T, ((2p+1)*l, (2p+2)*l, (2p+3)*l))
        end
        push!(T, (1, n+1, l+1))
        push!(T, (n-2l+1, n+2, n-l+1))
    else
        l = floor(Int, (3+sqrt(1+8n))/4)

        push!(p, Vec(a_x,b_y))
        push!(p, Vec(a_x,a_y))

        push!(v, f(a_x,b_y))
        push!(v, f(a_x,a_y))

        for p = 0:l-2
            for k = p*(2l-1)+1:p*(2l-1)+l-1
                push!(T, (k+1, k, l+k))
            end
            for k = p*(2l-1)+1:p*(2l-1)+l-2
                push!(T, (k+1, l+k, l+k+1))
            end
        end
        for p = 0:l-3
            for k = p*(2l-1)+l+1:p*(2l-1)+2l-2
                push!(T, (k+1, k, l+k))
            end
            for k = p*(2l-1)+l+1:p*(2l-1)+2l-1
                push!(T, (k, k+l-1, l+k))
            end
        end

        for p=0:l-3
            push!(T, (p*(2l-1) + 1, p*(2l-1) + l+1, p*(2l-1) + 2l))
        end

        for p=0:l-3
            push!(T, (p*(2l-1) + l, p*(2l-1) + 2l-1, p*(2l-1) + 3l-1))
        end

        push!(T, (n-2l+2, n+1, n-l+2))
        push!(T, (n-l+1, n+2, n))
    end

    p, T, v
end



meshdata(f) =
    triangle_meshdata(points(f), values(f), (domain(f).a, domain(f).b, domain(f).c),
                                            f.((domain(f).a, domain(f).b, domain(f).c)))

function triangle_meshdata(p, v, (a, b, c), (fa, fb, fc))
    n = length(p)
    T = Vector{NTuple{3,Int}}()


    if iseven(_padua_length(n))
        l = floor(Int, (1+sqrt(1+8n))/4)

        push!(p, b)
        push!(p, c)

        push!(v, fb)
        push!(v, fc)

        for p = 0:l-2
            for k = (2p*l)+1:(2p*l)+l-1
                push!(T, (k+1, k, l+k+1))
            end
            for k = (2p*l)+1:(2p*l)+l-1
                push!(T, (k, l+k, l+k+1))
            end
            for k = (2p*l)+l+1:(2p*l)+2l-1
                push!(T, (k+1, k, l+k))
            end
            for k = (2p*l)+l+2:(2p*l)+2l
                push!(T, (k, k+l-1, l+k))
            end
        end
        for p=0:l-3
            push!(T, ((2p+1)*l+1, (2p+2)*l+1, (2p+3)*l+1))
        end
        for p =0:l-2
            push!(T, ((2p+1)*l, (2p+2)*l, (2p+3)*l))
        end
        push!(T, (1, n+1, l+1))
        push!(T, (n-2l+1, n+2, n-l+1))
    else
        l = floor(Int, (3+sqrt(1+8n))/4)

        push!(p, Vec(c))
        push!(p, Vec(a))

        push!(v, fc)
        push!(v, fa)

        for p = 0:l-2
            for k = p*(2l-1)+1:p*(2l-1)+l-1
                push!(T, (k+1, k, l+k))
            end
            for k = p*(2l-1)+1:p*(2l-1)+l-2
                push!(T, (k+1, l+k, l+k+1))
            end
        end
        for p = 0:l-3
            for k = p*(2l-1)+l+1:p*(2l-1)+2l-2
                push!(T, (k+1, k, l+k))
            end
            for k = p*(2l-1)+l+1:p*(2l-1)+2l-1
                push!(T, (k, k+l-1, l+k))
            end
        end

        for p=0:l-3
            push!(T, (p*(2l-1) + 1, p*(2l-1) + l+1, p*(2l-1) + 2l))
        end

        for p=0:l-3
            push!(T, (p*(2l-1) + l, p*(2l-1) + 2l-1, p*(2l-1) + 3l-1))
        end

        push!(T, (n-2l+2, n+1, n-l+2))
        push!(T, (n-l+1, n+2, n))
    end

    p, T, v
end


P = JacobiTriangle()
f = expand(P, splat((x,y) -> cos(x*exp(y))))
(a,b,c) = (SVector(0.,0.), SVector(0.,1.), SVector(1.,0.))
triangle_meshdata(plotgridvalues(f)..., (a,b,c), getindex.(Ref(f), (a,b,c)))



# end # module
