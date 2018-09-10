Base.show(io::IO,d::Triangle) = print(io, "Triangle($(d.a),$(d.b),$(d.c))")
function Base.show(io::IO,s::JacobiTriangle)
    if domain(s) == Triangle()
        print(io, "JacobiTriangle($(s.α),$(s.β),$(s.γ))")
    else
        print(io, "JacobiTriangle($(s.α),$(s.β),$(s.γ),$(domain(s)))")
    end
end


function Base.show(io::IO,s::TriangleWeight)
    d=domain(s)
    #TODO: Get shift and weights right
    s.α ≠ 0 && print(io,"x^$(s.α)")
    s.β ≠ 0 && print(io,"y^$(s.β)")
    s.γ ≠ 0 && print(io,"(1-x-y)^$(s.γ)")

    print(io,"[")
    show(io,s.space)
    print(io,"]")
end
