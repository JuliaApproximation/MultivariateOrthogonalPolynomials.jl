## Disk specific
# function Plots.surface{S<:UnivariateSpace,
#                        V<:UnivariateSpace,
#                        SS<:DiskSpace}(f::ProductFun{S,V,SS};opts...)
#     x,y,vals=points(f,1),points(f,2),real(values(f))
#     surface([x x[:,1]],[y y[:,1]],[vals vals[:,1]];opts...)
# end
#

@recipe function f( ◣::Triangle)
    seriestype := :shape
    [ ◣.a[1], ◣.b[1], ◣.c[1], ◣.a[1]],[ ◣.a[2], ◣.b[2], ◣.c[2], ◣.a[2]]
end
