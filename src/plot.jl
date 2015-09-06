## Disk specific
function plot{S,V,SS<:DiskSpace}(f::ProductFun{S,V,SS};opts...)
    x,y,vals=points(f,1),points(f,2),real(values(f))
    surf([x x[:,1]],[y y[:,1]],[vals vals[:,1]];opts...)
end

