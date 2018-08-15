

export DiskSpace



struct DiskSpace{m,a,b,JS,S} <: AbstractProductSpace{Tuple{JS,S},ComplexF64,2}
    domain::Disk
    spacet::S
    DiskSpace(d,sp)=new(d,sp)
    DiskSpace(d::AnyDomain)=new(Disk(d),S())
end


DiskSpace(D::Disk,S::Space)=DiskSpace{0,0,0,JacobiSquare,typeof(S)}(D,S)
DiskSpace(D::Disk)=DiskSpace(D,Laurent())
DiskSpace(d::AnyDomain)=DiskSpace(Disk(d))
DiskSpace()=DiskSpace(Disk())

canonicalspace(D::DiskSpace)=D

spacescompatible{m,a,b,JS,S}(A::DiskSpace{m,a,b,JS,S},B::DiskSpace{m,a,b,JS,S})=true

coefficient_type{T<:Complex}(::DiskSpace,::Type{T})=T
coefficient_type{T<:Real}(::DiskSpace,::Type{T})=Complex{T}

domain(d::DiskSpace)=d.domain
function space(D::DiskSpace,k::Integer)
    @assert k==2
    D.spacet
end

Base.getindex(D::DiskSpace,k::Integer)=space(D,k)

Space(D::Disk)=DiskSpace(D)


columnspace{M,a,b,SS}(D::DiskSpace{M,a,b,SS},k)=(m=div(k,2);JacobiSquare(M+m+0.,a+m+0.,b+0.,Segment(D.domain.radius,0.)))

#transform(S::DiskSpace,V::Matrix)=transform([columnspace(S,k) for k=1:size(V,2)],S.spacet,V)


evaluate(f::AbstractVector,sp::DiskSpace,x...) =
    evaluate(ProductFun(Fun(sp,f)),x...)



function Base.real{JS,DD}(f::ProductFun{JS,Laurent{DD},DiskSpace{0,0,0,JS,Laurent{DD}}})
    cfs=f.coefficients
    n=length(cfs)

    ret=Array(Fun{JS,Float64},iseven(n) ? n+1 : n)
    ret[1]=real(cfs[1])

    for k=2:2:n
        # exp(1im(k-1)/2*x)=cos((k-1)/2 x) +i sin((k-1)/2 x)
        ret[k]=imag(cfs[k])
        ret[k+1]=real(cfs[k])
    end
    for k=3:2:n
        # exp(1im(k-1)/2*x)=cos((k-1)/2 x) +i sin((k-1)/2 x)
        ret[k]+=real(cfs[k])
        ret[k-1]-=imag(cfs[k])
    end

    ProductFun(ret,DiskSpace{0,0,0,JS,Fourier}(space(f).domain,Fourier()))
end




## Conversion
# These are placeholders for future

conversion_rule{m,a,b,m2,a2,b2,JS,FS}(A::DiskSpace{m,a,b,JS,FS},
                                      B::DiskSpace{m2,a2,b2,JS,FS})=DiskSpace{max(m,m2),min(a,a2),min(b,b2),JS,FS}(A.domain,B.spacet)

function coefficients{m,a,b,m2,a2,b2,JS,FS}(cfs::Vector,
                                            A::DiskSpace{m,a,b,JS,FS},
                                          B::DiskSpace{m2,a2,b2,JS,FS})
    g=ProductFun(Fun(A,cfs))
    rcfs=Fun{typeof(columnspace(B,1)),eltype(cfs)}[Fun(g.coefficients[k],columnspace(B,k)) for k=1:length(g.coefficients)]
    Fun(ProductFun(rcfs,B)).coefficients
end


# function coefficients{S,V,SS,T}(f::ProductFun{S,V,SS,T},sp::ProductRangeSpace)
#     @assert space(f,2)==space(sp,2)

#     n=min(size(f,2),length(sp.S))
#     F=[coefficients(f.coefficients[k],rangespace(sp.S.Rdiags[k])) for k=1:n]
#     m=mapreduce(length,max,F)
#     ret=zeros(T,m,n)
#     for k=1:n
#         ret[1:length(F[k]),k]=F[k]
#     end
#     ret
# end


## Operators

isfunctional{DS<:DiskSpace}(D::Dirichlet{DS},k)=k==1

isdiagop{DS<:DiskSpace}(L::Dirichlet{DS},k)=k==2
diagop{DS<:DiskSpace}(D::Dirichlet{DS},col)=Evaluation(columnspace(domainspace(D),col),false,D.order)


isdiagop{DS<:DiskSpace}(L::Laplacian{DS},k)=k==2
function diagop{DS<:DiskSpace}(L::Laplacian{DS},col)
    csp=columnspace(domainspace(L),col)
    rsp=columnspace(rangespace(L),col)
    Dt=Derivative(space(domainspace(L),2))
    c=Dt[col,col]


    r=Fun(identity,[domain(L).radius,0.])
    D=Derivative(csp)
    Δ=D^2+(1/r)*D+Multiplication((c/r)^2,csp)

    Δ^L.order
end


Conversion(a::DiskSpace,b::DiskSpace) =
    ConcreteConversion(a,b)

isproductop{DS1<:DiskSpace,DS2<:DiskSpace}(C::ConcreteConversion{DS1,DS2})=true
isdiagop{DS1<:DiskSpace,DS2<:DiskSpace}(C::ConcreteConversion{DS1,DS2},k)=k==2
diagop{DS1<:DiskSpace,DS2<:DiskSpace}(C::ConcreteConversion{DS1,DS2},col)=Conversion(columnspace(domainspace(C),col),
                                                                               columnspace(rangespace(C),col))
#deprod{DS1<:DiskSpace,DS2<:DiskSpace}(C::Conversion{DS1,DS2},k,::Colon)=ConstantOperator(1.0)


lap(d::Disk)=Laplacian(Space(d))
dirichlet(d::Disk)=Dirichlet(Space(d))
neumann(d::Disk)=Neumann(Space(d))

lap(d::DiskSpace)=Laplacian(d)
dirichlet(d::DiskSpace)=Dirichlet(d)
neumann(d::DiskSpace)=Neumann(d)



Laplacian(S::DiskSpace,k::Integer) =
    ConcreteLaplacian{typeof(S),Int,BandedMatrix{eltype(S)}}(S,k)

function rangespace{m,a,b,JS,S}(L::ConcreteLaplacian{DiskSpace{m,a,b,JS,S}})
    sp=domainspace(L)
    DiskSpace{m-2L.order,a+2L.order,b+2L.order,JS,S}(sp.domain,sp.spacet)
end



# special case of integer modes
function diagop{b}(L::ConcreteLaplacian{DiskSpace{0,0,b,JacobiSquare,Laurent}},col)
    S=columnspace(domainspace(L),col)
    Dp=DDp(S)
    Dm=DDm(rangespace(Dp))
    2Dm*Dp
end



function rangespace{b,JS,S}(L::ConcreteLaplacian{DiskSpace{0,0,b,JS,S}})
    sp=domainspace(L)
    DiskSpace{0,0,b+2,JS,S}(sp.domain,sp.spacet)
end
