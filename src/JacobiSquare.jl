export WeightedSquare, JacobiSquare


# represents function of the form r^m g(r^2)T
# as r^m P_k^{a,b}(1-2r^2)
# when domain is Segment(1,0)

immutable WeightedSquare{S} <: RealUnivariateSpace{Segment{Float64}}
    m::Float64
    space::S
end
WeightedSquare(m::Number,S)=WeightedSquare{typeof(S)}(m,S)

Base.promote_rule{S1,S2}(::Type{WeightedSquare{S1}},::Type{WeightedSquare{S2}})=WeightedSquare{promote_type(S1,S2)}
Base.convert{S1}(::Type{WeightedSquare{S1}},W::WeightedSquare)=WeightedSquare{S1}(W.m,W.space)

typealias JacobiSquare WeightedSquare{Jacobi{Float64,Segment{Float64}}}


JacobiSquare(m,d::Domain)=WeightedSquare(m,Jacobi(m,0,d))
JacobiSquare(m,a::Number,b::Number,d::Domain)=WeightedSquare(m,Jacobi(a,b,d))
JacobiSquare(m,a::Number,b::Number)=WeightedSquare(m,Jacobi(a,b,Segment(1.,0.)))
JacobiSquare(m)=JacobiSquare(m,Segment(1.,0.))


domain(A::WeightedSquare)=domain(A.space)


spacescompatible(A::WeightedSquare,B::WeightedSquare)=A.m==B.m && spacescompatible(A.space,B.space)

canonicalspace(S::WeightedSquare)=S


# We assume domains is [1.,0.]

points(S::WeightedSquare,n)=sqrt(points(S.space,n))
checkpoints(S::WeightedSquare)=sqrt(checkpoints(S.space))

plan_transform(S::WeightedSquare,vals::Vector)=(S,points(S,length(vals)),plan_transform(S.space,vals))

function transform(S::WeightedSquare,vals::Vector,plan)
    if S.m == 0
        transform(S.space,vals,plan[3])
    else
        transform(S.space,plan[2].^(-S.m).*vals,plan[3])
    end
end


plan_itransform(S::WeightedSquare,vals::Vector)=(points(S,length(vals)),plan_itransform(S.space,vals))
itransform(S::WeightedSquare,cfs::Vector,plan)=isempty(cfs)?cfs:plan[1].^S.m.*itransform(S.space,cfs,plan[2])

evaluate(f::AbstractVector,sp::WeightedSquare,x)=x.^sp.m.*evaluate(f,sp.space,x.^2)


## Operators

# Override JacobiWeight default
function Multiplication{DD}(f::Fun{JacobiWeight{Chebyshev{DD},DD}},ds::WeightedSquare)
    @assert ncoefficients(f)==1
    @assert f.space.α ==0.
    @assert isinteger(f.space.β)
    rs=WeightedSquare(ds.m+round(Int,f.space.β),ds.space)
    MultiplicationWrapper(f,SpaceOperator(ConstantOperator(2.0^f.space.β*f.coefficients[1]),ds,rs))
end


function Derivative(S::WeightedSquare)
     m=S.m
     d=domain(S)
     @assert d==Segment(1.,0.)

     JS=S.space
     D=Derivative(JS)
     M=Multiplication(2Fun(identity,d),rangespace(D))
     op=TimesOperator(M,D)
     if m==0
      # we have D[ f(r^2)] = 2r f'(r^2) = 2 r^(-1)*r^2 f'(r^2)
         DerivativeWrapper(SpaceOperator(op,S,WeightedSquare(-1,rangespace(op))),1)
     else
     # we have D[r^m f(r^2)] = r^{m-1} (m f(r^2) + 2r^2 f'(r^2))
        op=m+TimesOperator(M,D)
        DerivativeWrapper(SpaceOperator(op,S,WeightedSquare(m-1,rangespace(op))),1)
    end
end


function Derivative(S::WeightedSquare,k::Integer)
     if k==1
         Derivative(S)
     else
         D=Derivative(S)
         DerivativeWrapper(TimesOperator(Derivative(rangespace(D),k-1).op,D.op),k)
     end
end



# return the space that has banded Conversion to the other
conversion_rule(A::WeightedSquare,B::WeightedSquare)=WeightedSquare(max(A.m,B.m),conversion_type(A.space,B.space))
maxspace_rule(A::WeightedSquare,B::WeightedSquare)=WeightedSquare(min(A.m,B.m),maxspace(A.space,B.space))


##TODO:ConversionWrapper

function Conversion(A::WeightedSquare,B::WeightedSquare)
    if A.m==B.m
        ConversionWrapper(SpaceOperator(Conversion(A.space,B.space),A,B))
    else
        df=round(Int,A.m-B.m)
        @assert A.m > B.m &&  iseven(df)
        r=Fun(identity,domain(B))
        M=Multiplication(r.^div(df,2),B.space) #this is multiplication by r^(2*p)
        ConversionWrapper(SpaceOperator(TimesOperator(M,Conversion(A.space,B.space)),A,B))
    end
end




function Base.getindex{WS<:WeightedSquare}(op::Evaluation{WS,Bool},kr::Range)
    @assert !op.x && op.order <= 1
    m=op.space.m
    js=op.space.space
    if op.order ==0
        Evaluation(js,false,0)[kr]
    elseif m==0
        2Evaluation(js,false,1)[kr]
    else
        2Evaluation(js,false,1)[kr]+m*Evaluation(js,false,0)[kr]
    end
end





# represents
# D^+ = (D -m/r)/sqrt(2)
function DDp(S)
    m=S.m
    D=Derivative(setdomain(S.space,Segment()))
    rs=JacobiSquare(m+1,S.space.a+1,S.space.b+1)
    SpaceOperator(-4/sqrt(2)*D,S,rs)
end

# represents
# D^- = (D -m/r)/sqrt(2)
function DDm(S)
    m=S.m
    @assert S.space.a==m
    SD=JacobiSD(true,setdomain(S.space,Segment()))
    rs=JacobiSquare(m-1,S.space.a-1,S.space.b+1)
    SpaceOperator(2/sqrt(2)*SD,S,rs)
end




## jacobi square special


function transform(S::JacobiSquare,vals::Vector,sxw)
    @assert isapproxinteger(S.m)
    m=round(Int,S.m);a=S.space.a;b=S.space.b
    s,x,xw=sxw


    if S == s
        if S.m == 0
            return transform(S.space,vals,xw)
        else
            return transform(S.space,x.^(-m).*vals,xw)
        end
    end

    x,w=xw.points,xw.weights

    n=length(vals)


    if m==s.m==0
        V=jacobip(0:n-1,a,b,x)'
        nrm=(V.^2)*w
        (V*(w.*vals))./nrm
    elseif m==a && b==0
        ## Same as jacobitransform.jl

        w2=(1-x).^(m/2)
        mw=w2.*w
        V=jacobip(0:n-div(m,2)-1,a,b,x)'
        nrm=(V.^2)*(w2.*mw)
        (V*(mw.*vals))./nrm*2^(m/2)
    else
        error("transform only implemented for first case")
    end
end
