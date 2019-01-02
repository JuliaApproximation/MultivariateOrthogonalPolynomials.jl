using ApproxFun, MultivariateOrthogonalPolynomials
import ApproxFun: Vec, PiecewiseSegment, ZeroOperator
    import MultivariateOrthogonalPolynomials: DirichletTriangle

d = [Triangle(Vec(0,0), Vec(1,0), Vec(0,1)) , Triangle(Vec(1,1),Vec(1,0),Vec(0,1)) ,
    Triangle(Vec(1,1),Vec(0,1),Vec(0,2)) , Triangle(Vec(1,2),Vec(0,2),Vec(1,1)) ,
    Triangle(Vec(1,0), Vec(2,0), Vec(1,1)) , Triangle(Vec(2,1), Vec(1,1), Vec(2,0))]

∂d = components(PiecewiseSegment([Vec(0,0), Vec(1,0), Vec(2,0), Vec(2,1), Vec(1,1), Vec(1,2), Vec(0,2), Vec(0,1), Vec(0,0)]))
ιd = [Segment(Vec(0,1),Vec(1,0)), Segment(Vec(0,1), Vec(1,1)), Segment(Vec(1,1), Vec(2,0)),
        Segment(Vec(1,0), Vec(1,1)), Segment(Vec(0,2), Vec(1,1))]

length(∂d)

ds = vcat(fill.(DirichletTriangle{1,1,1}.(d),3)...) # repeat each triangle 3 times
rs = [Legendre.(∂d); fill.(Legendre.(ιd),2)...; fill.(JacobiTriangle.(d),3)...]


N,M = length(rs), length(ds)
    A = Matrix{Operator{Float64}}(undef, N,M)
    for K=1:N, J=1:M # fill with zeros
        A[K,J] = ZeroOperator(ds[J],rs[K])
    end
    # add boundary conditions
    for K = 1:length(∂d)
        for J = 1:length(d)
            if
        A[K,J] = I : ds[J] → rs[K]
    end


∂d[1] ⊆ d[1]

rs[3]



typeof(components(d))




length(ιd)




Vec{2,Float64}
Segment(Vec(0,0), Vec(1.0,0)) |> typeof



import Makie



S = DirichletTriangle{1,1,1}.(components(d))

C1 = I : S[1] → Legendre(Segment(Vec(0,0),Vec(1,0)))
C2 = I : S[1] → Legendre(Segment(Vec(1,0),Vec(0,0)))

a = Legendre(Segment(Vec(0,0),Vec(1,0)))
b = Legendre(Segment(Vec(1,0),Vec(0,0)))

@which Conversion(a,b)




DirichletTriangle{0,1,1}(d[1])

2
C1.op.op.ops




C2.op.op.ops

C1 == C2

C1[1:20,1:20] == C2[1:20,1:20]

Legendre(Segment(Vec(0,0),Vec(1,0)))

∂d = PiecewiseSegment([Vec(0.,0), Vec(1.,0), Vec(1,1), Vec(1,2), Vec(0,1), Vec(-1,1.5), Vec(0,0)])

o = Fun.(S, Ref([1.0]))

o[1]
