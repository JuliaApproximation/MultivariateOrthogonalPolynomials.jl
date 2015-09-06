# DiskFun

[![Build Status](https://travis-ci.org/dlfivefifty/DiskFun.jl.svg?branch=master)](https://travis-ci.org/dlfivefifty/DiskFun.jl)



The following solves Poisson `Δu =f` with zero Dirichlet conditions
on a disk

```julia
d = Disk()
f = Fun((x,y)->exp(-10(x+.2)^2-20(y-.1)^2),d) 
u = [dirichlet(d);lap(d)]\Any[0.,f]
ApproxFun.plot(u)                           # Requires Gadfly or PyPlot
```


The following solves beam equation `u_tt + Δ^2u = 0`
on a disk

```julia
d = Disk()
u0 = Fun((x,y)->exp(-50x^2-40(y-.1)^2)+.5exp(-30(x+.5)^2-40(y+.2)^2),d)
B= [dirichlet(d),neumann(d)]
L = -lap(d)^2
h = 0.001
timeevolution(2,B,L,u0,h)                 # Requires GLPlot
```
