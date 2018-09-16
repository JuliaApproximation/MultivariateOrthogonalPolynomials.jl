include("helmholtzhodge.jl")

using Random
Random.seed!(0)

function sphrandn(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, 2n-1)
    for i = 1:m
        A[i,1] = randn(T)
    end
    for j = 1:n-1
        for i = 1:m-j
            A[i,2j] = randn(T)
            A[i,2j+1] = randn(T)
        end
    end
    A
end

#=
N = 2 .^(5:13)
t = zeros(length(N), 2)
err = zeros(length(N))

j = 1
for n in N
    println("This is n: ", n)
    U1 = sphrandn(Float64, n, n)
    U2 = sphrandn(Float64, n, n)
    U1[1] = 0
    U2[1] = 0
    Us = zero(U1); Ut = zero(U1);
    V1 = zero(U1); V2 = zero(U1); V3 = zero(U1); V4 = zero(U1);

    HH = HelmholtzHodge(Float64, n)
    t[j, 1] = @elapsed for k = 1:10
        HelmholtzHodge(Float64, n)
    end
    t[j, 1] /= 10

    gradient!(U1, V1, V2)
    curl!(U2, V3, V4)

    V5 = V1 + V3
    V6 = V2 + V4

    helmholtzhodge!(HH, Us, Ut, V5, V6)

    t[j, 2] = @elapsed for k = 1:10
        helmholtzhodge!(HH, Us, Ut, V5, V6)
    end
    t[j, 2] /= 10
    global j += 1
end

j = 1
for n in N
    println("This is n: ", n)
    HH = HelmholtzHodge(Float64, n)
    for k = 1:10
        U1 = sphrandn(Float64, n, n)
        U2 = sphrandn(Float64, n, n)
        U1[1] = 0
        U2[1] = 0
        Us = zero(U1); Ut = zero(U1);
        V1 = zero(U1); V2 = zero(U1); V3 = zero(U1); V4 = zero(U1);

        gradient!(U1, V1, V2)
        curl!(U2, V3, V4)

        V5 = V1 + V3
        V6 = V2 + V4

        helmholtzhodge!(HH, Us, Ut, V5, V6)

        println("This is the spheroidal relative backward error: ", norm(Us-U1)/norm(U1))
        println("This is the toroidal relative backward error: ", norm(Ut-U2)/norm(U2))
        err[j] += norm(Us-U1)/norm(U1) + norm(Ut-U2)/norm(U2)
    end
    err[j] /= 10
    global j += 1
end
=#

using PyPlot

cd("/Users/Mikael/Dropbox/Helmholtz")

t = [0.000301073 8.59417e-5; 0.000870661 0.000313344; 0.00371459 0.0013411; 0.0158751 0.00519203; 0.0629211 0.0214159; 0.272275 0.0841661; 1.18461 0.338013; 5.22898 1.34888; 20.2509 5.41399]
err = [9.73175783254202e-16, 1.1848048133640504e-15, 1.2864591012075482e-15, 1.4887354817191668e-15, 2.0250488931546365e-15, 2.714940169354596e-15, 3.426270329252532e-15, 4.804155288069517e-15, 6.620111175740055e-15]

clf()
loglog(N.-1, t[:, 1], "xk", N.-1, t[:, 2], "+k", N.-1, (N.-1).^2/2e6, "-k")
xlabel("Degree \$n\$"); ylabel("Execution Time (s)"); grid()
legend(["Pre-computation","Execution","\$\\mathcal{O}(n^2)\$"])
savefig("helmholtzhodgetime.pdf")

clf()
loglog(N.-1, err, "xk", N.-1, sqrt.(N.*log.(N))*eps()/7, "-k")
xlabel("Degree \$n\$"); ylabel("Relative Error"); grid()
ylim((6e-16,1.2e-14))
legend(["Error","\$\\mathcal{O}(\\sqrt{n \\log(n)}\\epsilon)\$"])
savefig("helmholtzhodgeerror.pdf")
