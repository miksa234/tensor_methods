using LinearAlgebra
using LaTeXStrings
using Distributions
using Plots

include("../../sesh2/src/functions.jl")
include("../../sesh3/src/functions.jl")

function errplot(x, phi, d_phi_2, n)
    p = plot(x, phi,
             lw=3,
             titlefontsize=20,
             xlabelfontsize=14,
             ylabelfontsize=14,
             dpi=300,
             grid=false,
             size=(500, 400))
    plot!(p, x, d_phi_2,
          lw=3,
          title="n=$n",
          label=L"\nabla \frac{1}{2} \phi^2")
    savefig(p, "./plots/err_$n.png")
end

function main()
    nr = [(2, 7), (3, 23), (4, 49)]

    for (n, r) in nr
        V = [reshape(vcat([normalize(rand(-1:1, n^2)) for i=1:r]...), (n^2, r)) for _=1:3]  # guess
        T, U = multiplication_tensor(n)                                                     # given n^3 CPD
        V_hat, phi, d_phi_2 = cp_als(U, V, 10000)                                            # CP-ALS + err
        x = collect(1:length(phi))                                                          # for plot

        errplot(x, phi, d_phi_2, n)
    end
end

main()
