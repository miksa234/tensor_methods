#/usr/bin/julia

using LinearAlgebra
using Plots
using Distributions
using LaTeXStrings
using Random
using TSVD: tsvd # todo: implement
using TensorToolbox: tenmat, ttm, contract # todo:implement
using Plots.Measures

include("mps-tt.jl")
include("tucker-hosvd.jl")
include("unf-aprx.jl")
include("functions.jl")

function main()
    # Exercise 2
    d = 4;
    n = [20+k for k=1:d];
    r = [2*k for k=1:d];
    V = [rand(Uniform(-1, 1), n[k], r[k]) for k=1:d];
    S = rand(Uniform(-1, 1), r...);
    C = tucker_eval(S, V);
    Vs, S0, vals, errors = my_hosvd(C, r);

    p = plot(errors,
            ylabel="error",
            label=L"$\frac{\|\hat{A}_k - A\|_F}{\|A\|_F}$",
            xlabel=L"k",
            lw = 1,
            xticks=collect(1:d),
            markershape=:circle,
            legend=:topleft,
            margin=5mm,
            dpi=300,
            title="HOSVD of Uniform Tensor")
    savefig(p, "./plots/hosvd-uniform-error.png")

    # Exercise 3
    d = 4; n = 51;
    A = [f([t1, t2, t3, t4]) for t1 in t(n), t2 in t(n), t3 in t(n), t4 in t(n)];
    B = [g([t1, t2, t3, t4]) for t1 in t(n), t2 in t(n), t3 in t(n), t4 in t(n)];

    # Exercise 3a
    σ_unf_a = σ_unfold(A, d);
    σ_unf_b = σ_unfold(B, d);
    p1 = plot(margin=5mm)
    p2 = plot(margin=5mm)
    for k=1:d
        plot!(p1,
            σ_unf_a[k],
            ylabel=L"$\log(\Sigma_{A_k})$",
            label=L"$\Sigma_{A_%$k}$",
            xlabel=L"n",
            lw = 1,
            yaxis=:log,
            markershape=:circle,
            dpi=300,
            title="Unfolding of A")
        plot!(p2,
            σ_unf_b[k],
            ylabel=L"$\log(\Sigma_{B_k})$",
            label=L"$\Sigma_{_B%$k}$",
            xlabel=L"n",
            lw = 1,
            dpi=300,
            markershape=:circle,
            yaxis=:log,
            title="Unfolding of B")
    end
    savefig(p1, "./plots/singular-dec-A.png")
    savefig(p2, "./plots/singular-dec-B.png")

    _s = [1/(10^(j*2)) for j=1:5] # computer not goode enough for j = 6

    r_jk_a, ϵ_jk_a, σ_jk_a = rank_approx(A, tucker_unfold);
    r_jk_b, ϵ_jk_b, σ_jk_b = rank_approx(B, tucker_unfold);
    p1 = plot(size=[700, 350], margin=5mm, dpi=300)
    p2 = plot(size=[700, 350], margin=5mm, dpi=300)
    for k=1:d
        plot!(p1,
            1 ./ ϵ_s,
            r_jk_a[:, k],
            ylabel=L"$r_{jk}$",
            label=L"A-$\varepsilon_{j%$k}$",
            xlabel=L"\varepsilon_{j}^{-1}",
            lw = 2,
            markershape=:circle,
            xaxis=:log10,
            legend=:topleft,
            title=L"Tensor $f(x_1, x_2, x_3, x_4)$")
        plot!(p2,
            1 ./ ϵ_s,
            r_jk_b[:, k],
            ylabel=L"$r_{jk}$",
            label=L"B-$\varepsilon_{j%$k}$",
            xlabel=L"\varepsilon_{j}^{-1}",
            lw = 2,
            markershape=:circle,
            xaxis=:log10,
            legend=:topleft,
            title=L"Tensor $g(x_1, x_2, x_3, x_4)$")
    end
    savefig(p1, "./plots/hosvd-error-A.png")
    savefig(p2, "./plots/hosvd-error-B.png")

    # Exercise 3b
    Nj_hosvd_a = []; ϵ_hosvd_a = []; σ_hosvd_a = [];
    Nj_hosvd_b = []; ϵ_hosvd_b = []; σ_hosvd_b = [];

    println("Checking validity of error analysis for HOSVD")
    for j=1:length(ϵ_s)
        S0_a, Vs_a, vals_a, errors_a = my_hosvd(A, r_jk_a[j, :])
        N_vals = 0
        for V in Vs_a
            N_vals += length(V)
        end
        append!(Nj_hosvd_a, N_vals + length(S0_a))
        append!(ϵ_hosvd_a, errors_a[end])
        append!(σ_hosvd_a, [vals_a])
        println("ϵ_hosvd_a[j] <= |ϵ_jk_a|_f : $(ϵ_hosvd_a[j] <= norm(ϵ_jk_a)) ϵ_hosvd_a[j] = $(ϵ_hosvd_a[j])")

        S0_b, Vs_b, vals_b, errors_b = my_hosvd(B, r_jk_b[j, :])
        N_vals = 0
        for V in Vs_b
            N_vals += length(V)
        end
        append!(Nj_hosvd_b, N_vals + length(S0_b))
        append!(ϵ_hosvd_b, errors_b[end])
        append!(σ_hosvd_b, [vals_b])
        println("ϵ_hosvd_b[j] <= |ϵ_jk_b|_f : $(ϵ_hosvd_b[j] <= norm(ϵ_jk_b)) ϵ_hosvd_b[j] = $(ϵ_hosvd_b[j])")
    end

    # Exercise 3b
    j = 5
    p1 = plot(dpi=300, size=[800, 350], margin=5mm)
    p2 = plot(dpi=300, size=[800, 350], margin=5mm)
    for k=1:d
        plot!(p1, σ_unf_a[k][1:r_jk_a[j, k]] ./ σ_hosvd_a[j][k],
            ylabel=L"\frac{\sigma^{A_k}_\alpha}{\sigma^{hosvd}_{k\alpha}}",
            label=L"k=%$k",
            xlabel=L"j",
            xticks=collect(1:length(r_jk_a)),
            yticks=[1, 1.001],
            ylim=[1, 1.001],
            lw = 2,
            markershape=:circle,
            legend=:topleft,
            title=L"$f(x_1, x_2, x_3, x_4)$")
        plot!(p2, σ_unf_b[k][1:r_jk_b[j, k]] ./ σ_hosvd_b[j][k],
            ylabel=L"\frac{\sigma^{B_k}_\alpha}{\sigma^{hosvd}_{k\alpha}}",
            label=L"k=%$k",
            xlabel=L"j",
            yticks=[1, 1.001],
            ylim=[1, 1.001],
            lw = 2,
            markershape=:circle,
            legend=:topleft,
            title=L"$g(x_1, x_2, x_3, x_4)$")
    end
    savefig(p1, "./plots/hosvd-sigmaratio-a.png")
    savefig(p2, "./plots/hosvd-sigmaratio-b.png")

    # Exercise 3d

    p1 = plot(size=[700, 350], margin=5mm, dpi=300)
    p2 = plot(size=[700, 350], margin=5mm, dpi=300)
    plot!(p1,
        1 ./ ϵ_s,
        Nj_hosvd_a,
        ylabel=L"$N_{j}$",
        label=L"A-$N_{j}$",
        xlabel=L"\varepsilon_{j}^{-1}",
        lw = 2,
        markershape=:circle,
        xaxis=:log10,
        legend=:topleft,
        title=L"Tucker of $f(x_1, x_2, x_3, x_4)$")
    plot!(p2,
        1 ./ ϵ_s,
        Nj_hosvd_b,
        ylabel=L"$N_{j}$",
        label=L"B-$N_{j}$",
        xlabel=L"\varepsilon_{j}^{-1}",
        lw = 2,
        markershape=:circle,
        xaxis=:log10,
        legend=:topleft,
        title=L"Tucker of $g(x_1, x_2, x_3, x_4)$")
    savefig(p1, "./plots/hosvd-Nj-a.png")
    savefig(p2, "./plots/hosvd-Nj-b.png")

    # exercise 6
    d = 4;
    n = [20+k for k=1:d];
    r = [[2*k for k=1:(d-1)]..., 1];
    r_new = [1, r...]
    V = [rand(Uniform(-1, 1), (r_new[i], n[i], r_new[i+1])) for i=1:d];
    D = ttmps_eval(V, n)
    C, singular_val, errors= tt_svd(D, n, r, d);
    p = plot(errors,
             ylabel="error",
             label=L"$\frac{\|\hat{A}_k - A\|_f}{\|A\|_f}$",
             xlabel=L"k",
             lw = 1,
             xticks=collect(1:d),
             markershape=:circle,
             legend=:topleft,
             margin=5mm,
             dpi=300,
             title="TT-SVD of uniform tensor")
    savefig(p, "./plots/ttsvd-uniform-error.png")

    # exercise 7

    d = 4; n = 51;
    ndim = [n for i=1:d]
    A = [f([t1, t2, t3, t4]) for t1 in t(n), t2 in t(n), t3 in t(n), t4 in t(n)];
    B = [g([t1, t2, t3, t4]) for t1 in t(n), t2 in t(n), t3 in t(n), t4 in t(n)];

    ## Exercise 7a

    r_jk_a, ϵ_jk_a, σ_jk_a = rank_approx(A, ttmps_unfold);
    r_jk_b, ϵ_jk_b, σ_jk_b = rank_approx(B, ttmps_unfold);
    p1 = plot(size=[700, 350], margin=5mm, dpi=300)
    p2 = plot(size=[700, 350], margin=5mm, dpi=300)
    for k=1:d
        plot!(p1,
            1 ./ ϵ_s,
            r_jk_a[:, k],
            ylabel=L"$r_{jk}$",
            label=L"A-$\varepsilon_{j%$k}$",
            xlabel=L"\varepsilon_{j}^{-1}",
            lw = 2,
            markershape=:circle,
            xaxis=:log10,
            legend=:topleft,
            title=L"Tensor $f(x_1, x_2, x_3, x_4)$")
        plot!(p2,
            1 ./ ϵ_s,
            r_jk_b[:, k],
            ylabel=L"$r_{jk}$",
            label=L"B-$\varepsilon_{j%$k}$",
            xlabel=L"\varepsilon_{j}^{-1}",
            lw = 2,
            markershape=:circle,
            xaxis=:log10,
            legend=:topleft,
            title=L"Tensor $g(x_1, x_2, x_3, x_4)$")
    end
    savefig(p1, "./plots/tucker-error-A.png")
    savefig(p2, "./plots/tucker-error-B.png")

    # exercise 7b
    Nj_ttsvd_a = []; ϵ_ttsvd_a = []; σ_ttsvd_a = [];
    Nj_ttsvd_b = []; ϵ_ttsvd_b = []; σ_ttsvd_b = [];

    println("Checking validity of error analysis for TT-SVD")
    for j=1:size(ϵ_jk_a, 1)
        Vs_a, vals_a, errors_a = tt_svd(A, ndim, r_jk_a[j, :], d);
        N_vals = 0
        for V in Vs_a
            N_vals += length(V)
        end
        append!(Nj_ttsvd_a, N_vals)
        append!(ϵ_ttsvd_a, errors_a[end])
        append!(σ_ttsvd_a, [vals_a])
        println("ϵ_ttsvd_a[j] <= |ϵ_jk_a|_f : $(ϵ_ttsvd_a[j] <= norm(ϵ_jk_a)) ϵ_ttsvd_a[j] = $(ϵ_ttsvd_a[j])")

        Vs_b, vals_b, errors_b = tt_svd(B, ndim, r_jk_b[j, :], d);
        N_vals = 0
        for V in Vs_b
            N_vals += length(V)
        end
        append!(Nj_ttsvd_b, N_vals)
        append!(ϵ_ttsvd_b, errors_b[end])
        append!(σ_ttsvd_b, [vals_b])
        println("ϵ_ttsvd_b[j] <= |ϵ_jk_b|_f : $(ϵ_ttsvd_b[j] <= norm(ϵ_jk_b)) ϵ_ttsvd_b[j] = $(ϵ_ttsvd_b[j])")
    end

    j = 5
    p1 = plot(dpi=300, size=[800, 350], margin=5mm)
    p2 = plot(dpi=300, size=[800, 350], margin=5mm)
    for k=1:d-1
        plot!(p1, σ_unf_a[k][1:r_jk_a[j, k]] ./ σ_ttsvd_a[j][k],
            ylabel=L"\frac{\sigma^{A_k}_\alpha}{\sigma^{ttsvd}_{k\alpha}}",
            label=L"k=%$k",
            xlabel=L"j",
            xticks=collect(1:length(r_jk_a)),
            yticks=[1, 1000],
    #        ylim=[1, 1000],
            lw = 2,
            markershape=:circle,
            legend=:topleft,
            title=L"$f(x_1, x_2, x_3, x_4)$")
        plot!(p2, σ_unf_b[k][1:r_jk_b[j, k]] ./ σ_ttsvd_b[j][k],
            ylabel=L"\frac{\sigma^{B_k}_\alpha}{\sigma^{ttsvd}_{k\alpha}}",
            label=L"k=%$k",
            xlabel=L"j",
            yticks=[1, 1000],
    #        ylim=[1, 1000],
            lw = 2,
            markershape=:circle,
            legend=:topleft,
            title=L"$g(x_1, x_2, x_3, x_4)$")
    end
    savefig(p1, "./plots/ttsvd-sigmaratio-a.png")
    savefig(p2, "./plots/ttsvd-sigmaratio-b.png")


    # Exercise 7d
    p1 = plot(size=[700, 350], margin=5mm, dpi=300)
    p2 = plot(size=[700, 350], margin=5mm, dpi=300)
    plot!(p1,
        1 ./ ϵ_s,
        Nj_ttsvd_a,
        ylabel=L"$N_{j}$",
        label=L"A-$N_{j}$",
        xlabel=L"\varepsilon_{j}^{-1}",
        lw = 2,
        markershape=:circle,
        xaxis=:log10,
        legend=:topleft,
        title=L"MPS-TT of $f(x_1, x_2, x_3, x_4)$")
    plot!(p2,
        1 ./ ϵ_s,
        Nj_ttsvd_b,
        ylabel=L"$N_{j}$",
        label=L"B-$N_{j}$",
        xlabel=L"\varepsilon_{j}^{-1}",
        lw = 2,
        markershape=:circle,
        xaxis=:log10,
        legend=:topleft,
        title=L"MPS-TT of $g(x_1, x_2, x_3, x_4)$")
    savefig(p1, "./plots/ttsvd-Nj-a.png")
    savefig(p2, "./plots/ttsvd-Nj-b.png")

end

main()
