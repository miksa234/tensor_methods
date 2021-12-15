#/usr/bin/julia

using LinearAlgebra
using Plots
using Plots.Measures
using Distributions
using LaTeXStrings
using TSVD: tsvd
using TensorToolbox: tenmat, ttm, contract # todo:implement

include("arithmetic.jl")
include("helpers.jl")
include("ttmps.jl")

function main()
    # Exercise 3a)
    n = 10
    d = 4
    p_Σ = []
    p_R = []
    for (p, q) ∈ ([3, 4], [5, 7])
        X = [che_pol(sum([t_1/1, t_2/2, t_3/3, t_4/4]), p) for t_1 ∈ t(n), t_2 ∈ t(n), t_3 ∈ t(n), t_4 ∈ t(n)];
        Y = [che_pol(sum([t_1/1, t_2/2, t_3/3, t_4/4]), q) for t_1 ∈ t(n), t_2 ∈ t(n), t_3 ∈ t(n), t_4 ∈ t(n)];
        S = X .+ Y;
        Z = X .* Y;
        global X_Σ = []; global Y_Σ = []; global S_Σ = []; global Z_Σ = []
        global r_x = []; global r_y = []; global r_s = []; global r_z = []
        for k ∈ 1:d-1
            X_k = ttmps_unfold(X, k)
            Y_k = ttmps_unfold(Y, k)
            S_k = ttmps_unfold(S, k)
            Z_k = ttmps_unfold(Z, k)
            _, σ_x, _ = svd(X_k)
            _, σ_y, _ = svd(Y_k)
            _, σ_s, _ = svd(S_k)
            _, σ_z, _ = svd(Z_k)
            append!(X_Σ, [σ_x])
            append!(Y_Σ, [σ_y])
            append!(S_Σ, [σ_s])
            append!(Z_Σ, [σ_z])
            append!(r_x, [rank(X_k)])
            append!(r_y, [rank(Y_k)])
            append!(r_s, [rank(S_k)])
            append!(r_z, [rank(Z_k)])
        end
        p_σ = plot(layout = 4, margin=5mm, size=(800, 800), title="p=$p, q=$q", dpi=500)
        plot!(p_σ[1], X_Σ, markershape=:circle, ylabel=L"\Sigma_{X_k}", label=[L"X_1" L"X_2" L"X_3"])
        plot!(p_σ[2], Y_Σ, markershape=:circle, ylabel=L"\Sigma_{Y_k}", label=[L"Y_1" L"Y_2" L"Y_3"])
        plot!(p_σ[3], S_Σ, markershape=:circle, ylabel=L"\Sigma_{S_k}", label=[L"S_1" L"S_2" L"S_3"])
        plot!(p_σ[4], Z_Σ, markershape=:circle, ylabel=L"\Sigma_{Z_k}", label=[L"Z_1" L"Z_2" L"Z_3"])
        append!(p_Σ, [p_σ])
        p_r = plot(margin=5mm, size=(700, 350), title="p=$p, q=$q", ylabel=L"r_k", dpi=300)
        plot!(p_r, r_x, markershape=:circle, label=L"r_X")
        plot!(p_r, r_y, markershape=:circle, label=L"r_Y")
        plot!(p_r, r_s, markershape=:circle, label=L"r_S", alpha=0.5)
        plot!(p_r, r_z, markershape=:circle, label=L"r_Z")
        append!(p_R, [p_r])

    end
    savefig(p_Σ[1], "./plots/sigma_34.png")
    savefig(p_Σ[2], "./plots/sigma_57.png")
    savefig(p_R[1], "./plots/rank_34.png")
    savefig(p_R[2], "./plots/rank_57.png")


    # Exercise 3b)
    n = 10; p = 5; q = 7
    dims = [10 for i ∈ 1:4]

    X = [che_pol(sum([t_1/1, t_2/2, t_3/3, t_4/4]), p) for t_1 ∈ t(n), t_2 ∈ t(n), t_3 ∈ t(n), t_4 ∈ t(n)];
    Y = [che_pol(sum([t_1/1, t_2/2, t_3/3, t_4/4]), q) for t_1 ∈ t(n), t_2 ∈ t(n), t_3 ∈ t(n), t_4 ∈ t(n)];
    S = X .+ Y
    Z = X .* Y

    C_x, r_x, σ_x, ϵ_x = tt_svd_ϵ(X, 1e-12);
    C_y, r_y, σ_y, ϵ_y = tt_svd_ϵ(Y, 1e-12);
    C_s = tt_add(C_x, C_y, 1, 1);
    C_z = tt_mult(C_x, C_y);

    Q_s, r_s = t_mpstt_ϵ(C_s, 1e-12);
    Q_z, r_z = t_mpstt_ϵ(C_z, 1e-7);

    println("Exercise 3b")
    println("|ttmps_eval(C_s) - S||_F/||S||_F = $(norm(S - ttmps_eval(C_s, dims))/norm(S))")
    println("|ttmps_eval(C_z) - Z||_F/||Z||_F = $(norm(Z - ttmps_eval(C_z, dims))/norm(Z))")
    println()
    println("|ttmps_eval(Q_s) - S||_F/||S||_F = $(norm(S - ttmps_eval(Q_s, dims))/norm(S))")
    println("|ttmps_eval(Q_z) - Z||_F/||Z||_F = $(norm(Z - ttmps_eval(Q_z, dims))/norm(Z))")
end

main()
