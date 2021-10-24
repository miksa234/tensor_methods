#/usr/bin/julia

include("./functions/err_lucross.jl")
include("./functions/err_tsvd.jl")
include("./functions/err_plot.jl")
include("./functions/cs_plot.jl")

using LinearAlgebra
using Plots
using Maxvol
using TSVD
using LaTeXStrings

global n = 901;

f(x::Number, y::Number) = 1/(1 + x^2 + y^2);
g(x::Number, y::Number) = sqrt(x^2 + y^2)*(1 + 1/2*cos(15x + 22y));


function main()
    t = [(2*(i-1)/(n-1) - 1) for i in 1:n];

    A = [f(t_i, t_j) for t_i in t, t_j in t];
    B = [g(t_i, t_j) for t_i in t, t_j in t];

    x_a = collect(1:rank(A));
    x_b = collect(1:rank(B));
    x_a_l = log.(x_a); x_b_l = log.(x_b);

    # TSVD
    err_a_tsvd = sort_out_values(A, x_a, get_errors_tsvd);
    err_b_tsvd = sort_out_values(B, x_b, get_errors_tsvd);

    err_plot([x_a, x_b], [err_a_tsvd, err_b_tsvd], "lu-svd-err.png")


    # Cross-LU without pivoting
    err_a_lunp = cross_lu_error(A, x_a, Val(false));
    err_b_lunp = cross_lu_error(B, x_b, Val(false));

    err_plot([x_a, x_b], [err_a_lunp, err_b_lunp], "lu-np-err.png")


    # Cross-LU with pivoting
    err_a_lup = cross_lu_error(A, x_a, Val(true));
    err_b_lup = cross_lu_error(B, x_b, Val(true));

    err_plot([x_a, x_b], [err_a_lup, err_b_lup], "lu-p-err.png")

    # graph restrictions by LU pivoting
    r_a = [1, 2, 3, 4, 5];
    r_b = [1, 2, 3, 4, 5, 10, 15, 20 ,30 , 40];
    P_A_s = [M_cross_lu_err(A, i, Val(true))[2] for i in r_a];
    P_B_s = [M_cross_lu_err(B, i, Val(true))[2] for i in r_b];

    cs_plot(t, f, P_A_s, r_a, "a")
    cs_plot(t, f, P_B_s, r_b, "b")

end

main()
