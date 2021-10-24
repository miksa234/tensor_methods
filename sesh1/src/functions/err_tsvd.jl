#/usr/bin/julia

using LinearAlgebra
using Plots
using Maxvol
using TSVD
using LaTeXStrings

function get_errors_tsvd(M ,k)
    U, s, V = tsvd(M, k)
    M_hat = U * Diagonal(s) * transpose(V)
    dM = M_hat - M
    return [norm(dM), norm(dM, Inf)]
end;

function sort_out_values(M, x, error_function)
    errors = [error_function(M, i) for i in x]
    y_f = [i[1] for i in errors]
    y_m = [i[2] for i in errors]
    return y_f, y_m
end;
