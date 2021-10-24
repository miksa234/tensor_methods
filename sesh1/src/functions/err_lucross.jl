#/usr/bin/julia

using LinearAlgebra
using Plots
using Maxvol
using TSVD
using LaTeXStrings

function M_cross_lu_err(M, r, pivot=Val(true))
    M_mk = copy(M[:, 1:r]);
    C = copy(M_mk);
    piv, ntimes = maxvol!(C);
    F = lu(M_mk[piv, :], pivot, check=false)
    M_mk[piv, :] =  F.P * F.L * F.U ;
    return M - C*M[piv, :], piv
end

function cross_lu_error(M, x, pivot)
    y_frb = [norm(M_cross_lu_err(M, i,pivot)[1]) for i in x]
    y_max = [norm(M_cross_lu_err(M, i,pivot)[1], Inf) for i in x]
    return y_frb, y_max
end;
