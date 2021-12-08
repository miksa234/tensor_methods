#/usr/bin/julia

using LinearAlgebra
using Plots
using Plots.Measures
using Distributions
using LaTeXStrings
using Random
using TSVD: tsvd # todo: implement
using TensorToolbox: tenmat, ttm, contract # todo:implement

function ttmps_eval(U, n)
    A = U[1]
    for U_k=U[2:end]
        A = contract(A, U_k)
    end
    return reshape(A, n...)
end

function tt_svd(A, n, r, d)
    r_0 = 1
    r_new = [r_0, r...]
    S_0_hat = copy(A)
    S0s = []
    C = []; singular_val = []; errors = []
    for k=2:d
        B_k = reshape(S_0_hat, (r_new[k-1] * n[k-1], prod([n[i] for i=k:d])))
        U_hat, Sig_hat, V_hat = tsvd(convert(Matrix{Float64}, B_k), r_new[k])
        C_k = reshape(U_hat, (r_new[k-1], n[k-1], r_new[k]))
        W_k_hat = Diagonal(Sig_hat) * transpose(V_hat)
        S_0_hat = reshape(W_k_hat, (r_new[k], [n[i] for i=k:d]...))

        append!(C, [C_k])
        append!(singular_val, [Sig_hat])
        A_hat = ttmps_eval([C..., S_0_hat], n)
        append!(errors, norm(A_hat - A)/norm(A))
    end
    append!(C, [S_0_hat])
    return C, singular_val, errors
end
