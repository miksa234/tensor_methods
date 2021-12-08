#/usr/bin/julia

using LinearAlgebra
using Plots
using Distributions
using LaTeXStrings
using Random
using TSVD: tsvd # todo: implement
using TensorToolbox: tenmat, ttm, contract # todo:implement
using Plots.Measures

function my_hosvd(C, r)
    S0 = copy(C); d = length(size(C)); n = size(C)
    V = []; singular_values = []; errors = [];
    for k=1:d
        # reshape & permute S_{k-1} -> B
        B_perm = permutedims(S0, ([[i for i=1:d if i!=k]..., k]))
        B = reshape(B_perm, (prod([[r_i for  r_i in r[1:(k-1)]]..., [i_k for i_k in n[(k+1):d]]...])..., n[k]))
        B = convert(Matrix{Float64}, B)

        # rank r_k T-SVD of B -> B^hat
        U_hat, Sig_hat, V_hat  = tsvd(B, r[k])
        B_hat = U_hat * Diagonal(Sig_hat) * transpose(V_hat)

        # reshape & permute B^hat -> S_k
        W_hat = B_hat * V_hat
        W_reshape = reshape(W_hat , ([r_i for r_i in r[1:(k-1)]]..., [n_k for n_k in n[k+1:end]]..., r[k]))
        S0 = permutedims(W_reshape , ([i for i=1:k-1]..., d, [i for i=k:d-1]...))


        append!(V, [V_hat])
        append!(singular_values, [Sig_hat])
        C_hat = tucker_eval(S0, V)
        append!(errors, norm(C_hat - C)/norm(C))
    end
    return V, S0, singular_values, errors
end

function tucker_eval(S, V)
    d = length(V); A = copy(S)
    for (k, V_k) in enumerate(V)
        A = ttm(A, V_k, k)
    end
    return A
end
