#/usr/bin/julia

using LinearAlgebra
using Plots
using Distributions
using LaTeXStrings
using Random
using TSVD: tsvd # todo: implement
using TensorToolbox: tenmat, ttm, contract # todo:implement
using Plots.Measures

function σ_unfold(C, d)
    Σ_s = []
    for k=1:d
        U, Σ, V = svd(tenmat(C, k))
        append!(Σ_s, [Σ])
    end
    return Σ_s
end

function ttmps_unfold(A, k)
    n = size(A)
    d = length(n)
    C = copy(A)
    A_k = reshape(C, (prod([n[i] for i in 1:k]), prod([n[i] for i in k+1:d])))
    return A_k
end

function tucker_unfold(A, k)
    n = size(A)
    d = length(n)
    C = copy(A)
    C_perm = permutedims(C, ([i for i=1:d if i!=k]..., k))
    A_k = reshape(C_perm, (prod([n[i] for i=1:d if i!=k]), n[k]))
    return A_k
end

ϵ_s = [1/(10^(j*2)) for j=1:5] # computer not goode enough for j = 6
function rank_approx(C, method,ϵ_s=ϵ_s)
    d = length(size(C))
    ϵ_jk = []; r_jk = []; σ_jk = []
    for k=1:d
        C_k = method(C, k)
        for (j, ϵ_j) in enumerate(ϵ_s)
            for r=1:rank(C_k)
                U_hat, Σ_hat, V_hat = tsvd(C_k, r)
                C_k_hat = U_hat * Diagonal(Σ_hat) * transpose(V_hat)
                if norm(C_k_hat-C_k)/norm(C_k) <= ϵ_j
                    append!(ϵ_jk, norm(C_k_hat-C_k)/norm(C_k))
                    append!(r_jk, r)
                    append!(σ_jk, [Σ_hat])
                    break
                end
            end
        end
    end
    ndims = (length(ϵ_s), d)
    return reshape(r_jk, ndims), reshape(ϵ_jk, ndims), reshape(σ_jk, ndims)
end
