#/usr/bin/julia

using LinearAlgebra
using Plots
using Plots.Measures
using Distributions
using LaTeXStrings
using TSVD: tsvd
using TensorToolbox: tenmat, ttm, contract # todo:implement

@doc raw"""
    Output: the k-th TT-MPS unfolding matrix of $A$.
        A_{i_1, … , i_n} -> A_{i_1 ⋯ i_{k}, i_{k+1} ⋯ i_d}
"""
function ttmps_unfold(A, k)
    n = size(A)
    d = length(n)
    C = copy(A)
    A_k = reshape(C, (prod([n[i] for i in 1:k]), prod([n[i] for i in k+1:d])))
    return A_k
end

@doc raw"""
        TT-MPS decomposition evaluation to a Tensor
"""
function ttmps_eval(U, n)
    A = U[1]
    for U_k ∈ U[2:end]
        A = contract(A, U_k)
    end
    return reshape(A, n...)
end

@doc raw"""
        Truncated MPS-TT:
        Given a decomposition U of ranks $p_1, … ,p_{d-1}$ produce a decomposition of target
        ranks not exceeding $r = [r_1, … ,r_{d-1}]$ with the TT-MPS orthogonalization algorithm
"""
function t_mpstt(U, r)
    r_n = [1, r..., 1]
    d = length(U)
    Q = []; U_k = U[1]
    for k ∈ 2:d
        α_k_1, i_k, α_k = size(U_k)
        U_k_bar = reshape(U_k, (α_k_1*i_k, α_k))

        P_k, Σ_k, W_k = tsvd(U_k_bar, r_n[k])
        Q_k = reshape(P_k, (α_k_1, i_k, r_n[k]))
        Z_k = Diagonal(Σ_k) * transpose(W_k)
        U_k = contract(Z_k, U[k])
        append!(Q, [Q_k])
        if k == d
            append!(Q, [U_k])
        end
    end
    return Q
end

@doc raw"""
        Truncated MPS-TT:
        Same as above but with a tolerace = ϵ, with which the ranks are
        approximated. The difference is there are no target ranks required
"""
function t_mpstt_ϵ(U, ϵ)
    r = [1]
    d = length(U)
    dims = [size(U[k], 2) for k ∈ 1:d]
    δ = ϵ/sqrt(d-1) * norm(ttmps_eval(U, dims))
    Q = []; U_k = U[1]
    for k ∈ 2:d
        α_k_1, i_k, α_k = size(U_k)
        U_k_bar = reshape(U_k, (α_k_1*i_k, α_k))
        for r_k ∈ 1:rank(U_k_bar)
            global P_k, Σ_k, W_k = tsvd(U_k_bar, r_k)
            U_k_bar_hat = P_k * Diagonal(Σ_k) * transpose(W_k)
            if norm(U_k_bar - U_k_bar_hat)/norm(U_k_bar) ≤ δ
                append!(r, r_k)
                break
            end
        end
        Q_k = reshape(P_k, (α_k_1, i_k, r[k]))
        Z_k = Diagonal(Σ_k) * transpose(W_k)
        U_k = contract(Z_k, U[k])
        append!(Q, [Q_k])
        if k == d
            append!(Q, [U_k])
        end
    end
    return Q, r
end


@doc raw"""
    TT-SVD algorithm
"""
function tt_svd(A, r)
    n = size(A)
    d = length(n)
    r_new = [1, r...]
    S_0_hat = copy(A)
    S0s = []
    C = []; σ = []; ϵ = []
    for k=2:d
        B_k = reshape(S_0_hat, (r_new[k-1] * n[k-1], prod([n[i] for i=k:d])))
        U_hat, Σ_hat, V_hat = tsvd(convert(Matrix{Float64}, B_k), r_new[k])
        C_k = reshape(U_hat, (r_new[k-1], n[k-1], r_new[k]))
        W_k_hat = Diagonal(Σ_hat) * transpose(V_hat)
        S_0_hat = reshape(W_k_hat, (r_new[k], [n[i] for i=k:d]...))

        append!(C, [C_k])
        append!(σ, [Σ_hat])
        A_hat = ttmps_eval([C..., S_0_hat], n)
        append!(ϵ, norm(A_hat - A)/norm(A))
    end
    append!(C, [reshape(S_0_hat, (size(S_0_hat)..., 1))])
    return C, σ, ϵ
end

@doc raw"""
    Tolerace bound TT-SVD algorithm, no target ranks required.
    The ranks are calculated based on the tolerace specified.
"""
function tt_svd_ϵ(A, tol)
    n = size(A)
    d = length(n)
    r = [1]
    S_0_hat = copy(A)
    δ = tol/sqrt(d-1) * norm(A)
    C = []; σ = []; ϵ = []
    for k=2:d
        B_k = reshape(S_0_hat, (r[k-1] * n[k-1], prod([n[i] for i=k:d])))
        for r_k = 1:rank(B_k)
            global U_hat, Σ_hat, V_hat = tsvd(convert(Matrix{Float64}, B_k), r_k)
            B_k_hat = U_hat * Diagonal(Σ_hat) * transpose(V_hat)
            if norm(B_k - B_k_hat)/norm(B_k) ≤ δ
                append!(r, r_k)
                break
            end
        end
        C_k = reshape(U_hat, (r[k-1], n[k-1], r[k]))
        W_k_hat = Diagonal(Σ_hat) * transpose(V_hat)
        S_0_hat = reshape(W_k_hat, (r[k], [n[i] for i=k:d]...))

        append!(C, [C_k])
        append!(σ, [Σ_hat])
        A_hat = ttmps_eval([C..., S_0_hat], n)
        append!(ϵ, norm(A_hat - A)/norm(A))
    end
    append!(C, [reshape(S_0_hat, (size(S_0_hat)..., 1))])
    return C, r, σ, ϵ
end

