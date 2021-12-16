#/usr/bin/julia

using LinearAlgebra
using Plots
using Plots.Measures
using Distributions
using LaTeXStrings
using TSVD: tsvd
using TensorToolbox: tenmat, ttm, contract # todo:implement

@doc raw"""
    Addition of two tensors in the TT-MPS format of $A$  and $B$ of the same size
    $n_1 × ⋯ × n_d$ with different TT-MPS decompositions U with ranks $p_1, … , p_{d-1}$
    and V with ranks $q_1, … ,q_{d-1}$ respectively.

    Output is the TT-MPS representation of $A+B$ with ranks $(q_1 + p_1), ⋯ , (q_{d-1} + p_{d-1})$.
"""
function tt_add(U, V, α, β)
    W = []
    d = length(U)
    for k ∈ 1:d
        U_k = U[k]; V_k = V[k]
        p_k, n_k, p_k1 = size(U_k)
        q_k, n_k, q_k1 = size(V_k)
        if k == 1
            W_1 = zeros(1, n_k, q_k1 + p_k1)
            for i=1:n_k
                W_1[:, i, :] += hcat(α * U_k[:, i, :], β * V_k[:, i, :])
            end
            append!(W, [W_1])
        elseif k == d
            W_d = zeros(q_k + p_k, n_k, 1)
            for i=1:n_k
                W_d[:, i, :] += vcat(U_k[:, i, :], V_k[:, i, :])
            end
            append!(W, [W_d])
        else
            W_k = zeros(p_k + q_k, n_k, q_k1 + p_k1)
            for i=1:n_k
                W_k[:, i, :] += hcat(vcat(U_k[:, i, :], zeros(q_k, p_k1)),
                                 vcat(zeros(p_k, q_k1), V_k[:, i, :]))
            end
            append!(W, [W_k])
        end
    end
    return W
end

@doc raw"""
    Hadamard product of two tensors in the TT-MPS format of $A$  and $B$ of the same size
    $n_1 × ⋯ × n_d$ with different TT-MPS decompositions U with ranks $p_1, … , p_{d-1}$
    and V with ranks $q_1, … ,q_{d-1}$ respectively.

    Output is the representation of $A ⊙ B$ with ranks $(q_1 ⋅ p_1), ⋯ , (q_{d-1} ⋅ p_{d-1})$.
"""
function tt_mult(U, V)
    W = []
    d = length(U)
    for k ∈ 1:d
        p_k, n_k, p_k1 = size(U[k])
        q_k, n_k, q_k1 = size(V[k])
        W_k = zeros(p_k*q_k, n_k, p_k1 * q_k1)
        for i_k ∈ 1:n_k
            W_k[:, i_k, :] += kron(U[k][:, i_k, :], V[k][:, i_k, :])
        end
        append!(W, [W_k])
    end
    return W
end

@doc raw"""
    Matrix Vector product of two tensors in the TT-MPS format of $A$, size $(m_1⋯m_d×n_1⋯n_d)$  and $u$ size $(n_1⋯n_d)$
    with different TT-MPS decompositions U with ranks $p_1, … , p_{d-1}$
    and V with ranks $q_1, … ,q_{d-1}$ respectively.

    Output is the TT-MPS representation of $A ⋅ u$ with ranks $(q_1 ⋅ p_1), ⋯ , (q_{d-1} ⋅ p_{d-1})$.
"""
function tt_matvec(A, U)
    W = []
    d = length(U)
    for k ∈ 1:d
        α_k, n_k, m_k, α_k1 = size(A[k])
        β_k, m_k, β_k1 = size(U[k])
        W_k = zeros(α_k * β_k, n_k, α_k1 * β_k1)
        for i_k ∈ 1:n_k
            for j_k ∈ 1:m_k
                W_k[:, i_k, :] += kron(A[k][:, i_k, j_k, :], U[k][:, j_k, :])
            end
        end
        append!(W, [W_k])
    end
    return W
end
