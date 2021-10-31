#/usr/bin/julia

using LinearAlgebra

@doc raw"""
    mode_k_dot(T, Z, k)

Compute the mode-$k$ contraction of a tensor $S$ of size $n_1 ×  ⋯ × n_d$
with a matrix $Z$ of size $r × n_k$

Out: A tensor $T$ of size $n_1 ×  ⋯ ×  n_{k-1} ×  r ×  n_{k+1} ×  ⋯ × n_d$
"""
function mode_k_dot(S, Z, k)
    n = size(S)
    r = size(Z)[1]

    if length(n) < k
        error("Try a smaller k")
    end
    if (size(Z))[2] != n[k]
        error("Z not of the correct size for mode-k contraction")
    end

    end_dim = [[n[i] for i in 1:(k-1)]..., r, [n[i] for i in (k+1):length(n)]...]
    alpha_dim = [n[i] for i in 1:length(n) if i!=k]
    T = []
    for alpha in 1:(size(Z))[1]
        s = zeros(alpha_dim...)
        for i_k in 1:n[k]
            s +=  Z[alpha, i_k] * selectdim(S, k, i_k)
        end
        append!(T, s)
    end
    return reshape(T, end_dim...)
end

@doc raw"""
    multiplication_tensor(n)

Compute the multiplication tensor for the multiplication $T$ of two matrices $A$ and $B$
of size \alpha $n × n$, satisfying the column major convention

        $C_k = ∑_{i, j}^{n^2} T_{ijk} ⋅ A_i ⋅ B_k$
"""
function multiplication_tensor(n)
    T = zeros(n^2, n^2, n^2)
    U = zeros(n^2, n^3)
    V = zeros(n^2, n^3)
    W = zeros(n^2, n^3)

    count = 1
    for m=1:n:n^2, l=1:n
        for (i, j) in zip(collect(l:n:n^2), collect(m:m+n))
            k = (l-1)+m
            T[i, j, k] = 1
            U[i, count] = 1
            V[j, count] = 1
            W[k, count] = 1
            count += 1
        end
    end
    return T, (U, V, W)
end

@doc raw"""
    cpd_eval(U_i)

Evaluate a rank-r CPD of a tensor $S$ of size $n_1 × ⋯ × n_d$, given $d$
matrices $U_i$ of the size $n_k × r$.

        $U_i = [U_1, … , U_d]$
"""
function cpd_eval(U_i)
    reverse!(U_i)
    d = length(U_i)
    r = size(U_i[1])[2]
    n_d = [size(U_i[i])[1] for i in 1:d]
    s = zeros(n_d...)
    for alpha in 1:r
        k = kron(U_i[1][:, alpha], U_i[2][:, alpha])
        for i in 2:(d-1)
            k = kron(k, U_i[i+1][:, alpha])
        end
        s += reshape(k, n_d...)
    end
    return s
end

@doc raw"""
    cpd_multiply(A, B, U, V, W)

Compute the matrix multiplication $A ⋅ B$ using the rank-$n^3$ CPD of the multiplication Tensor,
without evaluating the tensor
"""
function cpd_multiply(A, B, U, V, W)
    n = size(A)[1]
    r = size(U)[2]
    C = zeros(n, n)
    for k=1:n^2
        C[k] = sum(sum(A[i] * U[i, alpha] for i=1:n^2)
                    * sum(B[j] * V[j, alpha] for j=1:n^2)
                    * W[k, alpha] for alpha=1:r)
    end
    return C
end

@doc raw"""
    strassen_alg(A, B)

Implementation of the Stranssens Algorithm for two matrices $A$ and $B$
of the size $2 × 2$
"""
function strassen_alg(A, B)
    if size(A) != (2, 2) || size(B) != (2, 2)
        error("Choose 2x2 Matrices")
    end
    C = zeros(2, 2)
    M_1 = (A[1] + A[4])*(B[1] + B[4])
    M_2 = (A[2] + A[4])*B[1]
    M_3 = A[1]*(B[3] - B[4])
    M_4 = A[4]*(B[2] - B[1])
    M_5 = (A[1] + A[3])*B[4]
    M_6 = (A[2] - A[1])*(B[1] + B[3])
    M_7 = (A[3] - A[4])*(B[2] + B[4])

    C[1] = M_1 + M_4 - M_5 + M_7
    C[2] = M_2 + M_4
    C[3] = M_3 + M_5
    C[4] = M_1 - M_2 + M_3 + M_6
    return C
end

@doc raw"""
    rank_7_CPD()

Get the rank-7 CPD of the $2 × 2$ multiplication tensor
(Strassens Algorithm)
"""
function rank_7_CPD()
    U = [1 0 1 0 1 -1 0  # A_i placement in M_{ijk}
         0 1 0 0 0 1 0
         0 0 0 0 1 0 1
         1 1 0 1 0 0 -1];

    V = [1 1 0 -1 0 1 0   # B_j placement in M_{ijk}
         0 0 0 1 0 0 1
         0 0 1 0 0 1 0
         1 0 -1 0 1 0 1];

    W = [1 0 0 1 -1 0 1  # M_{ijk} placement in C_k
         0 1 0 1 0 0 0
         0 0 1 0 1 0 0
         1 -1 1 0 0 1 0];

    return U, V, W
end
