#/usr/bin/julia

include("./functions.jl")

using LinearAlgebra

function main()
    # Exercise 3
    k = 2
    r = 1
    n = (4, 3)

    S = rand(1:10, n) # n_1 x ... x n_d array
    Z = rand(1:10, r, n[k]) # r x n_k array for k in {1,...,d}, r natural number
    T = mode_k_dot(S, Z, k)  # n_1 x ... x n_{k-1} x r x n_{k+1} x ... x n_d
    println("\n\nExercise 3")
    println("\nMode-k contraction of $(size(S)) with $(size(Z)) gets $(size(T))")

    # Exercise 4
    U_1 = rand(1:10, 5, 3)
    U_2 = rand(1:10, 4, 3)
    U_3 = rand(1:10, 3, 3)
    U_4 = rand(1:10, 2, 3)
    U_5 = rand(1:10, 1, 3)
    U = [U_1, U_2, U_3, U_4, U_5]

    s = cpd_eval(U)
    println("\n\nExercise 4")
    println("CPD evaluation of U_d (d=5) of sizes $(size(U_1)) ... $(size(U_5)): $(size(s))")

    # Exercise 5
    println("\n\nExercise 5")
    for n=1:8
        T, (U, V, W)= multiplication_tensor(n)
        println("For n=$n T == cpd_eval(U, V, W): ", cpd_eval([U, V, W]) == T)
    end

    # Exercise 6
    n = 4   # can take bigger n
    A = rand(1:10, n, n)
    B = rand(1:10, n, n)
    T, (U, V, W) = multiplication_tensor(n)
    println("\n\nExercise 6")
    println("\n Multiplication Tensor order $n")
    display(T)
    println("\nCPD multiplication yields: $(A*B == cpd_multiply(A, B, U, V, W))")

    # Exercise 7
    U, V, W = rank_7_CPD()
    A = rand(1:10, 2, 2)
    B = rand(1:10, 2, 2)
    println("\n\nExercise 5")
    println("\nTiming Strassens A⋅B")
    println(@time strassen_alg(A, B))

    println("\nTiming julia's A⋅B")
    println(@time A*B)
    println("\nDoes Strassen Algorithm get the right results? :$(strassen_alg(A,B) == A*B)")

    println("\nrank 7 CPD T  yields the right tensor $(cpd_eval([U, V, W]) == multiplication_tensor(2)[1])")
    println("\nCPD multiplication yields: $(A*B == cpd_multiply(A, B, U, V, W))")
end

main()
