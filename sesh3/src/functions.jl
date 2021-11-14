#/usr/bin/julia

using LinearAlgebra
using LaTeXStrings
using Distributions
using Plots

include("../../sesh2/src/functions.jl")

function cp_als(U, V, iter)
    d = length(U)
    R = size(U[1])[2]
    r = size(V[1])[2]
    dims = [size(U[l])[1] for l=1:d]
    Vk = copy(V)

    Fs = [eleprod([(Vk[l]' * U[l]) for l=1:d if l!=j]) for j=1:d]
    Gs = [eleprod([(Vk[l]' * Vk[l]) for l=1:d if l!=j]) for j=1:d]

    phi = []
    d_phi_2 = []
    for count=1:iter
        for k in [collect(2:d)..., reverse(collect(1:(d-1)))...]
            Fs[k] = eleprod([(Vk[l]' * U[l]) for l=1:d if l!=k])
            Gs[k] = eleprod([(Vk[l]' * Vk[l]) for l=1:d if l!=k])

            VV = kron(Gs[k], Matrix{Float64}(I, dims[k], dims[k]))
            VU = kron(Fs[k], Matrix{Float64}(I, dims[k], dims[k]))

            Vk[k] = reshape(VV\(VU * vec(U[k])), (dims[k], r))
        end
        append!(phi, norm(cpd_eval(Vk) - cpd_eval(U)))
        append!(d_phi_2, norm(diff(1/2*phi.^2)))
    end
    return Vk, phi, d_phi_2
end


function eleprod(A)
    d = length(A)
    Z = copy(A[1])
    for i=2:d
        Z = Z .* A[i]
    end
    return Z
end
