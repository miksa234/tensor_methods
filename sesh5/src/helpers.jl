#/usr/bin/julia

using LinearAlgebra
using Plots
using Plots.Measures
using Distributions
using LaTeXStrings
using TSVD: tsvd
using TensorToolbox: tenmat, ttm, contract # todo:implement

@doc raw"""
    Helper functions to calculated the tensor values of the Chebyshev Polynomials \textit{che_pol}
    on a Grid \textit{t(n)}
"""

t(n) = [2*(i-1)/(n-1)-1 for i ∈ 1:n]

function che_pol(x, q)
    if abs(x) ≤ 1
        return cos(q*acos(x))
    elseif x ≥ 1
        return cosh(q*acosh(x))
    elseif x ≤ -1
        return (-1)^q * cosh(q*acosh(-x))
    end
end
