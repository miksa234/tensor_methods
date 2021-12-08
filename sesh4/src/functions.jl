#/usr/bin/julia

t(n) = [(2*(i-1)/(n-1) - 1) for i in 1:n]

function f(x)
    s = 0
    for (k, xk) in enumerate(x)
        s = s .+ (xk.^2 / (8^(k-1)))
    end
    return (1 .+ s).^(-1)
end

function g(x)
    s1 = 0
    s2 = 0
    for (k, xk) in enumerate(x)
        s1 = s1 .+ (xk.^2 / (8^(k-1)))
        s2 = s2 .+ (4 * pi * xk)/(4^(k-1))
    end
    return sqrt.(s1) .* (1 .+ 1/2 * cos.(s2))
end
