#/usr/bin/julia

using LinearAlgebra
using Plots
using Maxvol
using TSVD
using LaTeXStrings

function cs_plot(x, f, P_M_s, r, name)
    x_red = x[1:3:n];
    s = surface(x_red, x_red, f, colorbar=false);
    c = contour(x_red, x_red, f, colorbar=false);
    for (i, r_i) in enumerate(r)
        p = plot(s, c,
                 layout=(1, 2),
                 titlefontsize=10,
                 xlabelfontsize=7,
                 ylabelfontsize=5,
                 dpi=300,
                 title="$f(x, y) at r=$r_i",
                 size=(600, 300))
        hline!(p[1], [0], x[P_M_s[i]], color=:red, opacity=0.5, label="", lw=2)
        hline!(p[2], [0], x[P_M_s[i]], color=:red, opacity=0.5, label="", lw=2)
        savefig(p, "cs-$name-$r_i.png")
    end
end
