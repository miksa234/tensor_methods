#/usr/bin/julia

using LinearAlgebra
using Plots
using Maxvol
using TSVD
using LaTeXStrings

function err_plot(x, y, save)
    p1 = plot(x[1], y[1][1],
              title="A: frb-error",
              lw=2,
              xlabel=L"ln(r)\;\;[\;]",
              ylabel=L"||\tilde{M} -M||\;\;[\;]",
              label="");
    p2 = plot(x[1], y[1][2],
              title="A: max-error",
              lw=2,
              xlabel=L"ln(r)\;\;[\;]",
              ylabel=L"||\tilde{M} -M||\;\;[\;]",
              label="");
    p3 = plot(x[2], y[2][1],
              title="B: frb-error",
              lw=2,
              xlabel=L"ln(r)\;\;[\;]",
              ylabel=L"||\tilde{M} -M||\;\;[\;]",
              label="");
    p4 = plot(x[2], y[2],
              title="B: max-error",
              lw=2,
              xlabel=L"ln(r)\;\;[\;]",
              ylabel=L"||\tilde{M} -M||\;\;[\;]",
              label="");

    savefig(plot(p1, p2, p3, p4,
                 layout=(2, 2),
                 titlefontsize=10,
                 xlabelfontsize=7,
                 ylabelfontsize=5,
                 dpi=300),
            save)
end
