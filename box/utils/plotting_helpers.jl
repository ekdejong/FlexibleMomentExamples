using Plots

include("./box_model_helpers.jl")

"""
  get_params(dist)

  - `dist` - is a particle mass distribution
Returns the names and values of settable parameters for a dist.
"""
function get_params(dist::CPD.PrimitiveParticleDistribution{FT} where {FT <: Real})
    params = Array{Symbol, 1}(collect(propertynames(dist)))
    values = Array{FT, 1}([getproperty(dist, p) for p in params])
    return params, values
end

"""
  plot_moments(sol, p; file_name = "test_moments.png")

  `sol` - ODE solution
  `p` - additional ODE parameters carried in the solver
Plots the moment time series results
"""
function plot_moments!(sol, p; file_name = "test_moments.png")
    time = sol.t
    moments = vcat(sol.u'...)

    Ndist = length(p.pdists)
    n_params = [nparams(p.pdists[i]) for i in 1:Ndist]
    Nmom_min = minimum(n_params)
    Nmom_max = maximum(n_params)
    moments_sum = zeros(length(time), Nmom_min)

    plt = Array{Plots.Plot}(undef, Nmom_max)
    for i in 1:Nmom_max
        plt[i] = plot()
    end

    for i in 1:Ndist
        for j in 1:n_params[i]
            ind = get_dist_moment_ind(p.NProgMoms, i, j)
            plt[j] = plot(
                plt[j],
                time,
                moments[:, ind],
                linewidth = 2,
                xaxis = "time [s]",
                yaxis = "M" * string(j - 1),
                label = "M_{" * string(j - 1) * "," * string(i) * "}",
                ylims = (-0.1 * maximum(moments[:, ind]), 1.1 * maximum(moments[:, ind])),
            )
            if j <= Nmom_min
                moments_sum[:, j] += moments[:, ind]
            end
        end
    end
    for i in 1:Nmom_min
        plt[i] = plot(
            plt[i],
            time,
            moments_sum[:, i],
            linestyle = :dash,
            linecolor = :black,
            label = "M_" * string(i - 1),
            linewidth = 2,
            ylims = (-0.1 * maximum(moments_sum[:, i]), 1.1 * maximum(moments_sum[:, i])),
        )
    end
    Nrow = floor(Int, sqrt(Nmom_max))
    Ncol = ceil(Int, sqrt(Nmom_max))
    if Nrow * Ncol < Nmom_max
        Nrow += 1
    end
    plot(
        plt...,
        layout = grid(Nrow, Ncol),
        size = (Ncol * 400, Nrow * 270),
        foreground_color_legend = nothing,
        left_margin = 5Plots.mm,
        bottom_margin = 5Plots.mm,
    )

    savefig(file_name)
end

"""
  plot_spectra(sol, p; file_name = "test_spectra.png", logxrange=(0, 8))

  `sol` - ODE solution
  `p` - additional ODE parameters carried in the solver
Plots the spectra
"""
function plot_spectra!(sol, p; file_name = "test_spectra.png", logxrange = (-15, -3), print = false)
    x = 10 .^ (collect(range(logxrange[1], logxrange[2], 100)))
    r = (x / 1000 * 3 / 4 / π) .^ (1 / 3) * 1e6 # plot in µm

    if print
        @show x
        @show r
    end

    moments = vcat(sol.u'...)
    Ndist = length(p.pdists)
    n_params = [nparams(p.pdists[i]) for i in 1:Ndist]

    plt = Array{Plots.Plot}(undef, 3)
    t_ind = [1, floor(Int, length(sol.t) / 2), length(sol.t)]
    sp_sum = zeros(length(r), 3)

    for i in 1:3
        plt[i] = plot()
        for j in 1:Ndist
            ind_rng = get_dist_moments_ind_range(p.NProgMoms, j)
            moms = moments[t_ind[i], ind_rng]
            pdist_tmp = update_dist_from_moments(p.pdists[j], ntuple(length(moms)) do i
                moms[i]
            end)
            plot!(
                r,
                3 * x .^ 2 .* pdist_tmp.(x),
                linewidth = 2,
                xaxis = :log,
                yaxis = "dm / d(ln r)",
                xlabel = "r",
                label = "Pdist " * string(j),
                title = "time = " * string(round(sol.t[t_ind[i]], sigdigits = 4)),
            )
            sp_sum[:, i] += 3 * x .^ 2 .* pdist_tmp.(x)

            if print
                @show 3 * x .^ 2 .* pdist_tmp.(x)
            end
        end
        plot!(r, sp_sum[:, i], linewidth = 2, linestyle = :dash, linecolor = :black, label = "Sum")
    end

    plot(
        plt...,
        layout = grid(1, 3),
        size = (1200, 270),
        foreground_color_legend = nothing,
        left_margin = 7Plots.mm,
        bottom_margin = 8Plots.mm,
    )

    savefig(file_name)
end

"""
  plot_params!(sol, p; file_name = "box_model_params.pdf")

  `sol` - ODE solution
  `p` - additional ODE parameters carried in the solver
Plots the evolution of particle distribution parameters in time (for normalized moments).
"""
function plot_params!(sol, p; yscale = :log10, file_name = "box_model.pdf")
    time = sol.t
    mom_norms = get_moments_normalizing_factors(p.NProgMoms, p.norms)
    moments = vcat(sol.u'...) ./ collect(mom_norms)'
    params = similar(moments)
    n_dist = length(p.pdists)
    plt = Array{Plots.Plot}(undef, n_dist)
    n_params = [nparams(p.pdists[i]) for i in 1:n_dist]
    for i in 1:n_dist
        ind_rng = get_dist_moments_ind_range(p.NProgMoms, i)
        for j in 1:size(params)[1]
            moms_tmp = moments[j, ind_rng]
            pdist_tmp = CPD.update_dist_from_moments(p.pdists[i], ntuple(length(moms_tmp)) do i
                moms_tmp[i]
            end)
            params[j, ind_rng] = vcat(get_params(pdist_tmp)[2]...)
        end

        plot()
        for j in ind_rng
            plot!(time, params[:, j], linewidth = 2, label = "p_" * string(j - ind_rng[1] + 1), yscale = yscale)
        end
        plt[i] = plot!(xaxis = "time", yaxis = "parameters (mode " * string(i) * ")")
    end
    nrow = floor(Int, sqrt(n_dist))
    ncol = ceil(Int, sqrt(n_dist))
    if nrow * ncol < n_dist
        nrow += 1
    end
    plot(
        plt...,
        layout = grid(nrow, ncol),
        size = (ncol * 400, nrow * 270),
        foreground_color_legend = nothing,
        left_margin = 5Plots.mm,
        bottom_margin = 5Plots.mm,
    )

    savefig(file_name)
end

"""
  print_box_results!(sol, p)

  `sol` - ODE solution
  `p` - additional ODE parameters carried in the solver
Prints the evolution of moments in time, plus the distribution parameters at a few times
"""
function print_box_results!(sol, p)
    time = sol.t
    moments = vcat(sol.u'...)

    Ndist = length(p.pdists)
    Nmom_min = minimum(p.NProgMoms)
    Nmom_max = maximum(p.NProgMoms)
    moments_sum = zeros(length(time), Nmom_min)

    for i in 1:Ndist
        for j in 1:Nmom_min
            ind = get_dist_moment_ind(p.NProgMoms, i, j)
            moments_sum[:, j] += moments[:, ind]
            @show moments[:, ind]
        end
    end
    @show time
    for j in 1:Nmom_min
        @show moments_sum[:, j]
    end

    t_ind = [1, ceil(Int, length(sol.t) / 2), length(sol.t)]
    params = zeros(length(t_ind), Ndist, Nmom_max)
    for i in 1:3
        for j in 1:Ndist
            ind_rng = get_dist_moments_ind_range(p.NProgMoms, j)
            moms = moments[t_ind[i], ind_rng]
            pdist_tmp = update_dist_from_moments(p.pdists[j], ntuple(length(moms)) do i
                moms[i]
            end)
            params[i, j, 1:p.NProgMoms[j]] = vcat(get_params(pdist_tmp)[2]...)
        end
    end
    @show t_ind
    @show params[:, 1, :]
    if Ndist > 1
        @show params[:, 2, :]
    end
end

