module SpikeTrainHMMUtils

ENV["GKS_ENCODING"] = "utf8"

using Dates
using LinearAlgebra
using Printf
using Random
using Serialization
using SpecialFunctions: loggamma
using Statistics
using Distributions
using LaTeXStrings
using Plots, Measures
using StatsPlots: groupedbar
using StateSpaceDynamics
using StateSpaceDynamics.ACDC
using StateSpaceDynamics: viterbi

include("FigureExport.jl")

export PoissonEmission
export generate_synthetic_spikes, generate_synthetic_spikes_negbin, describe_ground_truth
export compute_spike_analysis_poisson
export get_spike_plots, write_spike_statistics
export plot_spike_composition_grid

include("spikes/emission.jl")
include("spikes/generation.jl")
include("spikes/fitting.jl")
include("spikes/plots.jl")
include("spikes/statistics.jl")

end  # module SpikeTrainHMMUtils
