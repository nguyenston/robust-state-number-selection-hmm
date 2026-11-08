module DogNotebookUtils

using CSV
using Dates
using Printf
using DataFrames
using LinearAlgebra
using Random
using Serialization
using Statistics
using Plots
using StatsPlots
using Hungarian
using StateSpaceDynamics
using StateSpaceDynamics.ACDC

export load_dogmove_multi_csv, segment_task_bouts,
    compute_bouthmm_models_allruns, run_bouthmm_selection_analysis_allruns,
    plot_bouthmm_selection, plot_bouthmm_pairs,
    write_bouthmm_statistics, write_bouthmm_pair_statistics

include("FigureExport.jl")
include("dogs/data.jl")
include("dogs/fitting.jl")
include("dogs/alignment.jl")
include("dogs/layout.jl")
include("dogs/plots.jl")
include("dogs/statistics.jl")

end  # module DogNotebookUtils
