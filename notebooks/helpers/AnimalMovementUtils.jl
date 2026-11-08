module AnimalMovementUtils

# Must be set before GR initializes to fix minus sign / Greek char rendering in PDF
ENV["GKS_ENCODING"] = "utf8"

using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics
using Distributions
using LaTeXStrings
using Plots, Measures
using StateSpaceDynamics
using StateSpaceDynamics.ACDC

export haversine_dist, calculate_bearing, prepare_movement_data, get_most_observed_individual
export StepAngleEmission
export CardioidEmission, StepAngleCardEmission
export compute_movement_analysis, compute_movement_analysis_alt, compute_movement_analysis_alt2
export get_movement_plots, get_movement_plots_alt, get_movement_plots_alt2
export feature_xlims_from_res, component_feature_density, movement_state_feature_grid
export write_movement_statistics

export export_movement_state_grids

include("FigureExport.jl")
include("movement/data.jl")
include("movement/emissions.jl")
include("movement/fitting.jl")
include("movement/plots.jl")
include("movement/statistics.jl")

end  # module AnimalMovementUtils
