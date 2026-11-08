# =============================================================================
# GPS feature computation
# =============================================================================

"""Haversine distance in metres between two (lon, lat) points."""
function haversine_dist(lon1, lat1, lon2, lat2)
    R = 6_371_000.0
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    a = sin(dlat / 2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon / 2)^2
    return R * 2 * atan(sqrt(a), sqrt(1 - a))
end

"""Initial compass bearing (radians) from (lon1, lat1) to (lon2, lat2)."""
function calculate_bearing(lon1, lat1, lon2, lat2)
    dlon   = deg2rad(lon2 - lon1)
    lat1_r = deg2rad(lat1)
    lat2_r = deg2rad(lat2)
    y = sin(dlon) * cos(lat2_r)
    x = cos(lat1_r) * sin(lat2_r) - sin(lat1_r) * cos(lat2_r) * cos(dlon)
    return atan(y, x)
end

"""
    prepare_movement_data(df_raw) -> DataFrame

Compute step_length (m) and turn_angle (rad in [-π, π]) for every fix within each
individual's track, then drop rows with missing values. Near-zero steps are clamped
to 1e-5 m to prevent log(0).
"""
function prepare_movement_data(df_raw::DataFrame)
    df = copy(df_raw)
    sort!(df, ["individual-local-identifier", "timestamp"])

    n      = nrow(df)
    steps  = Vector{Union{Float64,Missing}}(missing, n)
    angles = Vector{Union{Float64,Missing}}(missing, n)

    for group in groupby(df, "individual-local-identifier")
        idx = parentindices(group)[1]
        for k in 1:(length(idx) - 1)
            i, ip1 = idx[k], idx[k + 1]
            lon_t,   lat_t   = df[i,   "location-long"], df[i,   "location-lat"]
            lon_tp1, lat_tp1 = df[ip1, "location-long"], df[ip1, "location-lat"]
            any(ismissing, (lon_t, lat_t, lon_tp1, lat_tp1)) && continue

            steps[i] = haversine_dist(lon_t, lat_t, lon_tp1, lat_tp1)

            if k > 1
                im1 = idx[k - 1]
                lon_tm1, lat_tm1 = df[im1, "location-long"], df[im1, "location-lat"]
                if !any(ismissing, (lon_tm1, lat_tm1))
                    b1  = calculate_bearing(lon_tm1, lat_tm1, lon_t, lat_t)
                    b2  = calculate_bearing(lon_t, lat_t, lon_tp1, lat_tp1)
                    ang = b2 - b1
                    ang >  π && (ang -= 2π)
                    ang <= -π && (ang += 2π)
                    angles[i] = ang
                end
            end
        end
    end

    df.step_length = steps
    df.turn_angle  = angles
    df_clean = dropmissing(df, [:step_length, :turn_angle])
    df_clean.step_length = [s <= 1e-6 ? 1e-5 : s for s in df_clean.step_length]
    return df_clean
end

"""
    get_most_observed_individual(df) -> (id::String, df_individual::DataFrame)

Select the individual-local-identifier with the most GPS fixes.
"""
function get_most_observed_individual(df::DataFrame)
    counts  = combine(groupby(df, "individual-local-identifier"), nrow => :count)
    best_id = counts[argmax(counts.count), "individual-local-identifier"]
    df_ind  = filter(row -> row["individual-local-identifier"] == best_id, df)
    println("Selected individual: $best_id  ($(nrow(df_ind)) observations)")
    return String(best_id), df_ind
end

function _timestamp_format()
    return dateformat"yyyy-mm-dd HH:MM:SS.s"
end

function _split_cleaned_individual_trials(
    df_animal::DataFrame; gap_threshold_hours::Float64=1.5
)
    n = nrow(df_animal)
    n == 0 && return DataFrame[], Int[], Float64[]

    ts = DateTime.(df_animal[:, "timestamp"], Ref(_timestamp_format()))
    gap_hours = Dates.value.(diff(ts)) ./ (1000 * 60 * 60)
    break_idxs = findall(>(gap_threshold_hours), gap_hours)

    trials = DataFrame[]
    start_idx = 1
    for b in break_idxs
        stop_idx = b - 1
        if start_idx <= stop_idx
            push!(trials, df_animal[start_idx:stop_idx, :])
        end
        # Rows b and b+1 contain step/angle features that cross the long gap, so
        # exclude them from the trialized fit.
        start_idx = b + 2
    end
    if start_idx <= n
        push!(trials, df_animal[start_idx:n, :])
    end

    return trials, break_idxs, gap_hours[break_idxs]
end

_movement_fit_data(res) = hasproperty(res.data, :obs_trials) ? res.data.obs_trials : res.data.obs

function _movement_states(res, model)
    fit_data = _movement_fit_data(res)
    if fit_data isa Vector
        return vcat(viterbi(model, fit_data)...)
    else
        return viterbi(model, fit_data)
    end
end

function _entropy_class_probs(γ)
    if γ isa Vector
        return sum(-sum(p > 0 ? p * log(p) : 0.0 for p in γ_trial) for γ_trial in γ)
    else
        return -sum(p > 0 ? p * log(p) : 0.0 for p in γ)
    end
end
