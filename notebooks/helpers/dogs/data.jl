function _unwrap_celllike(x)
    if x isa Matrix{Any} && length(x) == 1
        return x[1]
    elseif x isa Vector{Any} && length(x) == 1
        return x[1]
    end
    return x
end

"""
    sensor_table_to_dict(sensor_tbl)

Normalize a dog sensor payload into a `Dict{String,Any}` and unwrap singleton
cell-like containers left over from earlier MATLAB-shaped data structures.
"""
function sensor_table_to_dict(sensor_tbl::Dict)
    return Dict(String(k) => _unwrap_celllike(v) for (k, v) in sensor_tbl)
end

function windowed_mean_std(mat::AbstractMatrix, window::Int)
    window <= 1 && return (mat, zeros(eltype(mat), size(mat)))
    n = size(mat, 1)
    m = n ÷ window
    μ = Matrix{eltype(mat)}(undef, m, size(mat, 2))
    σ = Matrix{eltype(mat)}(undef, m, size(mat, 2))
    for i in 1:m
        r = (window * (i - 1) + 1):(window * i)
        v = @view mat[r, :]
        μ[i, :] .= vec(mean(v; dims=1))
        σ[i, :] .= vec(std(v; dims=1))
    end
    return μ, σ
end


function _annotation_dict_from_codes(names::Vector{String}, codes::AbstractMatrix{<:Integer})
    return Dict{String,Any}(
        "categoryNames" => reshape(Any[n for n in names], :, 1),
        "codes" => Int.(codes),
        "isOrdinal" => false,
        "isProtected" => false,
    )
end

function _canonical_label_names(lookup_df::DataFrame, key::String)
    sub = filter(r -> r.label_key == key && r.code > 0, lookup_df)
    seen = Set{String}()
    out = String[]
    for lbl in String.(sub.label)
        if !(lbl in seen)
            push!(out, lbl)
            push!(seen, lbl)
        end
    end
    if key == "Task"
        out = replace.(out, r"^Task\s+" => "")
    elseif key == "Behavior"
        out = replace.(out, r"^Behavior\s+" => "")
    end
    return out
end

_normalize_label_str(x) = ismissing(x) ? "" : String(x)

function _build_observations(ABack, ANeck, GBack, GNeck, window)
    A = 0.5 .* (ABack .+ ANeck)
    G = 0.5 .* (GBack .+ GNeck)
    A_mean, A_std = windowed_mean_std(A, window)
    G_mean, G_std = windowed_mean_std(G, window)

    eps = 1e-6
    A_std = log.(A_std .+ eps)
    G_std = log.(G_std .+ eps)

    A_mean_z = (A_mean .- mean(A_mean; dims=1)) ./ std(A_mean; dims=1)
    A_std_z = (A_std .- mean(A_std; dims=1)) ./ std(A_std; dims=1)
    G_mean_z = (G_mean .- mean(G_mean; dims=1)) ./ std(G_mean; dims=1)
    G_std_z = (G_std .- mean(G_std; dims=1)) ./ std(G_std; dims=1)
    observations = Matrix(transpose(hcat(A_mean_z, A_std_z, G_mean_z, G_std_z)))

    return (; A_mean_z, A_std_z, G_mean_z, G_std_z, observations)
end

"""
    load_dogmove_multi_csv(raw_csv_path, lookup_csv_path; window=10, dog_selector=:all)

Load and concatenate the processed DogMove CSV exports across one or more dogs,
returning the combined annotations and windowed observation features used by the
multi-dog notebook.
"""
function load_dogmove_multi_csv(raw_csv_path, lookup_csv_path; window=10, dog_selector=:all)
    raw_df = CSV.read(raw_csv_path, DataFrame)
    lookup_df = CSV.read(lookup_csv_path, DataFrame)
    sort!(raw_df, [:trial_index, :t])

    all_dogs = sort(unique(Int.(raw_df.dog_id)))
    selected_dogs = if dog_selector == :all
        all_dogs
    elseif dog_selector isa Integer
        [Int(dog_selector)]
    else
        sort(unique(Int.(collect(dog_selector))))
    end

    dog_set = Set(selected_dogs)
    raw_sel = filter(r -> Int(r.dog_id) in dog_set, raw_df)
    raw_df = nothing  # free the full DataFrame
    nrow(raw_sel) == 0 && error("No rows selected for dog_selector=$(dog_selector)")

    trial_meta = unique(select(raw_sel, [:trial_index, :dog_id, :test_num]))
    sort!(trial_meta, :trial_index)

    # Only store trial metadata — skip per-trial sensor copies to save memory
    DogMoveData = DataFrame(
        DogID=Int.(trial_meta.dog_id),
        TestNum=Int.(trial_meta.test_num),
    )

    ABack = Matrix{Float64}(select(raw_sel, [:A_back_x, :A_back_y, :A_back_z]))
    ANeck = Matrix{Float64}(select(raw_sel, [:A_neck_x, :A_neck_y, :A_neck_z]))
    GBack = Matrix{Float64}(select(raw_sel, [:G_back_x, :G_back_y, :G_back_z]))
    GNeck = Matrix{Float64}(select(raw_sel, [:G_neck_x, :G_neck_y, :G_neck_z]))

    task_names = _canonical_label_names(lookup_df, "Task")
    behavior_names = _canonical_label_names(lookup_df, "Behavior")

    task_map = Dict{String,Int}("Task none" => 0)
    for (i, n) in enumerate(task_names)
        task_map["Task " * n] = i
    end
    behavior_map = Dict{String,Int}("Behavior none" => 0)
    for (i, n) in enumerate(behavior_names)
        behavior_map[n] = i
    end

    for lbl in unique(String.(_normalize_label_str.(raw_sel.task_label)))
        isempty(lbl) && continue
        if !haskey(task_map, lbl)
            push!(task_names, replace(lbl, r"^Task\s+" => ""))
            task_map[lbl] = length(task_names)
        end
    end
    for lbl in unique(vcat(
        String.(_normalize_label_str.(raw_sel.behavior_label_1)),
        String.(_normalize_label_str.(raw_sel.behavior_label_2)),
        String.(_normalize_label_str.(raw_sel.behavior_label_3)),
    ))
        isempty(lbl) && continue
        if lbl != "Behavior none" && !haskey(behavior_map, lbl)
            push!(behavior_names, lbl)
            behavior_map[lbl] = length(behavior_names)
        end
    end

    task_labels = String.(_normalize_label_str.(raw_sel.task_label))
    task_codes = reshape([get(task_map, lbl, 0) for lbl in task_labels], :, 1)
    b1 = String.(_normalize_label_str.(raw_sel.behavior_label_1))
    b2 = String.(_normalize_label_str.(raw_sel.behavior_label_2))
    b3 = String.(_normalize_label_str.(raw_sel.behavior_label_3))
    behavior_codes = hcat(
        [get(behavior_map, lbl, 0) for lbl in b1],
        [get(behavior_map, lbl, 0) for lbl in b2],
        [get(behavior_map, lbl, 0) for lbl in b3],
    )
    raw_sel = nothing  # free filtered DataFrame

    # Only store annotation dicts in sensor_tbl — skip raw sensor matrices to save memory
    sensor_tbl = Dict{String,Any}(
        "Task" => _annotation_dict_from_codes(task_names, task_codes),
        "Behavior" => _annotation_dict_from_codes(behavior_names, behavior_codes),
        "PointEvent" => Dict{String,Any}(
            "categoryNames" => Matrix{Any}(undef, 0, 0),
            "codes" => Matrix{Int}(undef, size(task_codes, 1), 0),
            "isOrdinal" => false,
            "isProtected" => false,
        ),
    )
    sdict = sensor_table_to_dict(sensor_tbl)
    feats = _build_observations(ABack, ANeck, GBack, GNeck, window)
    return merge(
        (; DogMoveData, idx_best=1, sensor_tbl, sdict, WINDOW=window, selected_dog_ids=selected_dogs, selected_trial_count=nrow(trial_meta)),
        feats,
    )
end

extract_categories(col) = col isa Dict ? get(col, "categoryNames", nothing) : nothing

function _annotation_dict(col)
    x = _unwrap_celllike(col)
    x isa Dict && return x
    error("Unsupported annotation object: $(typeof(col))")
end

function _annotation_codes(payload, ncat::Int)
    x = _unwrap_celllike(payload)
    if x isa AbstractMatrix{<:Number}
        if ncat > 0 && size(x, 2) == ncat
            return [argmax(@view x[i, :]) for i in 1:size(x, 1)]
        elseif ncat > 0 && size(x, 1) == ncat
            return [argmax(@view x[:, j]) for j in 1:size(x, 2)]
        elseif size(x, 2) > 1
            return Int.(round.(x))
        else
            return Int.(round.(vec(x)))
        end
    elseif x isa AbstractVector{<:Number}
        return Int.(round.(x))
    elseif x isa Number
        return [Int(round(x))]
    elseif x isa AbstractArray
        out = Int[]
        for y in x
            append!(out, _annotation_codes(y, ncat))
        end
        return out
    else
        return Int[]
    end
end

function _annotation_payload(d::Dict{String,Any})
    payload = nothing
    for k in ("data", "codes", "labels", "values", "value", "idx", "index", "id")
        if haskey(d, k)
            payload = d[k]
            break
        end
    end
    if payload isa Dict
        inner = payload
        for k in ("data", "codes", "labels", "values", "value", "idx", "index", "id")
            if haskey(inner, k)
                payload = inner[k]
                break
            end
        end
    end
    payload === nothing && error("Could not find annotation payload; keys=$(collect(keys(d)))")
    return payload
end

"""
    extract_task_categories_and_codes(col)

Decode task category names and integer task codes from a notebook annotation
payload.
"""
function extract_task_categories_and_codes(col)
    d = _annotation_dict(col)
    names = haskey(d, "categoryNames") ? vec(string.(d["categoryNames"])) : String[]
    payload = _unwrap_celllike(_annotation_payload(d))
    codes = if payload isa AbstractMatrix{<:Number}
        Int.(round.(payload[:, 1]))
    elseif payload isa AbstractVector{<:Number}
        Int.(round.(payload))
    else
        _annotation_codes(payload, length(names))
    end
    return names, codes
end

"""
    segment_task_bouts(observations, sensor_dict; window, min_windows=2)

Segment the observation matrix into per-bout sequences, one for each contiguous
block of a non-"none" task.  Returns a NamedTuple with `sequences`,
`bout_task_codes`, `bout_window_ranges`, and `task_names`.
"""
function segment_task_bouts(observations, sensor_dict; window, min_windows=2)
    task_names, raw_codes = extract_task_categories_and_codes(sensor_dict["Task"])
    n_windows = size(observations, 2)
    wt_codes = window_task_code(raw_codes, n_windows, window)

    sequences = Matrix{Float64}[]
    bout_task_codes = Int[]
    bout_window_ranges = UnitRange{Int}[]

    i = 1
    while i <= length(wt_codes)
        code = wt_codes[i]
        if code == 0
            i += 1
            continue
        end
        j = i
        while j <= length(wt_codes) && wt_codes[j] == code
            j += 1
        end
        if j - i >= min_windows
            push!(sequences, observations[:, i:j-1])
            push!(bout_task_codes, code)
            push!(bout_window_ranges, i:j-1)
        end
        i = j
    end
    return (; sequences, bout_task_codes, bout_window_ranges, task_names)
end


window_task_code(raw_codes, n_windows, window; pad_to_n_windows=false) = begin
    n_codes = raw_codes isa AbstractMatrix ? size(raw_codes, 1) : length(raw_codes)
    n = min(n_codes, n_windows * window)
    m = n ÷ window
    out = pad_to_n_windows ? fill(0, n_windows) : Vector{Int}(undef, m)
    for i in 1:m
        r = (window * (i - 1) + 1):(window * i)
        w = raw_codes isa AbstractMatrix ? vec(@view raw_codes[r, :]) : @view raw_codes[r]
        vals = unique(w)
        cnts = [count(==(v), w) for v in vals]
        out[i] = vals[argmax(cnts)]
    end
    if pad_to_n_windows && m < n_windows && m > 0
        out[(m + 1):end] .= out[m]
    end
    out
end
