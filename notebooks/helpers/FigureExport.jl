# Figure export functions extracted unchanged from AnimalMovementUtils.jl.

function _trim_saved_figure!(path::AbstractString)
    if endswith(path, ".png")
        magick = Sys.which("magick")
        isnothing(magick) && return
        tmp = path * ".trim.png"
        try
            run(`$magick $path -trim -bordercolor white -border 12x12 +repage $tmp`)
            mv(tmp, path; force = true)
        catch
            isfile(tmp) && rm(tmp; force = true)
        end
    elseif endswith(path, ".pdf")
        pdfcrop = Sys.which("pdfcrop")
        isnothing(pdfcrop) && return
        tmp = path * ".crop.pdf"
        try
            run(`$pdfcrop --margins 6 $path $tmp`)
            mv(tmp, path; force = true)
        catch
            isfile(tmp) && rm(tmp; force = true)
        end
    end
    return
end

"""Save a plot as both PNG (raster) and PDF (vector) for manuscript use."""
function _save_png_pdf(p, dir::AbstractString, stem::AbstractString; trim::Bool = false)
    mkpath(dir)
    png_path = joinpath(dir, stem * ".png")
    pdf_path = joinpath(dir, stem * ".pdf")
    savefig(p, png_path)
    savefig(p, pdf_path)
    if trim
        _trim_saved_figure!(png_path)
        _trim_saved_figure!(pdf_path)
    end
end
