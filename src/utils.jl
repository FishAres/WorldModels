using Plots
using Flux


## Plotting functions
function quick_anim(data; fps=2, savestring="gabagoool.gif")
    plt = plot()
    anim = @animate for t = 1:size(data, 1)
        heatmap!(data[t, :, :], color=:greys, colorbar=:none)
    end
    gif(anim, savestring, fps=fps)
end


## NN functions
struct Reshape
    shape
end
Reshape(args...) = Reshape(args)
(r::Reshape)(x) = reshape(x, r.shape)
Flux.@functor Reshape ()

## squeeze mean

"Mean but drops the dimension it operates on"
dropmean(x, dim) = dropdims(mean(x, dims=dim), dims=dim)