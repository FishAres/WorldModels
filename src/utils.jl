using Plots
using Flux

## Plotting functions
function quick_anim(data; fps=2, savestring="digabagooool.gif")
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



pypath = "/home/ares/miniconda3/envs/worldmodels/bin/python"

ENV["PYTHON"] = pypath

using PyCall
using Plots

py"""
import h5py

def load_list_dict_h5py(fname):

    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
    return array_dict

tmp = load_list_dict_h5py("data/exp_raw/ball2.h5")

"""

tmp = py"load_list_dict_h5py"("data/exp_raw/shapes_train.h5")

using Images

tmp[20]["next_obs"]
heatmap(tmp[1]["obs"][3,1,:,:])

a = tmp[4]["obs"][:,3,:,:]
quick_anim(a[1:20,:,:], fps=4)


heatmap(tmp[1]["next_obs"][1,2,:,:])

tmp[1]["action"]