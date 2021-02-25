using DrWatson
using PyCall
using JLD2
using Base.Iterators:partition
using Flux:stack

py"""
import h5py
def load_list_dict_h5py(fname):
    "Create dict of [obs, act, newobs] for each timestep"
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
    return array_dict
"""

data = py"load_list_dict_h5py"("data/exp_raw/pong_train.h5")

function process_obs(data, key, batchsize; keepseq=false)
    obs = map(x -> Float32.(x[key][:,1:3,:,:,:]), data)
    # Partition to avoid stack overflow
    # (improve)
    out = []
    if length(obs) > 1000
        ds = partition(obs, 1000)
    else
        ds = obs
    end
    for preobs in partition(ds, 100)
        obsc = cat(preobs..., dims=5)
        obsc = map(x -> stack(x, 5), partition(eachslice(obsc, dims=5), batchsize))
        obsc = map(x -> collect(eachslice(x, dims=1)), obsc)
        push!(out, obsc)
    end
    out = vcat(out...)
    out = [map(x -> permutedims(x, [2,3,1,4]), k) for k in out]
    out
end

function process_data(datafile, batchsize, savestring)
    data = py"load_list_dict_h5py"(datafile)
    data_out = Dict()
    for key in ["obs", "next_obs"]
        data_out[key] = process_obs(data, key, batchsize)
    end
    data_out["action"] = map(x -> x["action"], data)

    return data_out
end

cube_data = process_data(datadir("exp_raw", "cubes_train.h5"), 32, "cubes_train")
@save datadir("exp_pro", "cube_train.jld2") cube_data
