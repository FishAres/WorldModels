##
using DrWatson
@quickactivate "WorldModels"
using LinearAlgebra, Statistics
using NPZ, HDF5
using Parameters:@with_kw
using Plots
using Lazy:@as

plotlyjs()

theme(:juno)

includet(srcdir("utils.jl"))
includet(srcdir("cvae.jl"))

action_data = h5open(datadir("exp_raw", "ball2.h5"))
img_data = npzread(datadir("exp_raw", "ball2.h5.npz"))

using Flux, Zygote
using Flux:stack, update!
using CUDA
device!(2)
CUDA.allowscalar(false)

using Base.Iterators:partition
using ProgressMeter: Progress, next!
using .cvae

function process_data(data, batchsize; keepseq=false)
    data = Float32.(data)
    # cat(normalize.(collect(eachslice(xs[1], dims=4)))..., dims=4)
    data[data .> 0.0] .= 1.0f0
    x = @as data begin
    map(x -> stack(x, 5), partition(eachslice(data, dims=1), batchsize))
    map(x -> collect(eachslice(x, dims=1)), data)
    end
    if !keepseq
        x = vcat(x...)
    end
    return x
end

##

@with_kw mutable struct Args
    batchsize::Int = 32
    z::Int = 10
end

args = Args()

x_train = process_data(img_data["train_x"], args.batchsize)
x_test = process_data(img_data["test_x"], 2)

##

function train(model, data, num_epochs, opt)
    # encoder_μ, encoder_logvar, decoder = map(gpu, model)
    encoder_μ, encoder_logvar, decoder = model
    ps = Flux.params(encoder_μ, encoder_logvar, decoder)
    dev = gpu

    for epoch in 1:num_epochs
        progress_tracker = Progress(length(data), 1, "Training epoch $epoch:")
        for (i, x) in enumerate(data)
            x = x |> dev
            loss, back = pullback(ps) do
                vae_loss(encoder_μ, encoder_logvar, decoder, x)
            end

            x = nothing
            gradients = back(1f0)
            Flux.update!(opt, ps, gradients)
            if isnan(loss)
                println("NaN encountered")
                break
            end
            next!(progress_tracker, showvalues=[(:loss, loss)])
        end
    end
    println("Training done!")
end

##
# stuff
xs = [Array(x) for x in x_train]
##

args.z = 3
device_reset!(device())
model = cVAE(args.z) |> gpu

opt = ADAM(0.01)
train(model, xs, 20, opt)


##
modl = model |> cpu
encoder_μ, encoder_logvar, decoder = modl
z, μ, logvar = sample_latent(modl[1:2]..., xs[1], dev=cpu)

pred = decoder(z)
pred0 = pred[:,:,1,:]

heatmap(pred0[:,:,2])


##
modl = model |> cpu

using BSON
# BSON.@save "cvae_adam001_z3_partial.bson" modl

##
BSON.@load "cvae_adam001_z8.bson" modl


##
encoder_μ, encoder_logvar, decoder = modl
z, μ, logvar = sample_latent(modl[1:2]..., xs[1], dev=cpu)

N = 100
args.z = 8
zRNN = Chain(
    GRU(args.z, args.z)
) |> gpu

eind = findfirst(x -> size(x, 2) < 32, zs)

# zs = [Flux.unsqueeze(sample_latent(modl[1:2]..., k, dev=cpu)[1], 1) for k in xs]
zs = [sample_latent(modl[1:2]..., k, dev=cpu)[1] for k in xs]
zc = cat(zs[1:eind - 1]..., dims=1)

zt = collect(partition(zs[1:eind - 1], 10))
zt[1]

out = zRNN.(zt[1])

ŷ = cat(out..., dims=3)

ys_orig = [decoder.(k) for k in zt]

xys = [k[1:end - 1] for k in zt]
yys = [k[2:end] for k in ys_orig]

model = modl |> gpu

encoder_μ, encoder_logvar, decoder = model

function rnn_loss(x, y)
    batchsize = 32
    ẑ = zRNN.(x)
    ŷ = decoder.(ẑ)
    sum(Flux.binarycrossentropy.(ŷ, y, agg=sum) ./ batchsize)
end

rnn_loss(xys[1] |> gpu, yys[1] |> gpu)

Xs = xys |> gpu
Ys = yys |> gpu

opt = ADAM(0.01)
ps = Flux.params(zRNN)



Flux.train!((x, y) -> rnn_loss(x, y), ps, zip(Xs, Ys), opt)

Y = zRNN(Xs[1][1]) |> decoder |> cpu

heatmap(dropdims(mean(Y[:,:,:,1], dims=3), dims=3))

# == Make training dataset for RNN


##
N = 80
ys = 3sin.(0.4 * (1:N)) .+ 1.
# plot!(ys)

zn = zeros(Float32, 8, N, 1)
zn[8,:] = ys
x̂s = cat(decoder.(eachslice(zn[:,:,:], dims=2))..., dims=4)
xsn = permutedims(dropdims(mean(x̂s, dims=3), dims=3), [3,1,2])

quick_anim(xsn, fps=4)
##
