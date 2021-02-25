##
using DrWatson
@quickactivate "WorldModels"
using LinearAlgebra, Statistics
using NPZ, HDF5
using Parameters:@with_kw
using Plots
using Lazy:@as

# plotlyjs()

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
# model = cVAE(args.z) |> gpu

using BSON
BSON.@load "saved_models/cvae_adam001_z8.bson" modl
model = modl |> gpu

opt = ADAM(0.01)
# train(model, xs, 20, opt)


##
modl = model |> cpu
encoder_μ, encoder_logvar, decoder = modl

zs = []
for i in 1:10
    z, μ, logvar = sample_latent(modl[1:2]..., xs[i], dev=cpu)
    pred = decoder(z)
    pred0 = dropdims(mean(pred, dims=3), dims=3)
    push!(zs, pred0)
end

out = cat(zs..., dims=4)

quick_anim(permutedims(out, [4,1,2,3])[:,:,:,1], fps=2)
# heatmap(pred0[:,:,2])

xs[1]

heatmap(xs[:,:,])
##
modl = model |> cpu

using BSON
# BSON.@save "cvae_adam001_z3_partial.bson" modl

##
BSON.@load "saved_models/cvae_adam001_z8.bson" modl
model = modl |> gpu
##
##
args.z = 8

function rnn_dataset(net, xs)
    zs = [sample_latent(net[1:2]..., k, dev=cpu)[1] for k in xs]
    eind = findfirst(x -> size(x, 2) < args.batchsize, zs)
    zt = collect(partition(zs[1:eind - 1], 10))

    ys_orig = [net[3].(k) for k in zt] # decoder
    Xs = [k[1:end - 1] for k in zt]
    Ys = [k[2:end] for k in ys_orig]

    return Xs, Ys
end

X, Y = rnn_dataset(modl, xs)
# model = modl |> gpu
encoder_μ, encoder_logvar, decoder = model

##
Xs = X |> gpu
Ys = Y |> gpu
##

using ParameterSchedulers
using ParameterSchedulers: Scheduler, Stateful, next!
s = Stateful(TriangleExp(λ0=0.001, λ1=0.2, period=10, γ=0.98))

zRNN = Chain(
    Dense(args.z, 128, relu),
    RNN(128, 64),
    Dense(64, args.z),
    # x -> tanh.(x),
    # Dense(args.z, args.z),
    x -> 3.0f0 * sin.(x) .+ 1.0f0,
) |> gpu

norm1(x) = sum(abs2, x)

function rnn_loss(rnn, decoder, x, y; dev=gpu)
    Flux.reset!(rnn)
    batchsize = 32
    ẑ = rnn.(x) # .+ [0.001f0 * dev(randn(8, batchsize)) for _ in 1:9]
    ŷ = decoder.(ẑ)
    reg_penalty = 0.01f0 * sum(norm1, Flux.params(zRNN))
    loss = mean(Flux.binarycrossentropy.(ŷ, y, agg=sum) ./ batchsize)
    loss + reg_penalty
end

vae = model
encoder_μ, encoder_logvar, decoder = vae

opt = ADAM(0.01)
ps = Flux.params(zRNN)

function train_rnn(rnn, decoder, data, num_epochs, opt; sched=false)
    ps = Flux.params(rnn)
    for epoch in 1:num_epochs
    # for (η, epoch) in zip(s, 1:num_epochs)
        progress_tracker = Progress(length(data), 1, "Training epoch $epoch:")
        # if sched
        #     opt.eta = ParameterSchedulers.next!(s)
        # end
        for (i, (x, y)) in enumerate(data)
            loss, back = pullback(ps) do
                rnn_loss(rnn, decoder, x, y)
            end
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

## ==== Train RNN
train_rnn(zRNN, decoder, zip(Xs, Ys), 1, opt)
##

function sample(net, x; len=10, dev=gpu)
    batchsize = size(x)[end]
    y = net(x) # .+ 0.01f0 * dev(randn(8, batchsize))
    ys = []
    push!(ys, y)
    for i in 2:len
        ŷ = net(y) # .+ 0.01f0 * dev(randn(8, batchsize))
        y = ŷ
        push!(ys, ŷ)
    end
    ys
end
##
BSON.@load  "saved_models/rnn_adam001_128_32_adjustout_noise001.bson" rnn
zRNN = gpu(rnn)

##
dys = Ys |> cpu
dy = cat(mean.(dys[1], dims=3)..., dims=3)
quick_anim(permutedims(dy[:,:,:,1], [3,1,2]))

Flux.reset!(zRNN)
out = cat(sample(zRNN, Xs[1][1])..., dims=3) |> cpu
heatmap(out[:,1,:])

preds0 = decoder.(sample(zRNN, Xs[1][1]))
preds = cat(mean.(preds0, dims=3)..., dims=3) |> cpu

quick_anim(permutedims(preds[:,:,:,1], [3,1,2]))
##

encoder_μ, encoder_logvar, decoder =  model



rnn = cpu(zRNN)
# BSON.@save "saved_models/rnn_adam001_128_32_adjustout_noise001_sin2.bson" rnn

zRNN

Xs[1][1]

decoder.(sample(zRNN, Xs[1][1]))

rnn_loss(zRNN, decoder, x, y)

loss, back = pullback(ps) do
    rnn_loss(zRNN, decoder, x, y)
end


opt = ADAM(0.01)
gradients = back(1f0)
Flux.update!(opt, ps, gradients)

ps = Flux.params(zRNN)

x, y = Xs[1], Ys[1]

length(ps)