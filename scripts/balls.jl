##
using DrWatson
@quickactivate "WorldModels"
using LinearAlgebra, Statistics
using NPZ, HDF5
using Parameters:@with_kw
using Plots
using Lazy:@as

gr()

theme(:juno)

includet(srcdir("utils.jl"))
includet(srcdir("cvae.jl"))

# data = h5open(datadir("exp_raw", "balls_train.h5"))
img_data = npzread(datadir("exp_raw", "balls_train_denser_long.h5.npz"))

using Flux, Zygote
using Flux:stack, update!
using CUDA
device!(0)
CUDA.allowscalar(false)

using Base.Iterators:partition
using ProgressMeter: Progress, next!
using ProgressMeter
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

x_train = @as xs img_data["train_x"] begin
    process_data(xs, args.batchsize, keepseq=true)
    [Array(x) for x in xs]
    # shuffle
end

xtrain_seq = x_train

x_test = @as xs img_data["test_x"] begin
    process_data(xs, args.batchsize)
    [Array(x) for x in xs]
end

using Random:shuffle
x_train = xx

xt = x_test
# xx = pong_data["obs"][end - num_test + 1:end] |> x -> vcat(x...)
using Images
img = colorview(RGB, permutedims(xx[2][:,:,:,1], [3,1,2]));
heatmap(img)
##
using ParameterSchedulers
using ParameterSchedulers: Scheduler, Stateful, next!
s = Stateful(TriangleExp(λ0=0.001, λ1=0.02, period=10, γ=0.98))

function test(model, data; dev=gpu, hps=hyperparams)
    encoder_μ, encoder_logvar, decoder = model
    L = length(data)
    avg_frac = 1.0f0 / L
    loss = 0.0f0
    for (i, x) in enumerate(data)
        x = x |> dev
        logp, klqp, reg = vae_loss(encoder_μ,
                                   encoder_logvar,
                                   decoder, x,
                                   β=hps[:β], λ=hps[:λ])
        loss += avg_frac * (-logp + klqp + reg)
    end
    return loss
end

@inline function train(
    model, data,
    num_epochs, opt;
    hps=hyperparams,
    schedule_lr=false,
    scd=Stateful(TriangleExp(λ0=0.001, λ1=0.02, period=10, γ=0.98)))
    # Function begins here
    encoder_μ, encoder_logvar, decoder = model
    ps = Flux.params(encoder_μ, encoder_logvar, decoder)
    train_data, test_data = data
    dev = gpu
    KL, LogQP, R = [], [], []
    for epoch in 1:num_epochs
        if schedule_lr
            opt.eta = ParameterSchedulers.next!(scd)
        end
        # kls, logps, regs = [], [], []
        stop_training = false

        @time loss, vlosses, stop_training = train_epoch(
            model, opt, ps, train_data, hps,
            epoch=epoch, stop_training=stop_training)


        if isnan(loss)
            println("NaN encountered")
            break
        end
        if stop_training
            println("Zero KL")
            stop_training = false
            break
        end

        push!(KL, vlosses[1])
        push!(LogQP, vlosses[2])
        push!(R, vlosses[3])
        # Test Loss
        test_loss = test(model, test_data, hps=hps, dev=dev)
        @info ("Epoch $(s.state) loss: $test_loss")

        if scd.state % 10 == 0
            plot_output(model, x_test[2], hps[:filename], scd.state)
        end
    end

    println("Training done!")
    return KL, LogQP, R
end

function train_epoch(model, opt, ps, train_data, hps; epoch=1, dev=gpu, stop_training=false)
    kls, logps, regs = [], [], []
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch:")
    encoder_μ, encoder_logvar, decoder = model
    for (i, x) in enumerate(train_data)
        x = x |> dev
        loss, back = pullback(ps) do
            logp, klqp, reg = vae_loss(
                encoder_μ,
                encoder_logvar,
                decoder, x,
                β=hps[:β], λ=hps[:λ])
            Zygote.ignore() do
                if klqp ≈ 0.0f0
                    stop_training = true
                    println("Zero KlQP")
                end
                push!(kls, klqp |> cpu)
                push!(logps, logp |> cpu)
                push!(regs, reg |> cpu)
            end
            -logp + klqp + reg
        end
        x = nothing
        gradients = back(1f0)
        Flux.update!(opt, ps, gradients)
        ProgressMeter.next!(progress_tracker, showvalues=[(:loss, loss)])
    end
    return loss, [kls, logps, regs], stop_training

end

##
args.z = 64
device_reset!(device())
model = cVAE(args.z) |> gpu
encoder_μ, encoder_logvar, decoder = model

##
hp = Dict(
    :β => 20.0f0,
    :λ => 0.01f0,
    :filename => "balls_16_triexp_sched",
)

opt = ADAM(0.01)

s = Stateful(TriangleExp(λ0=0.00001, λ1=0.002, period=10, γ=0.96))
# plot(s.schedule.(0:100))
##
KL, LogQP, R = train(model, [x_train, x_test], 2, opt,
                     hps=hp, schedule_lr=true, scd=s,)


loss, vlosses, stop_training = train_epoch(model, opt, Flux.params(encoder_μ, encoder_logvar, decoder), x_train, hp)

plot(vcat(KL...))
plot(vcat(KL...) - vcat(LogQP...), label="likelihood")

plot(vcat(R...))
##
function plot_output(model, xs, filename, epoch; ind=1, dev=cpu)
    modl = model |> cpu
    encoder_μ, encoder_logvar, decoder = modl
    z, μ, logvar = sample_latent(modl[1:2]..., xs, dev=dev)
    pred = decoder(z)
    y = pred[:,:,:,ind]
    yimg = colorview(RGB, permutedims(y, [3,1,2]))
    p = plot(yimg)
    show(p)
    savefig(p, "plots/VAEs/$(filename)_epoch_$(epoch).pdf")
end

plot_output(model, xx[2], "tempus", 1, ind=3)

##
modl = model |> cpu
encoder_μ, encoder_logvar, decoder = modl
z, μ, logvar = sample_latent(modl[1:2]..., x_train[1], dev=cpu)
pred = decoder(z)
##
y = pred[:,:,:,1]
yimg = colorview(RGB, permutedims(y, [3,1,2]))
plot(yimg)

##
x = x_test[2][:,:,:,2]
img = colorview(RGB, permutedims(x, [3,1,2]))
plot(img)

##
using BSON
BSON.@save "saved_models/balls_cvae_z16_4eps_adam_cycled_beta20_v.bson" modl

##

plot(z')

modelpath = "saved_models/cubes_cvae_z32_5eps_adam_0001_beta1_5.bson"
BSON.@load modelpath modl

model |> cpu

modl = model |> cpu
encoder_μ, encoder_logvar, decoder = modl
z, μ, logvar = sample_latent(modl[1:2]..., xx[2], dev=cpu)
pred = decoder(z)
y = pred[:,:,:,3]
yimg = colorview(RGB, permutedims(y, [3,1,2]))
p = plot(yimg)

x_train[1]