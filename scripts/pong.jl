##
using DrWatson
@quickactivate "WorldModels"
using LinearAlgebra, Statistics
using JLD2
using Parameters:@with_kw
using Plots
using Lazy:@as

theme(:vibrant)

includet(srcdir("utils.jl"))
includet(srcdir("cvae.jl"))

@load datadir("exp_pro", "pong_data.jld2") pong_data

using Flux, Zygote
using Flux:stack, update!
using CUDA
device!(1)
CUDA.allowscalar(false)
using ProgressMeter: Progress, next!
using .cvae

##

@with_kw mutable struct Args
    batchsize::Int = 32
    z::Int = 10
end

args = Args()

num_test = 4 # number of validation samples

x_train = pong_data["obs"][1:end - num_test] |> x -> vcat(x...)
x_test = pong_data["obs"][end - num_test + 1:end] |> x -> vcat(x...)

y_train = pong_data["next_obs"][1:end - num_test] |> x -> vcat(x...)
y_test = pong_data["next_obs"][end - num_test + 1:end] |> x -> vcat(x...)

xx = x_train

thresh(x) = [x .> 0.02f0] .= 1.0f0

histogram(xx[1][:])


heatmap(xx[1][:,:,1,1])

##

function train(model, data, num_epochs, opt)
    encoder_μ, encoder_logvar, decoder = model
    ps = Flux.params(encoder_μ, encoder_logvar, decoder)
    dev = gpu

    for epoch in 1:num_epochs
        progress_tracker = Progress(length(data), 1, "Training epoch $epoch:")
        for (i, x) in enumerate(data)
            x = x |> dev
            loss, back = pullback(ps) do
                vae_loss(encoder_μ, encoder_logvar, decoder, x, λ=0.1f0)
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

args.z = 4
device_reset!(device())
model = cVAE(args.z) |> gpu
encoder_μ, encoder_logvar, decoder = model

opt = ADAM(0.001)

train(model, x_train, 100, opt)


##
modl = model |> cpu
encoder_μ, encoder_logvar, decoder = modl
z, μ, logvar = sample_latent(modl[1:2]..., x_train[10], dev=cpu)

pred = decoder(z)
pred0 = pred[:,:,2,:]

heatmap(pred0[:,:,1])

##
modl = model |> cpu

using BSON
# BSON.@save "cvae_adam001_z3_partial.bson" modl

##
BSON.@load "cvae_adam001_z8.bson" modl

