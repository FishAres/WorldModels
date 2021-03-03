module cvae

export cVAE, sample_latent, vae_loss # , Reshape

using LinearAlgebra, Statistics
using Flux
using Flux:logitbinarycrossentropy
using CUDA:device, CuDevice
using Revise, DrWatson

includet(srcdir("utils.jl"))

struct Reshape
    shape
end
Reshape(args...) = Reshape(args)
(r::Reshape)(x) = reshape(x, r.shape)
Flux.@functor Reshape ()


function cVAE(z)
    encoder = Chain(
        Conv((4, 4), 3 => 16, relu, stride=(2, 2)),
        Conv((4, 4), 16 => 32, relu, stride=(2, 2)),
        Conv((4, 4), 32 => 128, relu, stride=(2, 2)),
        Flux.flatten,
        Dense(2048, 256, relu)
    )

    encoder_μ = Chain(encoder, Dense(256, z))
    encoder_logvar = Chain(encoder, Dense(256, z))

    decoder = Chain(
        Dense(z, 256),
        BatchNorm(256, relu),
        Dense(256, 2 * 256),
        BatchNorm(2 * 256, relu),
        Reshape(4, 4, 32, :),
        ConvTranspose((5, 5), 32 => 32, stride=(2, 2)),
        BatchNorm(32, relu),
        ConvTranspose((4, 4), 32 => 16, stride=(2, 2)),
        BatchNorm(16, relu),
        ConvTranspose((4, 4), 16 => 3, σ, stride=(2, 2))
    )
    return encoder_μ, encoder_logvar, decoder
end



function sample_latent(encoder_μ, encoder_logvar, x; dev=gpu)
    μ = encoder_μ(x)
    logvar = encoder_logvar(x)
    z = μ + dev(0.1f0 * randn(Float32, size(μ))) .* exp.(logvar)
    return z, μ, logvar
end

function vae_loss(encoder_μ, encoder_logvar, decoder, x; β=1.0f0, λ=0.01f0)
    batchsize = size(x)[end]
    z, μ, logvar = sample_latent(encoder_μ, encoder_logvar, x)
    x̂ = decoder(z)
    # reconstruction loss
    logp_xz = -(Flux.binarycrossentropy(x̂,
     x, agg=sum))  / batchsize
    # kldiv loss
    kl_qp = β * sum(@. (exp(2f0 * logvar) + μ^2 - 2f0 * logvar - 1f0)) / batchsize
    # weight regularizxation (L2 norm)
    reg = λ * norm(Flux.params(decoder))
    # elbo = -logp_xz + kl_qp
    # return elbo + reg
    return logp_xz, kl_qp, reg

end



end