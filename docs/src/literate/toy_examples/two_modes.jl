#=
Example from Jérôme Tubiana thesis, Sec. 4.1.2.
=#

using Test: @test, @testset, @inferred
using RestrictedBoltzmannMachines: BinaryRBM, energy, free_energy, pcd!
using RestrictedBoltzmannMachines: wmean, training_epochs, minibatches, log_likelihood, moving_average
import Flux
import Makie
import CairoMakie
import AlgebraOfGraphics as AoG
using LogExpFunctions: softmax
using Statistics: mean

# Dataset with P(0,0) = P(1,1) = 0.48 and P(0,1) = P(1,0) = 0.02.

data = [
    0; 0 ;;
    0; 1 ;;
    1; 0 ;;
    1; 1 ;;
]
wts = [0.48; 0.02; 0.02; 0.48]
data = repeat(data, 1, 4)
wts = repeat(wts, 4)

data = BitArray(
    cat(
        repeat([0, 0], 1, 480),
        repeat([0, 1], 1, 20),
        repeat([1, 0], 1, 20),
        repeat([1, 1], 1, 480);
        dims=2
    )
);

# Training params

batchsize = 10
epochs = training_epochs(; nsamples=size(data)[end], nupdates=100000, batchsize)

# Train RBM with PCD-1

rbm = BinaryRBM(zeros(2), zeros(1), randn(2,1) / 10);
lls_pcd = Float64[]
function callback(; rbm, kw...)
    push!(lls_pcd, mean(log_likelihood(rbm, data)))
end
pcd!(rbm, data; wts, epochs, batchsize, center=false, mode=:pcd, callback, optim=Flux.Descent(0.1));

# Model probabilities

softmax(free_energy(rbm, BitArray([0; 0 ;; 0; 1 ;; 1; 0 ;; 1; 1])))

# Train RBM with exact sampling (extensive enumeration of states).

rbm = BinaryRBM(zeros(2), zeros(1), randn(2,1) / sqrt(2));
lls_exact = Float64[]
function callback(; rbm, kw...)
    push!(lls_exact, mean(log_likelihood(rbm, data)))
end
pcd!(rbm, data; epochs, batchsize, center=true, mode=:exact, callback, optim=Flux.Descent(0.1));

# Model probabilities

softmax(free_energy(rbm, BitArray([0; 0 ;; 0; 1 ;; 1; 0 ;; 1; 1])))

# Plot training likelihood

fig = Makie.Figure()
ax = Makie.Axis(fig[1,1], width=500, height=300)
Makie.lines!(ax, lls_pcd; color=(:blue, 0.25))
Makie.lines!(ax, lls_exact; color=(:red, 0.25))
Makie.lines!(ax, moving_average(lls_pcd, 100); color=:blue)
Makie.lines!(ax, moving_average(lls_exact, 100); color=:red)
Makie.resize_to_layout!(fig)
fig
