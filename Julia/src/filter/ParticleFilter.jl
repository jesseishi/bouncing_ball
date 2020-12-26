# A simple particle filter, the bootstrap filter.
module ParticleFilter
export nothing

using Random
using Statistics
using LinearAlgebra
using StatsBase

# We're going to cheat a little bit and give the particle filter access to the
#  'exact' dynamics of the ball. TODO: Make a simplified ball model that the
#  particle filter must use.
include("../ball/Ball.jl")
using .Ball


struct State
    pos_hat::Vector{Float64}
    particles_state::Vector{Ball.State}
    particles_params::Vector{Ball.Params}
    weights::Vector{Float64}
end

struct Params
    N::Int64                # Number of particles [-].
    σ_pos::Vector{Float64}  # Initial standard deviation for position perturbation [m].
    σ_vel::Vector{Float64}  # Initial standard deviation for velocity perturbation [m].
    λ::Float64              # Tuning parameter for regularization.
end
params() = Params(500, [0.2, 0.2], [0.2, 0.2], 0.1)

# Initialize with N particles that are slightly perturbed and have normalized weights.
function init(pos_star, params::Params)

    # TODO: regularization on state _and_ params maybe.
    particles_state = [perturb(Ball.State(pos_star, [0.8, 0]), params) for _ = 1:params.N]
    particles_params = [perturb(Ball.params(), params) for _ = 1:params.N]

    weights = ones(params.N) / params.N

    # Make an intermediate state of the filter to get a position estimate.
    state = State([0, 0], particles_state, particles_params, weights)
    state = estimate_pos_hat(state)

    # Return the initial state of the filter.
    return state
end

# Step the whole particle filter and return the new state of the filter.
function step(state::State, params::Params, pos_star, Δt)

    # Each subfunction returns an entire new state of the particle filter. This
    #  is very handy when testing different parts and gives it a nice structure.
    # It might be nicer to make it a mutable state.

    # Propagate each particle.
    state  = propagate(state, Δt)

    # Recalculate the weights.
    state = calculate_weights(state, pos_star)

    # Make an estimate and this state for plotting.
    state = estimate_pos_hat(state)
    state_before_resample = state

    # Resample particles and reset the weights.
    state = resample(state, params)

    # Regularize particles by giving them a random perturbation.
    state = regularize(state, params, pos_star)

    return state, state_before_resample
end

# Propagate each particle.
function propagate(state::State, Δt)

    particles_state = state.particles_state
    for (i, (particle_state, particle_params)) in enumerate(zip(state.particles_state, state.particles_params))

        # We want a smaller update size for the particles than this filter, so step it 10 times.
        particles_state[i] = Ball.step(particle_state, particle_params, Δt, Δt/10)

    end

    # Return the new state of the filter.
    return State(state.pos_hat, particles_state, state.particles_params, state.weights)
end

# Recalculate the weights based on a new measurement.
function calculate_weights(state::State, pos_star)

    # Calculate the distance between each particle and the measurement.
    distances = [norm(particle.pos - pos_star) for particle in state.particles_state]

    # The higher the distance the lower the weight should be.
    # extra_distances = distances .- maximum(distances)
    # weights = exp.(-extra_distances)
    weights = maximum(distances) .- distances

    # Normalize the weights so they sum to 1.
    normalize!(weights, 1)

    # Return the updated state of the particle filter.
    return State(state.pos_hat, state.particles_state, state.particles_params, weights)
end

function resample(state::State, params::Params)

    # Pick random particles according to their weight. A particle can be picked multiple times.
    particles_state = sample(state.particles_state, Weights(state.weights), params.N, replace=true)

    # Reset the weights.
    weights = ones(params.N) / params.N

    # Return the new state of the filter.
    return State(state.pos_hat, particles_state, state.particles_params, state.weights)
end

# Regularizing.
function regularize(state::State, params::Params, pos_star)

    # Calculate the sample covariance, this is a 2x2 matrix relating the covariance
    # of the x positions and y positions.
    # E.g. this makes the spread around x higher than around y if there's generally more error in x.
    Pk = 1/(params.N - 1) * sum([(particle.pos - pos_star) * (particle.pos - pos_star)'  for particle in state.particles_state])

    # To make gaussian random draws from this we take random draws with unit covariance
    # and multiply them with the cholesky factor of Pk, for this we take the lower triangular matrix C.
    try
        C = cholesky(Pk)
    catch e
        println(pos_star)
        println(Pk)
    end
    C = cholesky(Pk)
    random_perturbations = C.U' * randn(2, params.N)

    # We now multiply these perturbations by a small tuning parameter λ.
    # Not sure why or how to choose this parameter.
    random_perturbations *= params.λ

    # Add the random perturbations.
    particles_state = [particle + Ball.State(perturbation, [0, 0]) for (particle, perturbation)
                       in zip(state.particles_state, eachcol(random_perturbations))]

    # Return the updated state of the filter.
    return State(state.pos_hat, particles_state, state.particles_params, state.weights)
end

# Perturb a particle slightly with some change in pos and vel.
function perturb(particle::Ball.State, params::Params)

    Δpos = params.σ_pos .* randn(2)

    # The velocity perturbation is proportional to the velocity.
    Δvel = particle.vel .* params.σ_vel .* randn(2)

    return particle + Ball.State(Δpos, Δvel)
end

# TODO: regularization function for ball parameters.
function perturb(p_params::Ball.Params, params::Params)
    return p_params
end

# A simple weighted average will provide the estimate.
function estimate_pos_hat(state::State)

    # Not the most beautiful line. The [1] is needed to reduce an array of arrays to a vector.
    # We take the sum and not the mean because the weights add up to 1.
    pos_hat = sum([particle.pos * w for (particle, w) in zip(state.particles_state, state.weights)], dims=1)[1]

    # Return the updated filter state.
    return State(pos_hat, state.particles_state, state.particles_params, state.weights)
end

# A function to help with logging the data.
function get_particles_data(state::State, params::Params)

    # We make an array with rows for x-positions, y-positions, and weights of each particle.
    particles_data = Array{Float64, 2}(undef, 3, params.N)

    particles_data[1, :] = [particle.pos[1] for particle in state.particles_state]
    particles_data[2, :] = [particle.pos[2] for particle in state.particles_state]
    particles_data[3, :] = state.weights

    return particles_data
end


end  # Module ParticleFilter