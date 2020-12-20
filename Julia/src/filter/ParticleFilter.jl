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
    σ_pos::Vector{Float64}  # Standard deviation for position perturbation [m].
    σ_vel::Vector{Float64}  # Standard deviation for velocity perturbation [m].
end
params() = Params(1000, [0.2, 0.2], [0.1, 0.1])

# Initialize with N particles that are slightly perturbed and have normalized weights.
function init(pos_star, params::Params)

    # TODO: regularization on state _and_ params maybe.
    particles_state = [regularize(Ball.State(pos_star, [0.8, 0]), params) for _ = 1:params.N]
    particles_params = [regularize(Ball.params(), params) for _ = 1:params.N]

    weights = ones(params.N) / params.N

    pos_hat = measure(particles_state, weights)

    return State(pos_hat, particles_state, particles_params, weights)
end

# Step the whole particle filter and return the measurement.
function step(state::State, params::Params, pos_star, Δt)

    # TODO: The structure here is kind of ugly. Things we can do:
    # Make the state mutable.
    # Have each function here return an entire state.

    # Propagate each particle.
    particles_state  = propagate(state, params, Δt)

    # Recalculate the weights.
    weights = calculate_weights(particles_state, pos_star)

    # Make a measurement.
    pos_hat = measure(particles_state, weights)

    # Resample particles.
    particles_state = resample(particles_state, weights, params)

    # # Regularize particles by giving them a random perturbation.
    particles_state = [regularize(particle_state, params) for particle_state in particles_state]
    particles_params = [regularize(particle_params, params) for particle_params in state.particles_params]

    return State(pos_hat, particles_state, state.particles_params, weights)
end

# Propagate each particle.
function propagate(state::State, params, Δt)

    particles_state = state.particles_state
    for (i, (particle_state, particle_params)) in enumerate(zip(state.particles_state, state.particles_params))

        # We want a smaller update size for the particles than this filter.
        particles_state[i] = Ball.step(particle_state, particle_params, Δt, Δt/10)

    end

    return particles_state
end

# Recalculate the weights based on a new measurement.
function calculate_weights(particles_state::Vector{Ball.State}, pos_star)

    # Calculate the distance between each particle and the measurement.
    distances = [norm(particle.pos - pos_star) for particle in particles_state]

    # The higher the distance the lower the weight should be.
    weights = exp.(-distances)
    # weights = maximum(distances) .- distances

    # Normalize the weights so they sum to 1.
    return normalize!(weights, 1)
end

function resample(particles_state::Vector{Ball.State}, weights, params)::Vector{Ball.State}

    # Pick random particles according to their weight. A particle can be picked multiple times.
    sample(particles_state, Weights(weights), params.N, replace=true)
end

# Perturb a particle slightly with some change in pos and vel.
function regularize(state::Ball.State, params::Params)

    Δpos = params.σ_pos .* randn(2)

    # The velocity perturbation is proportional to the velocity.
    Δvel = state.vel .* params.σ_vel .* randn(2)

    return state + Ball.State(Δpos, Δvel)
end

# TODO: regularization function for ball parameters.
function regularize(p_params::Ball.Params, params::Params)
    return p_params
end

# A simple weighted average will provide the measurement.
function measure(particles::Vector{Ball.State}, weights)

    # Not the most beautiful line. The [1] is needed to reduce an array of arrays to a vector.
    # We take the sum and not the mean because the weights add up to 1.
    sum([particle.pos * w for (particle, w) in zip(particles, weights)], dims=1)[1]
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