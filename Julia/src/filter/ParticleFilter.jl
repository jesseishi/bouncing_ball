# A simple particle filter, the bootstrap filter.
module ParticleFilter
export nothing

using Random
using Statistics

# We're going to cheat a little bit and give the particle filter access to the
#  'exact' dynamics of the ball. TODO: Make a simplified ball model that the
#  particle filter must use.
include("../ball/Ball.jl")
using .Ball


struct State
    particles::Vector{Ball.State}
    weights::Vector{Float64}
end

struct Params
    N::Int64                # Number of particles [-].
    σ_pos::Vector{Float64}  # Standard deviation for position perturbation [m].
    σ_vel::Vector{Float64}  # Standard deviation for velocity perturbation [m].
end
params() = Params(3, [0.1, 0.1], [0, 0])

function init(pos_star, params::Params)
    particles = [regularize(Ball.State(pos_star, [0, 0]), params) for _ = 1:params.N]
    weights = ones(params.N) / params.N

    return State(particles, weights)
end

# Perturb a particle slightly with some change in pos and vel.
function regularize(state::Ball.State, params::Params)
    Δpos = params.σ_pos .* randn(2)
    Δvel = params.σ_vel .* randn(2)

    return state + Ball.State(Δpos, Δvel)
end

# A simple weighted average will provide the measurement.
function measure(state::State)

    # Not the most beautiful line. The [1] is needed to reduce an array of arrays to a vector.
    # We take the sum and not the mean because the weights add up to 1.
    sum([particle.pos * w for (particle, w) in zip(state.particles, state.weights)], dims=1)[1]
end

# A function to help with logging the data.
function get_particles_data(state::State, params::Params)

    # We make an array with rows for x-positions, y-positions, and weights of each particle.
    particles_data = Array{Float64, 2}(undef, 3, params.N)

    particles_data[1, :] = [particle.pos[1] for particle in state.particles]
    particles_data[2, :] = [particle.pos[2] for particle in state.particles]
    particles_data[3, :] = state.weights

    return particles_data
end


end  # Module ParticleFilter