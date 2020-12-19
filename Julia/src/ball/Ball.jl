# The ball module
module Ball
export nothing

import Base: +, *
using LinearAlgebra


# Global constants.
g0 = [0, -9.81]  # Gravitational acceleration in x, y coordinates [m/s2].
ρ = 1.225        # Air density [kg/m3].


# State and parameters of the ball, with their default constuctors.
struct State
    pos::Vector{Float64}
    vel::Vector{Float64}
end
state() = State([0, 5], [1, 0])

struct Params
    m::Float64   # Mass [kg]
    r::Float64   # Radius [m]
    cd::Float64  # Drag coefficient [-]
    k::Float64   # Spring constant [N/m]
    c::Float64   # Damper constant [N/(m/s)]
end
params() = Params(1, 1, 0, 1000, 2)


# Add a addition and multiplication function for the ballstate.
+(state1::State, state2::State) = State(state1.pos + state2.pos, state1.vel + state2.vel)
*(m, state::State) = State(m * state.pos, m * state.vel)


# Get the derivative of the state at a certain state and time.
function state_dot(state::State, params::Params)

    # Calculate the force on the ball.
    f = g0 * params.m

    # If the ball is in the air and moving we have a drag force.
    if state.pos[2] > params.r && !iszero(norm(state.vel))
        q = 0.5ρ * norm(state.vel)^2
        A = π * params.r^2
        fd = params.cd * q * A

        # Put it in the direction opposite to the velocity.
        f += -fd * state.vel / norm(state.vel)

    # And when touching the ground we add a spring and damper force.
    # For now this only acts in the vertical direction.
    elseif state.pos[2] < params.r
        f[2] += params.k * (params.r - state.pos[2])
        f[2] += -params.c * state.vel[2]

    end

    # Finally get the acceleration and construct state_dot.
    acc = f / params.m

    return State(state.vel, acc)
end


end  # Module.