# The ball module
module Ball
export nothing

import Base: +, *
using LinearAlgebra


# Global constants (TODO: Move to WorldParams?).
g0 = [0, -9.81]  # Gravitational acceleration in x, y coordinates [m/s2].
ρ = 1.225        # Air density [kg/m3].


# State and parameters of the ball, with their default constuctors.
struct State
    pos::Vector{Float64}
    vel::Vector{Float64}
end
state() = State([0, 10], [0.8, 0])

struct Params
    m::Float64   # Mass [kg]
    r::Float64   # Radius [m]
    cd::Float64  # Drag coefficient [-]
    k::Float64   # Spring constant [N/m]
    c::Float64   # Damper constant [N/(m/s)]
end
params() = Params(1, 1, 0.01, 1000, 2)


# Add a addition and multiplication function for the ballstate.
+(state1::State, state2::State) = State(state1.pos + state2.pos, state1.vel + state2.vel)
*(m, state::State) = State(m * state.pos, m * state.vel)


# Get the derivative of the state at a certain state and time.
function ode(state::State, params::Params)

    # Calculate the force on the ball.
    f = g0 * params.m

    # If the ball is in the air and moving we have a drag force.
    if state.pos[2] > params.r / 2 && !iszero(norm(state.vel))
        q = 0.5ρ * norm(state.vel)^2
        A = π * params.r^2
        fd = params.cd * q * A

        # Put it in the direction opposite to the velocity.
        f += -fd * state.vel / norm(state.vel)

    # And when touching the ground we add a spring and damper force.
    # For now this only acts in the vertical direction.
    elseif state.pos[2] < params.r / 2
        f[2] += params.k * (params.r / 2 - state.pos[2])
        f[2] += -params.c * state.vel[2]

    end

    # Finally get the acceleration and construct ode.
    acc = f / params.m

    return State(state.vel, acc)
end


# RK4 implementation to make a timestep.
function step(state::State, params::Params, Δt)
    k1 = ode(state,              params)
    k2 = ode(state + 0.5Δt * k1, params)
    k3 = ode(state + 0.5Δt * k2, params)
    k4 = ode(state +    Δt * k3, params)

    state_dot = 1/6 * (k1 + 2k2 + 2k3 + k4)

    return state + Δt * state_dot
end

# Step but with a certain max dt.
function step(state::State, params::Params, Δt, max_dt)
    N_steps = ceil(Int, Δt / max_dt)

    dt = Δt / N_steps

    new_state = state
    for _ in 1:N_steps
        new_state = step(new_state, params, dt)
    end

    return new_state
end

# TODO: Would be pretty cool to add a variable step solver.


end  # Module Ball.