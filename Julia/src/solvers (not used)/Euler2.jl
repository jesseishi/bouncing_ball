# A 2nd order implementation of the Euler method.
# This only works if the state vector is [pos_x pos_y vel_x vel_y].
module Euler2
export nothing


# Unfortunately it's hard to make this work in general as this step function now
# has to know BallState in order to work as the state update isn't just a linear update.
# Do the double dot to indicate that Ball is already included in the parent module
# or something (This is the same stuff Tucker was complaining about.).
using ..Ball


function step(state, state_dot, Δt)
    Δpos = state_dot.pos * Δt + 0.5 * Δt^2 * state_dot.vel
    Δvel = state_dot.vel * Δt
    return Ball.State(state.pos + Δpos, state.vel + Δvel)
end


end  # Module.