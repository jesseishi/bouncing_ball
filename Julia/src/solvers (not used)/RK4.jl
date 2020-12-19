# The famous RK4.
module RK4
export nothing


# We also have to use the Ball module here to get the BallParams needed for the
# state_dot function.
using ..Ball


function step(state, f, Δt)
    k1 = f(state);
    k2 = f(state + 0.5Δt * k1);
    k3 = f(state + 0.5Δt * k2);
    k4 = f(state + Δt * k3);

    state_dot = (k1 + 2 * k2 + 2 * k3 + k4) / 6;

    return state + Δt * state_dot
end


end  # Module.
