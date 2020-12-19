# The easiest method to go from t to t + Δt.
module Euler
export nothing


function step(state, state_dot, Δt)
    return state + Δt * state_dot
end


end  # Module.
