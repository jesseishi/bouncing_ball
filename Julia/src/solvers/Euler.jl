module Euler
export nothing


function step(state, state_dot, dt)
    return state + dt * state_dot
end


end  # Module.
