# A 2D position sensor.
module Sensor
export nothing

using Random


# Sensor parameters.
struct Params
    bias::Vector{Float64}
    σ::Vector{Float64}
end
params() = Params([0, 0], [0.4, 0.4])


# Measure the position.
function measure(pos::Vector, params::Params)
    noise = params.σ .* randn(size(pos))  # TODO: make a AR(1) walk (using sensor state).
    return pos + params.bias + noise
end


end  # Module Sensor