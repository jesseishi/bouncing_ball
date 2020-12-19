# A 2D position sensor.
module Sensor
export nothing

using Random


# State of the sensor could be used to keep track of a moving bias.
struct State
end

# Sensor parameters.
struct Params
    μ::Vector{Float64}
    σ::Vector{Float64}
end
params() = Params([0, 0], [0, 0])


# Measure the position.
function measure(pos::Vector, params::Params)
    noise = params.μ + params.σ .* randn(size(pos))
    return pos + noise
end


end  # Module PosSensor