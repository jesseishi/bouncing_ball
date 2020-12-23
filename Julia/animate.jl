using Plots
using HDF5

@userplot BallPlot
@recipe function f(cp::BallPlot)
    pos, i = cp.args
    xlims --> [0, maximum(pos[1, :])]
    ylims --> [0, maximum(pos[2, :])]

    label --> "Ball"
    xguide --> "x-position [m]"
    yguide --> "y-position [m]"

    # Plot all the points from 1 to i.
    pos[1, 1:i], pos[2, 1:i]
end

@userplot SensorPlot
@recipe function f(cp::SensorPlot)
    pos_star, i = cp.args

    seriestype --> :scatter
    label --> "Measurement"

    pos_star[1, 1:i], pos_star[2, 1:i]
end

@userplot EstimationPlot
@recipe function f(cp::EstimationPlot)
    pos_hat, i = cp.args

    seriestype --> :scatter
    label --> "Estimation"

    pos_hat[1, 1:i], pos_hat[2, 1:i]
end

@userplot ParticlesPlot
@recipe function f(cp::ParticlesPlot)
    particles, i = cp.args

    seriestype --> :scatter
    seriesalpha --> 10 .* particles[3, :, 1:i]
    label --> ""
    seriescolor --> "black"

    particles[1, :, 1:i], particles[2, :, 1:i]
end

h5open("Julia/results/data/results.h5", "r") do fid
    global t_continuous = read(fid["continuous/t"])
    global pos = read(fid["continuous/ball_pos"])

    global t_discrete = read(fid["discrete/t"])
    global pos_star = read(fid["discrete/pos_star"])
    global pos_hat = read(fid["discrete/pos_hat"])
    global particles = read(fid["discrete/particles"])
end

# The sensor has less measurements so we need to figure out how much less.
n_continuous = length(t_continuous)
n_discrete = length(t_discrete)
rel_sensor_freq = (n_discrete-1) / (n_continuous-1)

anim = @animate for i = 1:n_continuous

    # Since there are less measurements i_discrete is less that i, this ensures they still line up.
    i_discrete = floor(Int, (i-1)*rel_sensor_freq+1 + eps(maximum(Float64, n_continuous)))

    # Particles, plot these first because we want the rest to be on top of it.
    particlesplot(particles, i_discrete)
    ballplot!(pos, i)
    sensorplot!(pos_star, i_discrete)
    estimationplot!(pos_hat, i_discrete)
end

# This real time stuff doesn't really work...
fps_real_time = n_continuous / (t_continuous[end] - t_continuous[1])
gif(anim, "Julia/results/plots/joe.gif", fps=fps_real_time)
