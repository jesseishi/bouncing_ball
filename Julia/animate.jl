using Plots

@userplot BallPlot
@recipe function f(cp::BallPlot)
    xs, ys, i = cp.args
    xlims --> [0, maximum(xs)]
    ylims --> [0, maximum(ys)]

    label --> "Ball"
    xguide --> "x-position [m]"
    yguide --> "y-position [m]"

    # Plot all the points from 1 to i.
    xs[1:i], ys[1:i]
end

@userplot SensorPlot
@recipe function f(cp::SensorPlot)
    xs, ys, i = cp.args

    seriestype --> :scatter
    label --> "Measurement"

    xs[1:i], ys[1:i]
end

ball_data = DataFrame(CSV.File("Julia/results/data/ball_data"))
sensor_data = DataFrame(CSV.File("Julia/results/data/sensor_data"))

# The sensor has less measurements so we need to figure out how much less.
n = length(ball_data.x)
n_sensor = length(sensor_data.x)
rel_sensor_freq = (sensor_data.t[2] - sensor_data.t[1]) / (ball_data.t[2] - ball_data.t[1])

anim = @animate for i = 1:n
    ballplot(ball_data.x, ball_data.y, i)

    # Since there are less measurements i_sensor is less that i, this ensures they still line up.
    i_sensor = ceil(Int, i/rel_sensor_freq)
    sensorplot!(sensor_data.x, sensor_data.y, i_sensor)
end

# This real time stuff doesn't really work...
fps_real_time = n / (ball_data.t[end] - ball_data.t[1])
gif(anim, "Julia/results/plots/joe.gif", fps=fps_real_time)
