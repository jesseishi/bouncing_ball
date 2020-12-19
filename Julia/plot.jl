# Usings.
using Plots
using DataFrames
using CSV


# Load the data.
ball_data = DataFrame(CSV.File("Julia/results/ball_data"))
sensor_data = DataFrame(CSV.File("Julia/results/sensor_data"))


# Plot it.
plot(ball_data.x, ball_data.y)
scatter!(sensor_data.x, sensor_data.y)
