# Usings.
using Plots
using DataFrames
using CSV


# Load the data.
ball_data = DataFrame(CSV.File("Julia/results/ball_data"))


# Plot it.
plot(ball_data.x, ball_data.y)
