# Usings.
using Plots
using HDF5


h5open("Julia/results/data/results.h5", "r") do fid
    global t_continuous = read(fid["continuous/t"])
    global pos = read(fid["continuous/ball_pos"])

    global discrete_t = read(fid["discrete/t"])
    global pos_star = read(fid["discrete/pos_star"])
    global pos_hat = read(fid["discrete/pos_hat"])
    global particles = read(fid["discrete/particles"])
end


# Plot it.
scatter(particles[1, :, :], particles[2, :, :], alpha=10 .* particles[3, :, :], color="black", label="")
plot!(pos[1, :], pos[2, :], label="true", xlims=[0, maximum(pos[1, :])], ylims=[0, maximum(pos[2, :])], color="green")
scatter!(pos_hat[1, :], pos_hat[2, :], label="estimate", color="orange")
scatter!(pos_star[1, :], pos_star[2, :], label="measurement", color="blue")
