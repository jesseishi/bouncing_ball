include("Ball.jl")
using .Ball

state = Ball.state()
params = Ball.params()

Ball.ode(state, params)
Ball.step(state, params, 0.1)
