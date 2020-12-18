include("Ball.jl")
using .Ball

state = Ball.state()
params = Ball.params()

state_dot = Ball.state_dot(state, params)

state + 0.1state_dot
