push!(LOAD_PATH,dirname(@__FILE__))
reload("Neural.jl")

using FactCheck
using Neural

topology = [3,4,5] #"output" is the last one
Wz = Neural.zero_weights(topology)
W = Neural.random_weights(topology)
dW2 = Neural.zero_weights_derivatives(topology, 2)
dW3 = Neural.zero_weights_derivatives(topology, 3)
n = Neural.Net(topology, W)
activation = Neural.activation
x = [2.0 5.0 1.0]'
y = activation(W[2] * activation(W[1]*x))

facts("Test layer creation") do
  @fact length(Wz) --> 2
  @fact size(Wz[1]) --> (4,3)
  @fact size(Wz[2]) --> (5,4)
end

facts("Test layer derivatives creation") do
  @fact length(dW2) --> 1
  @fact length(dW3) --> 2
  @fact size(dW2[1]) --> (4,4,3)
  @fact size(dW3[1]) --> (5,4,3)
  @fact size(dW3[2]) --> (5,5,4)

end

facts("Test flatten/unflatten") do
  flattened = Neural.flatten_layers(W)
  @fact Neural.unflatten_layers(flattened, topology) --> W
  @fact length(size(flattened)) --> 1
end


facts("Test flatten/unflatten") do
  flattened = Neural.flatten_layers(W)
  @fact Neural.unflatten_layers(flattened, topology) --> W
  @fact length(size(flattened)) --> 1
end

facts("Test apply") do
  o = Neural.apply(n.layers, x)
  @fact o --> y
end

facts("Test derivative") do
  (dW, o) = Neural.net_derivative_example(W, x, topology)
  @fact o --> y
  @fact length(dW) --> 2
  @fact size(dW[1]) --> (5,4,3)
  @fact size(dW[2]) --> (5,5,4)
end
