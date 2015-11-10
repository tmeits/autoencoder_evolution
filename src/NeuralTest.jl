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
y = [0.5 0.3 0.4 0.1 0.7]'
o = activation(W[2] * activation(W[1]*x))
xs = rand(3,2)
ys = rand(5,2)
os = activation(W[2] * activation(W[1]*xs))


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
  o2 = Neural.apply(n.layers, x)
  @fact o2 --> o
end

facts("Test derivative") do
  (dW, o2) = Neural.net_derivative_example(W, vec(x), topology)
  @fact o2 --> vec(o)
  @fact length(dW) --> 2
  @fact size(dW[1]) --> (5,4,3)
  @fact size(dW[2]) --> (5,5,4)
end

facts("Test gradient checking") do
  EPS = 1.0^-5
  (dW, o) = Neural.net_derivative_example(W, vec(x), topology)
  dWp = Neural.zero_weights_derivatives(topology, 3)
  #println(string("dW", dW))
  Wl = deepcopy(W)
  for l=1:length(W)
    #Wl = copy(W[l])
    (nUnits, nWeights) = size(Wl[l])
    for u = 1:nUnits
      for weight = 1:nWeights
        v = Wl[l][u, weight]
        Wl[l][u, weight] += EPS
        Oplus = Neural.apply(Wl, x)
        Wl[l][u, weight] -= 2*EPS
        Ominus = Neural.apply(Wl, x)
        dWp[l][:,u,weight] = (Oplus - Ominus) / (2 * EPS)
        Wl[l][u, weight] += EPS
        @fact Wl[l][u, weight] --> v "Wrong restored value"
      end
    end
  end
  #println(string("dWp", dWp))
  for l = 1:length(W)
    @fact dWp[l] --> roughly(dW[l]; atol=EPS) "wrong derivative at layer $l"
  end
end

facts("Test network error derivative") do
  EPS = 1.0^-5
  dE = Neural.net_error_derivative(W, x, y, topology)
  dWp = Neural.zero_weights(topology)
  Wl = deepcopy(W)
  for l=1:length(W)
    (nUnits, nWeights) = size(Wl[l])
    for u = 1:nUnits
      for weight = 1:nWeights
        v = Wl[l][u, weight]
        Wl[l][u, weight] += EPS
        Oplus = Neural.net_error(Wl, x, y)
        Wl[l][u, weight] -= 2*EPS
        Ominus = Neural.net_error(Wl, x, y)
        dWp[l][u,weight] = (Oplus - Ominus) / (2 * EPS)
        Wl[l][u, weight] += EPS
        @fact Wl[l][u, weight] --> v "Wrong restored value"
      end
    end
  end
  #println(string("dWp", dWp))
  for l = 1:length(W)
    @fact dWp[l] --> roughly(dE[l]; atol=EPS) "wrong error derivative at layer $l"
  end
end

facts("Test network error derivative for dataset") do
  EPS = 1.0^-5
  dE = Neural.net_error_derivative(W, xs, ys, topology)
  dWp = Neural.zero_weights(topology)
  Wl = deepcopy(W)
  for l=1:length(W)
    (nUnits, nWeights) = size(Wl[l])
    for u = 1:nUnits
      for weight = 1:nWeights
        v = Wl[l][u, weight]
        Wl[l][u, weight] += EPS
        Oplus = Neural.net_error(Wl, xs, ys)
        Wl[l][u, weight] -= 2*EPS
        Ominus = Neural.net_error(Wl, xs, ys)
        dWp[l][u,weight] = (Oplus - Ominus) / (2 * EPS)
        Wl[l][u, weight] += EPS
        @fact Wl[l][u, weight] --> v "Wrong restored value"
      end
    end
  end
  #println(string("dWp", dWp))
  for l = 1:length(W)
    @fact dWp[l] --> roughly(dE[l]; atol=EPS) "wrong error derivative at layer $l"
  end
end

facts("Test net error") do
  e = Neural.net_error(W, xs, os)
  e2 = Neural.net_error(W, xs + rand(size(xs)), os)
  e3 = Neural.net_error(W, xs, os + rand(size(os)))
  @fact e --> 0
  @fact e2 --> greater_than(0)
  @fact e3 --> greater_than(0)
end

facts("Test training") do
  #xs = [1 2 ; ]
  #NeuralTest.topology = [3,1,1]
  n = Neural.Net(topology, copy(W))
  Neural.train!(n, xs, ys)
  os2 = Neural.apply(n.layers, xs)
  println(n.layers)
  @fact os2 --> roughly(ys)
end

#=
facts("Test net training") do
  n = Neural.Net(topology, copy(W))
  Neural.train!(n, xs, ys)
  os2 = Neural.apply(n.layers, xs)
  println(n.layers)
  @fact os2 --> roughly(ys)
end
=#

#=
using Datasets
x,y,l=Datasets.iris()
ys=zeros(3,length(y))
  for i=1:unique(y)
   ys[i,y.==i]=1
  end
=#
