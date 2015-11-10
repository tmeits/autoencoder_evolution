module Neural
export network_topology
using Optim

typealias Layer Array{Float64,2}
typealias Layers Vector{Layer}

typealias LayerDerivative Array{Float64,2}
typealias LayerDerivatives Vector{LayerDerivative}

typealias FlattenedLayers Vector{Float64}
typealias Input Array{Float64,1}
typealias Inputs Array{Float64,2}
typealias Output Array{Float64,1}
typealias Outputs Array{Float64,2}

typealias LayerOutputs Vector{Output}


typealias Topology Vector{Int}

type Net
    topology::Topology
    layers::Layers
end

function network_topology(x::Inputs,hidden::Int,y::Outputs)
  inputs,examples=size(x)
  outputs,examples=size(y)
  [inputs,hidden,outputs]
end

function unflatten_layers(w::FlattenedLayers,topology::Topology)
  result=Layers(length(topology)-1)
  base=1
  for i=1:length(topology)-1
      w_size=topology[i]*topology[i+1]
      wi=w[base:base+w_size-1]
      wi = reshape(wi, topology[i+1], topology[i])
      base+=w_size
      result[i] = wi
  end
  result
end


function flatten_layers(unflattened_w::Layers)
  w=zeros(0)
  for i=1:length(unflattened_w)
    f=reshape(unflattened_w[i],length(unflattened_w[i]))
    w=[w ; f]
  end
  w
end

#R -> R, for each element. Vectorized implementation.
function activation(x)
  tanh(x)
end

#R -> R, for each element. Vectorized implementation.
function activation_derivative(x)
  (1-tanh(x).^2)
end

#Rn -> Rn, for each element. Vectorized implementation.
function loss_derivative(y_estimated,y)
    (y_estimated-y)
end

#Rn -> R, for each element. Vectorized implementation.
function loss(y_estimated,y)
    0.5*sum((y_estimated-y).^2,1)
end

function apply(w::Layers,x::Inputs)
  for i=1:length(w)
    x=activation(w[i]*x)
  end
  x
end


function optim_error(w::FlattenedLayers, x::Inputs, y::Outputs, topology::Topology)
    w=unflatten_layers(w,topology)
    e=net_error(w,x,y)
    println(string("error=",e))
    return e
end
function net_error(w::Layers,x::Inputs,y::Outputs)
    y_estimated=apply(w,x)
    mean(loss(y_estimated,y))
end


function optim_error_derivative(w::FlattenedLayers, x::Inputs, y::Outputs, topology::Topology)
    w=unflatten_layers(w,topology)
    net_error_derivative(w,x,y,topology)
end


function backward(dW::LayerDerivatives,w::Layers, o::LayerOutputs,net::LayerOutputs, topology::Topology,ld::Output)
  #W indexados con 1 based y topology con 2-based
  output_layer_index=length(o)

  dW[end]=ld.*activation_derivative(o[end])
  delta=dW[end] # n_L
  for layer=output_layer-1:-1:2
      delta= (w[layer+1]'*delta).*activation_derivative(net[layer]) #  ((n_l+1 x n_l)' * n_l+1) .* n_l -> n_l
      # delta is a column vector
      dW[j]+=delta*o[layer-1]' #  n_l + n_l-1
  end

end


function forward(w::Layers, x::Input, topology::Topology)#::LayerOutputs
  o=LayerOutputs()
  net=LayerOutputs()
  append!(o,x)
  for i=1:length(w)
    net_i=w[i]*x
    x=activation(net_i)
    append!(net,net_i)
    append!(o,x)
  end
  o,net
end

function net_error_derivative(w::Layers, x::Inputs, y::Outputs, topology::Topology)
    dE=zero_weights(topology)
    n=size(x,2)
    for i=1:n
      xi=x[:,i]
      yi=y[:,i]
      o,net=forward(w,xi,topology)
      ld=loss_derivative(o[end],yi)
      backward!(dE,w,o,net,topology,ld)
    end
    for j=1:length(dE)
      dE[j] /= n
    end
    return dE
end

function zero_weights(topology::Topology)
  w = Layers(length(topology)-1)
  for i=1:length(topology)-1
    w[i]=zeros(topology[i+1], topology[i])
  end
  w
end

function random_weights(topology::Topology)
  w = Layers(length(topology)-1)
  for i=1:length(topology)-1
    w[i]=rand(topology[i+1],topology[i])
  end
  w
end

function train(topology::Topology, x::Inputs, y::Outputs)
      n = Net(random_weights(n.topology), topology)
      train!(n, x, y)
      return n
end

function train!(n::Net,x::Inputs,y::Outputs)
      f(w) = optim_error(w,x,y,n.topology)
      g = function(w,storage); storage = flatten_layers(optim_error_derivative(w,x,y,n.topology)); end
      #n.layers = random_weights(n.topology)
      flattened_layers = flatten_layers(n.layers)
      Optim.optimize(f, flattened_layers, method = :nelder_mead)
      #Optim.optimize(f, g, flattened_layers, method = :gradient_descent)
      n.layers = unflatten_layers(flattened_layers, n.topology)
end

end
