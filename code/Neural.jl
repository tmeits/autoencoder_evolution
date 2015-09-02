module Neural

using Optim

typealias Layer = Array{Float64,2}
typealias Layers = Layer[]
typealias FlattenedLayers = Vector{Float64}
typealias Input =  Array{Float64,2}
typealias Output =  Array{Float64,2}

typealias Topology = Int[]

type Net
    topology::Topology
    layers:Layers
end

function network_topology(x::Input,hidden::Int,y::Output)
  inputs,examples=size(x)
  outputs,examples=size(y)
  [inputs,hidden,outputs]
end

function unflatten_layers(w::FlattenedLayers,topology::Topology)
  w=Array{Float64,2}[]
  base=1
  for i=1:length(topology)-1
      w_size=topology[i]*topology[i+1]
      wi=w[base:base+w_size-1]
      base+=w_size
      push!(w,wi)
  end
  w
end


function flatten_layers(unflattened_w::Layers)
  w=zeros(0)
  for i=1:length(unflattened_w)
    f=reshape(unflattened_w[i],length(unflattened_w[i]))
    w=[w ; f]
  end
  w
end

function optim_error(w::FlattenedLayers,x::Input,y::Output,topology::Topology)
    w=unflatten_layers(w,topology)
    net_error(w,x,y)
end


function optim_error_derivative(w::FlattenedLayers,x::Input,y::Output,topology::Topology)
    w=unflatten_layers(w,topology)
    net_error_derivative(w,x,y)
end

function apply(w::Layers,x::Input)
  for i=1:length(w)
    x=tanh(w[i,:,:]*x)
  end
  x
end

function net_error(w::Layers,x::Input,y::Output)
    y_estimated=apply(w,x)
    sum((y-y_estimated).^2)
end

function net_error_derivative(w::Layers,x::Input,y::Output)
    0
end

function random_weights(topology::Topology)
  for i=1:length(topology)-1
    w[i]=rand(topology[i+1],topology[i])
  end
  w
end

function train!(n::Net,x::Input,y::Output)
      f(w) = optim_error(w,x,y,n.topology)
      g(w) = optim_error_derivative(w,x,y,n.topology)
      n.layers=random_weights(n.topology)
      Optim.optimize(f, flatten_layers(n.layers), method = ::l_bfgs)
end
