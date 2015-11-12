module Neural
export network_topology
using Optim

macro print_size(zzz)
  :(println( $(string(zzz)) * " has size -> " * string(size($zzz)) ))
end

macro print_var(zzz)
  :(println($(string(zzz)) * " = " * string($zzz) ))
end


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
  w=Float64[]
  for i=1:length(unflattened_w)
    f=reshape(unflattened_w[i],length(unflattened_w[i]))
    w=[w ; f]
  end
  w
end

#R -> R, for each element. Vectorized implementation.
function activation(x)
  1.0./(1.0+exp(-x)) #tanh(x)
end

#R -> R, for each element. Vectorized implementation.
function activation_derivative(x)
   a=activation(x)
   a.*(1-a) #(1-tanh(x).^2)
end


function loss_derivative(y_estimated::Output,y::Output)
    loss_derivative(y_estimated'',y'')
end


#Rn -> Rn, for each element. Vectorized implementation.
function loss_derivative(y_estimated::Outputs,y::Outputs)
    (y_estimated-y)
end

#Rn -> R, for each element. Vectorized implementation.
function loss(y_estimated::Output,y::Output)
    loss(y_estimated'',y'')
end

function loss(y_estimated::Outputs,y::Outputs)
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
    error=net_error(w,x,y)
    @print_var error
    return error
end
function net_error(w::Layers,x::Inputs,y::Outputs)
    y_estimated=apply(w,x)
    mean(loss(y_estimated,y))
end


function optim_error_derivative(w::FlattenedLayers, x::Inputs, y::Outputs, topology::Topology)
    w=unflatten_layers(w,topology)
    net_error_derivative(w,x,y,topology)
end


function backward!(dE::LayerDerivatives,w::Layers, o::LayerOutputs,net::LayerOutputs,ld::Outputs)
  #W indexados con 1 based y topology con 2-based
  output_layer_index=length(o)
  delta=ld.*activation_derivative(net[end]) #  n_L .*  n_L -> n_L
  a=delta*(o[end-1])'
  dE[end]+= a #n_L x n_L-1
  for layer=length(w)-1:-1:1
      delta= (w[layer+1]'*delta) .* activation_derivative(net[layer]) #  ((n_l+1 x n_l)' * n_l+1) .* n_l -> n_l
      dE[layer]+=delta*o[layer]' #  n_l x 1 * 1 x n_l-1 -> n_l x n_l-1
  end

end

function forward(w::Layers, x::Input)#::LayerOutputs
  o=Output[]
  net=Output[]
  push!(o,x)
  for i=1:length(w)
    net_i=w[i]*x
    x=activation(net_i)
    push!(net,net_i)
    push!(o,x)
  end
  o,net
end

function net_error_derivative(w::Layers, x::Inputs, y::Outputs)
    dE=zero_weights(w)
    n=size(x,2)
    for i=1:n
      #@print_var(i)
      xi=x[:,i]
      yi=y[:,i]
      o,net=forward(w,xi)
      ld=loss_derivative(o[end],yi)
      backward!(dE,w,o,net,ld)
    end
    for j=1:length(dE)
      dE[j] /= n
    end
    return dE
end


function zero_weights(w::Layers)
  wz = Layers(length(w))
  for i=1:length(wz)
    wz[i]=zeros(w[i])
  end
  wz
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
      n = Net(topology,random_weights(topology))
      train!(n, x, y)
      return n
end

function train!(n::Net,x::Inputs,y::Outputs)
      f(w) = optim_error(w,x,y,n.topology)
      g = function(w,storage); storage = flatten_layers(optim_error_derivative(w,x,y,n.topology)); end
      #n.layers = random_weights(n.topology)
      flattened_layers = flatten_layers(n.layers)
      Optim.optimize(f, flattened_layers, method = :gradient_descent)
      #Optim.optimize(f,g, flattened_layers, method = :cg)
      n.layers = unflatten_layers(flattened_layers, n.topology)
end


function trainbp(topology::Topology, x::Inputs, y::Outputs,iterations::Int,learning_rate::Float64)
      n = Net(topology,random_weights(topology))
      trainbp!(n,x,y,iterations,learning_rate)
      n
end

function trainbp!(n::Net, x::Inputs, y::Outputs,iterations::Int,learning_rate::Float64)
    for i=1:iterations
          dE=net_error_derivative(n.layers,x,y)
          for i=1:length(dE)
              n.layers[i]-=learning_rate*dE[i]
              #println(dE[i])
          end
          @print_var net_error(n.layers,x,y)
    end
end


function trainbp_minibatch(topology::Topology, x::Inputs, y::Outputs,iterations::Int,learning_rate::Float64,batch_size::Int)
      n = Net(topology,random_weights(topology))
      trainbp_minibatch!(n,x,y,iterations,learning_rate,batch_size)
      n
end

function trainbp_minibatch!(net::Net, x::Inputs, y::Outputs,iterations::Int,learning_rate::Float64,batch_size::Int)
    n=size(x,2)
    for i=1:iterations
          indices=randperm(n)[1:batch_size]
          dE=net_error_derivative(net.layers,x[:,indices],y[:,indices])
          for i=1:length(dE)
              net.layers[i]-=learning_rate*dE[i]
              #println(dE[i])
          end
          @print_var net_error(net.layers,x,y)
    end
end

end
