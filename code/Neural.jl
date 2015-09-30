module Neural
export network_topology
using Optim

typealias Layer Array{Float64,2}
typealias Layers Vector{Layer}

typealias LayerDerivative Array{Float64,3}
typealias LayerDerivatives Vector{LayerDerivative}

typealias FlattenedLayers Vector{Float64}
typealias Input Array{Float64,2}
typealias Output Array{Float64,2}

typealias Topology Vector{Int}

type Net
    topology::Topology
    layers::Layers
end

function network_topology(x::Input,hidden::Int,y::Output)
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

function apply(w::Layers,x::Input)
  for i=1:length(w)
    x=activation(w[i]*x)
  end
  x
end


function optim_error(w::FlattenedLayers, x::Input, y::Output, topology::Topology)
    w=unflatten_layers(w,topology)
    net_error(w,x,y)
end
function net_error(w::Layers,x::Input,y::Output)
    y_estimated=apply(w,x)
    mean(loss(y_estimated,y))
end


function optim_error_derivative(w::FlattenedLayers, x::Input, y::Output, topology::Topology)
    w=unflatten_layers(w,topology)
    net_error_derivative(w,x,y,topology)
end

function derivative_example_same_layer(net::Input, dw::LayerDerivative, oAnt::Output)
    outputs,current,previous=size(dw)
    dO=activation_derivative(net) # j * j
    for o=1:outputs # derivative != 0 only if output = current
        println(string("dO size", size(dO)))
        println(string("oAnt size", size(oAnt)))
        dw[o,o,:]= oAnt * dO[o] # d0 is a j*j  matrix, but it is diagonal, so diag(d0)*oAnt=d0.*oAnt
    end
end

function einsum23_21(matrix, tensor)
  #2nd index of matrix and 1st index of tensor
  (m1, m2) = size(matrix)
  (t1, t2, t3) = size(tensor)
  result = zeros(m1, t2, t3)
  if m2 != t1
    error("Indexes m1 and t2 don't match.")
  end
  for i=1:m1
    for j=1:t2
      for k=1:t3
        result[i,j,k] = matrix[m1,:] * tensor[:,t2,t3]
      end
    end
  end
  return result
end

function net_derivative_example(w::Layers, x::Input, topology::Topology)
  oAnt=x;
  dWj=zero_weights_derivatives(topology,2) #list of tensors
  Wj=w[1]
  net=Wj*oAnt
  o=activation(net)
  derivative_example_same_layer(net, dWj[1], oAnt)
  #W indexados con 1 based y topology con 2-based
  for j=2:length(topology)-1
    Wj=w[j] # j x (j-1)
    oAnt=o
    net=Wj*oAnt
    o=activation(net)
    dO=activation_derivative(net) #current
    dWja=dWj # list of derivatives of O[j-1] wrt W[1:j-1]
    dWj=zero_weights_derivatives(topology, j)

    A = diagm(vec(dO)) * Wj# j x (j-1)
    for jp=1:j-1
        dWjp=dWja[jp] #O[j-1] wrt W[jp] --  j-1 x (jp x jp-1)
        dWj[jp]= einsum23_21(A,dWjp)# derivative of O[j] wrt W[jp] -- j x (jp x jp-1)
    end
    derivative_example_same_layer(net,dWj[j],oAnt) # jo=jw
  end
  return (dWj, o)
end

function net_error_derivative(w::Layers, x::Input, y::Output, topology::Topology)
    dE=zero_weights(topology)
    n=size(x,2)
    for i=1:n
      xi=x(:,i)
      yi=y(:,i)
      dWi,oi=net_derivative_example(w,xi,topology)
      for j=1:length(dWi)
        ld=loss_derivative(oi,yi)
        dEij=zeros(size(dE[j]))
        for k=1:length(ld)
          dEij+=dWi[j][k,:,:]*ld[k]
        end
        dE[j]=dE[j]+dEij
      end
    end
    for j=1:length(dE)
      dE[j] /= n
    end
    return dE
end

function zero_weights_derivatives(topology::Topology, output_layer::Int)
  w = LayerDerivatives(output_layer - 1)
  for i = 1:output_layer-1
    w[i]=zeros(topology[output_layer], topology[i+1], topology[i])
  end
  w
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

function train!(n::Net,x::Input,y::Output)
      f(w) = optim_error(w,x,y,n.topology)
      g(w) = optim_error_derivative(w,x,y,n.topology)
      n.layers=random_weights(n.topology)
      Optim.optimize(f,g,flatten_layers(n.layers), method = :l_bfgs)
end

end
