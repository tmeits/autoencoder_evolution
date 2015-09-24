module Neural

using Optim

typealias Layer = Array{Float64,2}
typealias Layers = Layer[]

typealias LayerDerivative = Array{Float64,3}
typealias LayerDerivatives = LayerDerivative[]

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

function activation(x)
  tanh(x)
end

function activation_derivative(x)
  (1-tanh(x).^2)
end

#Rnxn -> Rnxn
function loss_derivative(y_estimated,y)
    (y_estimated-y)
end

#Rnxn -> Rn
function loss(y_estimated,y)
    0.5*sum((y_estimated-y).^2,1)
end

function apply(w::Layers,x::Input)
  for i=1:length(w)
    x=activation(w[i]*x)
  end
  x
end


function optim_error(w::FlattenedLayers,x::Input,y::Output,topology::Topology)
    w=unflatten_layers(w,topology)
    net_error(w,x,y)
end
function net_error(w::Layers,x::Input,y::Output)
    y_estimated=apply(w,x)
    mean(loss(y_estimated,y))
end


function optim_error_derivative(w::FlattenedLayers,x::Input,y::Output,topology::Topology)
    w=unflatten_layers(w,topology)
    net_error_derivative(w,x,y,topology)
end

function derivative_example_same_layer(net::Input,dw::LayerDerivative,oAnt::Output)
    outputs,current,previous=size(dw)
    dO=activation_derivative(net) # j * j
    for o=1:outputs # derivative != 0 only if output = current
          dw[o,o,:]= dO.*oAnt # d0 is a j*j  matrix, but it is diagonal, so diag(d0)*oAnt=d0.*oAnt
    end
end

#function einsum(a,b,ia,ib)
#    sa=size(a)
#    sb=size(b)
#    splice!(sa,ia)
#    splice!(sb,ib)
#    c=zeros([sa sb])
#end

function net_derivative_example(w::Layers,x::Input,topology:Topology)
  oAnt=x;
  dWj=zero_weights_derivatives(topology[1:2],2)
  Wj=w[1]
  net=Wj*oAnt
  o=activation(net)
  derivative_example_same_layer(net,dWjo[1],oAnt)
  #W indexados con 1 based y topology con 2-based
  for j=2:length(topology)-1
    Wj=w[j] # j x (j-1)
    oAnt=o
    net=Wj*oAnt
    o=activation(net)
    dO=activation_derivative(net) #current
    dWja=dWj # list of derivatives of O[j-1] wrt W[1:j-1]
    dWj=zero_weights_derivatives(topology[1:j],topology[j])

    A = diagm(vec(dO)) * Wj# j x (j-1)
    for jp=1:j-1
        dWjp=dWja[jp] #O[j-1] wrt W[jp] --  j-1 x (jp x jp-1)
        dWj[jp]= einsum(A,dWjp,2,1)# derivative of O[j] wrt W[jp] -- j x (jp x jp-1)
    end
    derivative_example_same_layer(net,dWjo[j],oAnt) # jo=jw
  end
  dW
end

function net_error_derivative(w::Layers,x::Input,y::Output,topology:Topology)
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
        dE[j]=dE[j]+dEij
      end
    end
    dW/n
end

function zero_weights_derivatives(topology::Topology,outputs)
  for i=1:length(topology)-1
    w[i]=zeros(outputs,topology[i+1],topology[i])
  end
  w
end

function zero_weights(topology::Topology)
  for i=1:length(topology)-1
    w[i]=zeros(topology[i+1],topology[i])
  end
  w
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
      Optim.optimize(f,g,flatten_layers(n.layers), method = ::l_bfgs)
end
