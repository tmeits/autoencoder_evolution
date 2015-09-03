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

function derivative_example_same_layer(w::Layer,dw::LayerDerivative,oAnt::Output)
    current,previous,outputs=size(dw)
    #size(oAnt)=previous
    net=w*oAnt; # current x previous * previous -> current
    O=activation(net) #current
    dO=activation_derivative(net) #current
    for i=1:current
      dw[i,:,i]=dO[i]*oAnt;
    end
end

function net_derivative_example(w::Layers,x::Input,topology:Topology)
  oAnt=x;
  dWjo=zero_weights_derivatives(topology[1:2],2)
  Wjo=w[1]
  net=Wjo*oAnt
  o=activation(net)
  derivative_example_same_layer(w[1],dWjo[1],oAnt)

  for jo=3:length(topology)
    Wjo=w[jo]

    oAnt=o
    o=Wjo*oAnt
    dO=activation_derivative(net) #current

    dWant=dWjo
    dWjo=zero_weights_derivatives(topology[1:jo],topology[jo])

    for jw=1:jo-1
        for i=1:
          for i2=1:
            
          end
        end
    end
    derivative_example_same_layer(w[jo],dWjo[jo],oAnt) # jo=jw

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
          dEij+=dWi[j][:,:,k]*ld[k]
        dE[j]=dE[j]+dEij
      end
    end
    dW/n
end

function zero_weights_derivatives(topology::Topology,outputs)
  for i=1:length(topology)-1
    w[i]=zeros(topology[i+1],topology[i],outputs)
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
