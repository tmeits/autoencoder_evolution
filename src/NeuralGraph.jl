

type Net
  layers::Vector{Layer}
end

type NetDerivative
  layers::Vector{LayerDerivative}
end

abstract LayerDerivative
abstract Layer

type SigmoidLayer <: Layer

end

type ReluLayer <: Layer

end

function forward(l::Layer,x::Inputs)#::(Inputs,Inputs) #net input and output

end

# delta dC/dz
function backward!(d::LayerDerivative,l::Layer,x::Inputs,delta::Inputs)#::Inputs #delta of this layer
  delta= (w[layer+1]'*delta) .* activation_derivative(net[layer]) #  ((n_l+1 x n_l)' * n_l+1) .* n_l -> n_l
  dE[layer]+=delta*o[layer]' #  n_l x 1 * 1 x n_l-1 -> n_l x n_l-1
end

function backward!(d::LayerDerivative,l::Layer,x::Inputs,delta::Inputs)#::Inputs #delta of this layer
  delta=ld.*activation_derivative(net[end]) #  n_L .*  n_L -> n_L
  a=delta*(o[end-1])'
  dE[end]+= a #n_L x n_L-1
end

function backward!(dE::LayerDerivatives,w::Layers, o::LayerOutputs,net::LayerOutputs,ld::Outputs)
  #W indexados con 1 based y topology con 2-based
  output_layer_index=length(o)

  delta=backward!(dE[end],w[end],o[end],net[end])
  for layer=length(w)-1:-1:1
      delta=backward!(dE[layer],w[layer],o[layer],delta)
  end

end

function forward(w::Layers, x::Input)#::LayerOutputs
  o=Output[]
  net=Output[]
  push!(o,x)
  for i=1:length(w)
    net,x=forward(w[i],x)
    push!(net,net_i)
    push!(o,x)
  end
  o,net
end

function update!(l::Layer,d::LayerDerivative)

end
