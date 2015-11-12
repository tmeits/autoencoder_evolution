# Pkg.add("Optim")
# Pkg.add("FactCheck")

push!(LOAD_PATH,dirname(@__FILE__))
include("Neural.jl")
include("Utils.jl")

using FactCheck
using Neural
using Utils
using RDatasets



function get_iris()
  iris = dataset("datasets", "iris")
  iris_x =convert(Matrix{Float64},iris[1:4])'
  iris_y_labels=1.*(iris[5].=="setosa")+2.*(iris[5].=="versicolor")+3.*(iris[5].=="virginica")
  iris_labels=unique(iris_y_labels)
  iris_y=zeros(length(iris_labels),length(iris_y_labels))
  for i=1:length(iris_y_labels)
    iris_y[iris_y_labels[i],i]=1
  end
  iris_x=iris_x.-mean(iris_x,2)
  iris_x=iris_x./std(iris_x,2)
  iris_x= [ones(1,size(iris_x,2)); iris_x]
  iris_x,iris_y
end


topology = [3,10,2] #"output" is the last one
Wz = Neural.zero_weights(topology)
W = Neural.random_weights(topology)
dW = Neural.zero_weights(topology)
activation = Neural.activation
x = [0.1 0.5 0.9]'
n = Neural.Net(topology, W)
y = [0.5 0.3]'
o = activation(W[2] * activation(W[1]*x))
random_samples=10
xs = rand(3,random_samples)
ys = rand(2,random_samples)
os = activation(W[2] * activation(W[1]*xs))

function derivative(W::Neural.Layers,x::Neural.Inputs,y::Neural.Outputs)
  dE = Neural.zero_weights(W)
  o,net=Neural.forward(W,vec(x))
  delta=Neural.loss_derivative(o[3]'',y).*Neural.activation_derivative(net[2])
  dE[2]=delta * o[2]'
  delta=(W[2]'*delta).*Neural.activation_derivative(net[1])
  dE[1]=delta * o[1]'
  dE
end

facts("Test layer creation") do
  @fact length(Wz) --> 2
  @fact size(Wz[1]) --> (topology[2],topology[1])
  @fact size(Wz[2]) --> (topology[3],topology[2])
end

facts("Test layer derivatives creation") do
  @fact length(dW) --> 2
  @fact size(dW[1]) --> (topology[2],topology[1])
  @fact size(dW[2]) --> (topology[3],topology[2])
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

facts("Test loss") do
  @fact Neural.loss(vec(y),vec(y)) --> [0.0]''
  @fact Neural.loss(y,y) --> [0.0]''
  @fact Neural.loss(ys,ys) --> zeros(1,size(ys,2))
  for i=1:size(ys,2)
    yi=ys[:,i]
    @fact Neural.loss(yi,yi+1)[1]  --> greater_than(0.0)
  end
end

facts("Test loss derivative") do
  @fact Neural.loss_derivative(y,y) --> [0.0;0.0]''
  @fact Neural.loss_derivative(ys,ys) --> zeros(ys)
  for i=1:size(ys,2)
    yi=ys[:,i]
    dy=Neural.loss_derivative(yi+1,yi)
    dy2=Neural.loss_derivative(yi,yi+1)
    for j=1:length(dy)
      @fact dy[j]  --> 1
      @fact dy2[j]  --> -1
    end
  end
end

facts("Test apply") do
  o2 = Neural.apply(n.layers, x)
  @fact o2 --> o
end

facts("Test forward") do
  o2,net = Neural.forward(n.layers, vec(x))
  @fact vec(o2[end]) --> roughly(vec(o))
end

facts("Test derivative sizes") do
  dE = Neural.net_error_derivative(W,x,y)
  @fact length(dE) --> 2
  @fact size(dE[1]) --> (topology[2],topology[1])
  @fact size(dE[2]) --> (topology[3],topology[2])
end


facts("Test derivative with analytic method") do
  dE2 = Neural.net_error_derivative(W,x,y)
  dE=derivative(W,x,y)
  @fact dE2[1] --> roughly(dE[1])
  @fact dE2[2] --> roughly(dE[2])
end

# facts("Test gradient checking") do
#   EPS = 1.0^-20
#
#   dE = Neural.zero_weights(topology)
#   o,net=Neural.forward(W,vec(x))
#   ld=Neural.loss_derivative(o[end],vec(y))
#   Neural.backward!(dE,W,o,net,ld)
#
#   dWp = Neural.zero_weights(topology)
#   Wl = deepcopy(W)
#   for l=1:length(W)
#     #Wl = copy(W[l])
#     (nUnits, nWeights) = size(Wl[l])
#     for u = 1:nUnits
#       for weight = 1:nWeights
#         v = Wl[l][u, weight]
#         Wl[l][u, weight] += EPS
#         Oplus = Neural.loss(Neural.apply(Wl, x),y)
#         Wl[l][u, weight] -= 2*EPS
#         Ominus = Neural.loss(Neural.apply(Wl, x),y)
#         dWp[l][u,weight] = (Oplus - Ominus)[1] / (2 * EPS)
#         Wl[l][u, weight] = v
#       end
#     end
#   end
#   #println(string("dWp", dWp))
#   for l = 1:length(W)
#     @fact dE[l] --> roughly(dWp[l]) "wrong derivative at layer $l"
#   end
#   #println(dWp[1])
#   #println(dE[1])
# end

# facts("Test network error derivative for dataset") do
#   EPS = 1.0^-5
#   dE = Neural.net_error_derivative(W, xs, ys, topology)
#   dWp = Neural.zero_weights(topology)
#   Wl = deepcopy(W)
#   for l=1:length(W)
#     (nUnits, nWeights) = size(Wl[l])
#     for u = 1:nUnits
#       for weight = 1:nWeights
#         v = Wl[l][u, weight]
#         Wl[l][u, weight] += EPS
#         Oplus = Neural.net_error(Wl, xs, ys)
#         Wl[l][u, weight] -= 2*EPS
#         Ominus = Neural.net_error(Wl, xs, ys)
#         dWp[l][u,weight] = (Oplus - Ominus) / (2 * EPS)
#         Wl[l][u, weight] += EPS
#         @fact Wl[l][u, weight] --> v "Wrong restored value"
#       end
#     end
#   end
#   #println(string("dWp", dWp))
#   for l = 1:length(W)
#     @fact dWp[l] --> roughly(dE[l]; atol=EPS) "wrong error derivative at layer $l"
#   end
# end

# facts("Test net error") do
#   e = Neural.net_error(W, xs, os)
#   e2 = Neural.net_error(W, xs + rand(size(xs)), os)
#   e3 = Neural.net_error(W, xs, os + rand(size(os)))
#   @fact e --> 0
#   @fact e2 --> greater_than(0)
#   @fact e3 --> greater_than(0)
# end

# facts("Test training") do
#   #xs = [1 2 ; ]
#   #NeuralTest.topology = [3,1,1]
#   n = Neural.Net(topology, copy(W))
#   Neural.train!(n, xs, ys)
#   os2 = Neural.apply(n.layers, xs)
#   println(n.layers)
#   @fact os2 --> roughly(ys)
# end


facts("Test net training") do
  xs,ys=get_iris()
  topology=[size(xs,1),5,4,4,size(ys,1)]
  n=Neural.trainbp_minibatch(topology, xs, ys,6000,0.9,20)
  #n=Neural.train(topology, xs, ys)
  os2 = Neural.apply(n.layers, xs)
  confusion=Utils.confusion_matrix(Utils.probability_to_binary_max(ys),Utils.probability_to_binary_max(os2))
  mean_error=mean((round(os2)-ys).^2)
  println(mean_error)
  println(confusion)
  #println(os2')
  #println(round(os2'))
  #println(ys')
  #@fact round(os2)' --> roughly(ys')
end



# using Datasets
# x,y,l=Datasets.iris()
# ys=zeros(3,length(y))
#   for i=1:unique(y)
#    ys[i,y.==i]=1
#   end

return
