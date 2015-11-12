module Utils export @print_var,@print_size

macro print_size(zzz)
  :(println( $(string(zzz)) * " has size -> " * string(size($zzz)) ))
end

macro print_var(zzz)
  :(println($(string(zzz)) * " = " * string($zzz) ))
end




function indmaximum2(A::Array{Float64,2},dim)
  @assert 1<=dim && dim<=2
  s=size(A)
  n=s[dim]
  #other_dim=dim % 2 + 1
  ind=zeros(Int,n)
  for i=1:n
    slice=slicedim(A,dim,i)
    indexes=max_index(vec(slice))
    ind[i]=indexes[1]
  end
  ind
end

function binary_to_label(y::Matrix{Int},classes)
  assert(all(sum(y,1).<=1))
  assert(all(sum(y,1).>=0))
  f(r)= if (sum(r)==0) 0 else indmax(r) end
  vec(mapslices(f,y,1))
end

# sum(y,1)== 0 o 1
function confusion_matrix(y::Matrix{Int},o::Matrix{Int})
  classes=size(y,1)
  y_label=binary_to_label(y,classes)
  o_label=binary_to_label(o,classes)
  confusion_matrix(y_label,o_label,classes)
end

function confusion_matrix(y::Vector{Int},o::Vector{Int},classes)
  confusion=zeros(classes+1,classes)
  #rows are estimated class,columns true classes
  for i=1:length(y)
    if o[i]==0
      confusion[classes+1,y[i]]+=1
    else
      confusion[o[i],y[i]]+=1
    end
  end
  confusion
end


function probability_to_binary_max(o::Matrix{Float64})#::Matrix{Int}
  r=zeros(Int,size(o))
  indices=mapslices(indmax,o,1)
  for i=1:length(indices)
    r[indices[i],i]=1
  end
  r
end
end
