  a=rand(3,4)
  b=rand(4,2,2)
  ia=2
  ib=1
  sa=collect(size(a))
  sb=collect(size(b))
  dim_size=sa[ia]
  splice!(sa,ia)
  splice!(sb,ib)
  new_size=[sa;sb]
  c=zeros(new_size...)

for i=1:dim_size
  
end
