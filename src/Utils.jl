module Utils export @print_var,@print_size

macro print_size(zzz)
  :(println( $(string(zzz)) * " has size -> " * string(size($zzz)) ))
end

macro print_var(zzz)
  :(println($(string(zzz)) * " = " * string($zzz) ))
end



end
