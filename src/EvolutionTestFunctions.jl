module EvolutionTestFunctions
using Evolution

  function cubic_ranges(dimensions,radius)
    rectangular_ranges(dimensions,-radius,radius)
  end
  function rectangular_ranges(dimensions,min_range,max_range)
    max_ranges=max_range*ones(dimensions)
    min_ranges=min_range*ones(dimensions)
    min_ranges,max_ranges
  end

  function sphere(x)
    -dot(x,x)
  end


  function SphereSearchSpaceConfig(dimensions=2)
    #Minimum 0 at (0,...,0)
    fitness=sphere
    min_ranges,max_ranges=cubic_ranges(dimensions,1)
    pc=Evolution.SearchSpaceConfig(min_ranges,max_ranges,fitness,dimensions)
  end


function rosenbrock(x)
  -sum(100*(x[2:end]-x[1:end-1].^2).^2+(x[1:end-1]-1).^2)
end

function RosenbrockSearchSpaceConfig(dimensions=2)
  #Minimum 0 at (1,...,1)
  fitness=rosenbrock
  min_ranges,max_ranges=cubic_ranges(dimensions,5)
  pc=Evolution.SearchSpaceConfig(min_ranges,max_ranges,fitness,dimensions)
end


function styblinski_tang(x)
  -sum(x.^4-16*x.^2+5*x)/2-39.166165*length(x)
end

function StyblinskiTangSearchSpaceConfig(dimensions=2)
  #Minimum -39.166165*dimensions at (-2.90,...,-2.90) (aprox)
  fitness=styblinski_tang
  min_ranges,max_ranges=cubic_ranges(dimensions,5)
  pc=Evolution.SearchSpaceConfig(min_ranges,max_ranges,fitness,dimensions)
end



function ackley(a)
  x=a[1]
  y=a[2]
  -(-20*exp(-0.2*sqrt(0.5*(x^2+y^2)))-exp(0.5*(cos(2*pi*x)+cos(2*pi*y)))+exp(1)+20)
end

function AckleySearchSpaceConfig()
  #Minimum 0 at (0,0)
  dimensions=2
  fitness=ackley
  min_ranges,max_ranges=cubic_ranges(dimensions,5)
  pc=Evolution.SearchSpaceConfig(min_ranges,max_ranges,fitness,dimensions)
end


end
