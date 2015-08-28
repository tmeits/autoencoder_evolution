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
    fitness=sphere
    min_ranges,max_ranges=cubic_ranges(dimensions,1)
    pc=Evolution.SearchSpaceConfig(min_ranges,max_ranges,fitness,dimensions)
  end

function rosenbrock(x)
  -sum(100*(x[2:end]-x[1:end-1].^2).^2+(x[1:end-1]-1).^2)
end

function SphereSearchSpaceConfig(dimensions=2)
  fitness=rosenbrock
  min_ranges,max_ranges=cubic_ranges(dimensions,1)
  pc=Evolution.SearchSpaceConfig(min_ranges,max_ranges,fitness,dimensions)
end



end
