module Evolution

type EvolutionConfig
  learning_rate::Float64
  crossover_probability::Float64
  tolerance::Float64
  max_iterations::Int
  population_size::Int
end

type SearchSpaceConfig
    min_ranges::Array{Float64,1}
    max_ranges::Array{Float64,1}
    fitness::Function
    dimensions::Int
end

type Population
    config::SearchSpaceConfig
    individuals::Array{Float64,2}
end

function random_population(pc::SearchSpaceConfig,population_size::Int)
  population::Population
  individuals = rand(pc.dimensions,population_size)
  for i=1:population_size
    individuals[:,i] = individuals[:,i].* (pc.max_ranges-pc.min_ranges)+pc.min_ranges
  end
  population=Population(pc,individuals)
  population
end


function differential_evolution(ec::EvolutionConfig,ssc::SearchSpaceConfig)
   p=random_population(ssc,ec.population_size)
   i=0
   best_fitness=maximum(population_fitnesses(p))
   last_best_fitness=best_fitness*2+2*ec.tolerance+1 # dummy value to force entry condition to true
   #print_results(p,i)
   while(i<ec.max_iterations && abs(best_fitness-last_best_fitness)>=ec.tolerance)
      p=differential_evolution_iteration(ec,p)
      last_best_fitness=best_fitness
      best_fitness=maximum(population_fitnesses(p))
      i+=1
      #print_results(p,i)
   end
   p,i
end

function differential_evolution_iteration(c::EvolutionConfig,p::Population)

  for i=1:size(p.individuals,2)
      base_mutant=generate_base_mutated(c,i,p)
      crossovers=rand(p.config.dimensions).<c.crossover_probability
      crossovers[rand(1:p.config.dimensions)]=true
      mutant=p.individuals[:,i]
      mutant[crossovers]=base_mutant[crossovers]
      if (p.config.fitness(p.individuals[:,i])<p.config.fitness(mutant))
        p.individuals[:,i]=mutant
      end
  end
  p
end

function generate_base_mutated(c::EvolutionConfig,excluded_index::Int,p::Population)
  indices=random_indices_except(size(p.individuals,2),3,[excluded_index])
  base=p.individuals[:,indices[1]]
  r1=p.individuals[:,indices[2]]
  r2=p.individuals[:,indices[3]]
  base_mutant=base+c.learning_rate*(r2-r1)
  base_mutant
end


function population_fitnesses(p::Population)
  mapslices(p.config.fitness, p.individuals, 1)
end

function random_indices_except(n,k,except)
  if (k>n)
    error("No puedo darte k porque tengo n y k > n")
  end
  selected_indices=zeros(1,k);
  all_indices=1:n
  for i=1:k
    r=rand(all_indices)
    while (r in except) || (r in selected_indices)
      r=rand(all_indices)
    end
    selected_indices[i]=r;
  end
  selected_indices
end

function print_results(p,iterations)
  fitnesses=population_fitnesses(p)
  results=[fitnesses ; p.individuals]'
  println("DE results :")
  println(results)
  @printf "%d iterations executed\n" iterations
  best_individual_index=indmax(fitnesses)
  best_individual=p.individuals[:,best_individual_index]
  best_individual_fitness=p.config.fitness(best_individual)
  @printf "Best individual (fitness %f)\n" best_individual_fitness
  println(best_individual)
end


end
