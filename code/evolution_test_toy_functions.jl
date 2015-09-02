using Evolution
using EvolutionTestFunctions

tic()
max_dimensions=100
repetitions=10
constructors=[EvolutionTestFunctions.SphereSearchSpaceConfig,EvolutionTestFunctions.RosenbrockSearchSpaceConfig,EvolutionTestFunctions.StyblinskiTangSearchSpaceConfig]
difficulty=[1,3,1.5]


for i=1:length(constructors)
  print("Function ")
  println(constructors[i])
  for dimension=2:30:max_dimensions+2
    ssc=constructors[i](dimension)
    average_fitness=0.0

    tolerance=0.000000000000000000
    max_iterations=floor(10*dimension*difficulty[i])
    population_size=floor(10*dimension*difficulty[i])
    ec=Evolution.EvolutionConfig(0.5,0.1,tolerance,max_iterations,population_size)

    for r=1:repetitions
      final_population,iterations=Evolution.differential_evolution(ec,ssc)
      p=final_population
      fitnesses=Evolution.population_fitnesses(p)
      best_individual_index=indmax(fitnesses)
      best_individual_fitness=fitnesses[best_individual_index]
      average_fitness=average_fitness+best_individual_fitness
    end
    average_fitness=average_fitness/repetitions
    @printf "   N=%02d" dimension
    @printf " mean fitness=%f\n"  average_fitness
  end
end

#Evolution.print_results(final_population,iterations)

time=toc()
@printf "Time elapsed %f \n" time
