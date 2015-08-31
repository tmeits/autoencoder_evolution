using Evolution
using EvolutionTestFunctions

tic()
max_dimensions=40
repetitions=10
constructors=[EvolutionTestFunctions.SphereSearchSpaceConfig,EvolutionTestFunctions.RosenbrockSearchSpaceConfig,EvolutionTestFunctions.StyblinskiTangSearchSpaceConfig]

for i=1:length(constructors)
  print("Function ")
  println(constructors[i])
  for dimension=1:10:max_dimensions
    ssc=constructors[i](dimension)
    average_fitness=0.0

    tolerance=0.000000000000000000
    max_iterations=10*dimension
    population_size=10*dimension
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
