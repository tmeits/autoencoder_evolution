using Evolution
using EvolutionTestFunctions

tic()
tolerance=0.000000000000000000
max_iterations=100
population_size=10
ec=Evolution.EvolutionConfig(0.2,0.01,tolerance,max_iterations,population_size)

ssc=EvolutionTestFunctions.SphereSearchSpaceConfig(1)
#ssc=EvolutionTestFunctions.RosenbrockSearchSpaceConfig(2)

final_population,iterations=Evolution.differential_evolution(ec,ssc)
time=toc()
Evolution.print_results(final_population,iterations)
@printf "Time elapsed %f \n" time
