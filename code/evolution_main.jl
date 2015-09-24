using Evolution
using EvolutionTestFunctions

tic()
tolerance=0.000000000000000000
max_iterations=1000
population_size=20
ec=Evolution.EvolutionConfig(0.5,0.1,tolerance,max_iterations,population_size)

#ssc=EvolutionTestFunctions.SphereSearchSpaceConfig(50)
#ssc=EvolutionTestFunctions.RosenbrockSearchSpaceConfig(100)
ssc=EvolutionTestFunctions.StyblinskiTangSearchSpaceConfig(20)
#ssc=EvolutionTestFunctions.AckleySearchSpaceConfig()

final_population,iterations=Evolution.differential_evolution(ec,ssc)
time=toc()

Evolution.print_results(final_population,iterations)
@printf "Time elapsed %f \n" time
