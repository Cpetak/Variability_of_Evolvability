import numpy as np
import argparse
from tqdm import trange
from pathlib import Path

import helper

"""#Evolutionary algorythm"""

def evolutionary_algorithm(pop_size, grn_size, num_cells, dev_steps, mut_rate, num_generations, selection_prop, rules, mut_size, folder, seed_ints, season_len, job_array_id):

  #Setting up
  #job_array_id = job_array_id + 5

  #Creating start expression pattern
  geneid = 1
  seeds=[]
  inputs=[]

  random_seed = False
  #seed_ints = [0,1] #index of input in the 100 inputs file
  print(seed_ints)

  if random_seed:
      start_patterns = np.loadtxt("100_inputs.txt")
      start_patterns = np.reshape(start_patterns, (100,22)).astype(int)
      for seed_int in seed_ints:
        print(seed_int)
        seeds.append(start_patterns[int(seed_int),:])
        start_expression = helper.seed2expression(start_patterns[int(seed_int),:], pop_size, num_cells, grn_size, geneid)
        inputs.append(start_expression)
  else:
    for seed_int in seed_ints:
      #Make seeds, 1024 is one-hot
      start_pattern = helper.seedID2string(seed_int, num_cells)
      seeds.append(start_pattern)
      #Make starting expression for whole population
      start_expression = helper.seed2expression(start_pattern, pop_size, num_cells, grn_size, geneid)
      inputs.append(start_expression)

  print(seeds)

  #Creating targets
  targets=[]
  seeds_ints=[]
  for idx, seed in enumerate(seeds):
    targets.append(helper.rule2targets_wrapped_wstart(int(rules[idx]), L=dev_steps+1, N=num_cells, start_pattern=seed))
    binary_string = ''.join(seed.astype(str))
    seeds_ints.append(int(binary_string, 2)) #for file naming
  seeds_id='-'.join([str(number) for number in seeds_ints]) #id of start pattern for each season
  rules_id='-'.join([str(number) for number in rules])
 
  #Logging
  max_fits = []
  ave_fits = []
  all_pheno_stds = []
  pheno_stds = []
  
  saveat = list(range(num_generations))

  if season_len > num_generations: #static experiment
    filename = f"{folder}/stats_{season_len}_{rules[0]}_{seeds_ints[0]}_{job_array_id}"
  else:
    filename = f"{folder}/stats_{season_len}_{rules_id}_{seeds_id}_{job_array_id}"

  #Defining variables
  curr = 0
  worst= -num_cells*dev_steps #not a bug because you are getting the first row good for sure
  selection_size=int(pop_size*selection_prop)
  num_child = int(pop_size / selection_size) - 1
  tot_children = num_child * selection_size
  num_genes_mutate = int((grn_size + 2) * grn_size * tot_children * mut_rate)

  #Creating population
  pop = np.random.randn(pop_size, grn_size+2, grn_size).astype(np.float64)

  # Main for loop
  for gen in trange(num_generations):

    # Generating phenotypes
    #Return [pop_size, dev_stepss+1, num_cellsxgrn_size] np.float64 array
    phenos = helper.develop(inputs[curr], pop, dev_steps, pop_size, grn_size, num_cells)
    #get second gene for each cell only, the one I decided will matter for the fitness
    #pop_size, dev_steps, NCxNG
    p=phenos[:,:,1::grn_size]

    # Logging phenotypic variation
    if gen > 0:
      #across pop
      all_pheno_std=np.std(p, axis=0)
      all_pheno_std = all_pheno_std.mean()
      all_pheno_stds.append(all_pheno_std)

      #kids of same parent
      child_phenotypes = p[children_locs] 
      reshaped=np.reshape(child_phenotypes, (num_child, len(parent_locs), (dev_steps+1)*num_cells))
      pheno_std=np.std(reshaped,axis=0) #one std for each of the parents, so pop_size*trunc_prop now 10
      pheno_std = pheno_std.mean(1).mean()
      pheno_stds.append(pheno_std)

    #Calculating fitnesses
    fitnesses = []
    for target in targets:
      temp_fitnesses = helper.fitness_function_ca(p, target)
      temp_fitnesses=1-(temp_fitnesses/worst) #0-1 scaling
      fitnesses.append(temp_fitnesses)
    
    fitnesses_to_use = fitnesses[curr]

    #Selection
    perm = np.argsort(fitnesses_to_use)[::-1]

    #Logging
    best_grn = pop[perm[0]]
    max_fit=fitnesses[curr].max().item()
    ave_fit=fitnesses[curr].mean().item()
    max_fits.append(max_fit)  # keeping track of max fitness
    ave_fits.append(ave_fit)  # keeping track of average fitness

    # location of top x parents in the array of individuals
    parent_locs = perm[:selection_size]
    # location of individuals that won't survive and hence will be replaced by others' children
    children_locs = perm[selection_size:]

    parents = pop[parent_locs]
    children = np.tile(parents, (num_child, 1, 1))

    #Mutation
    mutations = np.random.randn(num_genes_mutate) * mut_size
    x, y, z = children.shape
    xs = np.random.choice(x, size=num_genes_mutate)
    ys = np.random.choice(y, size=num_genes_mutate)
    zs = np.random.choice(z, size=num_genes_mutate)
    children[xs, ys, zs] = children[xs, ys, zs] + mutations

    pop[children_locs] = children  # put children into population

    #Change environment
    if gen % season_len == season_len - 1: # flip target
      curr = (curr + 1) % len(targets)

  #Saving to file
  with open(filename+"_maxfits.txt", 'w') as f:
    np.savetxt(f, max_fits, newline=" ")
  with open(filename+"_avefits.txt", 'w') as f:
    np.savetxt(f, ave_fits, newline=" ")
  with open(filename+"_all_pheno_stds.txt", 'w') as f:
    np.savetxt(f, all_pheno_stds, newline=" ")
  with open(filename+"_pheno_stds.txt", 'w') as f:
    np.savetxt(f, pheno_stds, newline=" ")

  return max_fit

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--pop_size', type=int, default=1000, help="Population size")
  parser.add_argument('--grn_size', type=int, default=22, help="GRN size") 
  parser.add_argument('--num_cells', type=int, default=22, help="Number of cells") 
  parser.add_argument('--dev_steps', type=int, default=22, help="Number of developmental steps") 

  parser.add_argument('--selection_prop', type=float, default=0.1, help="Percent pruncation") 
  parser.add_argument('--mut_rate', type=float, default=0.1, help="Number of mutations") 
  parser.add_argument('--mut_size', type=float, default=0.5, help="Size of mutations") 
  parser.add_argument('--num_generations', type=int, default=30_000, help="Number of generations") #9899
  parser.add_argument('--season_len', type=int, default=300, help="season length")

  parser.add_argument('--seed_ints', nargs='+', default=[4147842,1238860], help='List of seeds in base 10')
  parser.add_argument('--rules', nargs='+', default=[30,30], help='List of rules')

  parser.add_argument('--job_array_id', type=int, default=0, help="Job array id to distinguish runs")

  args = parser.parse_args()

  #69904,149796
  #1024
  #to_seed = lambda n, N : np.array(list(map(int, format(n, f"0{N}b"))))

  #Writing to file
  folder_name = Path("~/scratch/non_detailed_save/full_pheno_std_long").expanduser()
  #folder = helper.prepare_run(folder_name)
  args.folder = folder_name

  #args.num_cells = args.dev_steps

  #Make sure that user provided a rule and a seed for each alternative environment
  assert len(args.rules) == len(args.seed_ints), f"Num rules {len(args.rules)} != num seeds {len(args.seed_ints)}"

  print("running code", flush=True)
  evolutionary_algorithm(**vars(args))

  print("completed")

  
    
