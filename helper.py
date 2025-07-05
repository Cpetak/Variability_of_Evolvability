#IMPORT PACKAGES

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib.lines import Line2D
from numba import njit, prange
from pathlib import Path
import seaborn as sns
import hashlib
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import math
import colorsys
from collections import defaultdict
from matplotlib.patches import Circle
from scipy.stats import mannwhitneyu

#Parameter for sigmoid function
ALPHA = 10

def prepare_run(folder_name):
    
    folder = Path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)

    return folder

def map_to_range(value):
  return int(hashlib.sha256(str(value).encode()).hexdigest(), 16) % (2**32)

def calculate_distance(x1, y1, x2, y2):
    # Calculate the distance using the distance formula
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def calc_div_BH(kid_fits, landmarks):
    d_A = np.sqrt((kid_fits[:,:,:,0,:,:] - landmarks.iloc[0]["x"]) ** 2 + (kid_fits[:,:,:,1,:,:] - landmarks.iloc[0]["y"]) ** 2) # distance to target A
    d_B = np.sqrt((kid_fits[:,:,:,0,:,:] - landmarks.iloc[1]["x"]) ** 2 + (kid_fits[:,:,:,1,:,:] - landmarks.iloc[1]["y"]) ** 2) # distance to target B

    dists = np.minimum(d_A, d_B)
    max_distance=calculate_distance(landmarks.iloc[1]["x"],landmarks.iloc[1]["y"],landmarks.iloc[6]["x"],landmarks.iloc[6]["y"])
    dists = 1-dists/max_distance # if distance is 0, it is 1 (max), if distance is 1, which is the biggest, it is 0 (min)
    dists = dists ** 2 #make it nonlinear to punish for really bad fitness, away from any target
    dists = np.mean(dists, axis=4) #so range: 0-1 and the bigger the better - the closer it is to one of the targets

    std1 = np.std(kid_fits[:,:,:,0,:,:], axis = 4)
    std2 = np.std(kid_fits[:,:,:,1,:,:], axis = 4)
    stds = (std1 + std2) /2
    max_stds = (np.std([landmarks.iloc[0]["x"], landmarks.iloc[1]["x"]]) + np.std([landmarks.iloc[0]["y"], landmarks.iloc[1]["y"]])) /2
    tresholded_stds = np.minimum(stds, max_stds) # greater std than max_std is not needed to be a perfect diversifier
    f_stds = tresholded_stds/max_stds #what percentage of max this is, so range 0-1. 1 = as diverse as it can be, the bigger the better

    div_BH = (f_stds + dists) / 2 #averaged so that is it between 0 and 1
    div_BH_mean = np.mean(div_BH, axis = 3) #average across parents 

    return div_BH, div_BH_mean

def calc_conz_BH(all_fits, landmarks_list):
    #input: whole dataset with all experiments, repetitions, generations, fitnesses, individuals

    #setting up variables
    bestgen=np.array([landmarks_list[0],landmarks_list[1]])
    max_distance_best = calculate_distance(bestgen[0], bestgen[1], landmarks_list[1], landmarks_list[2]) #bottom left corner
    max_distance_line = calculate_distance(bestgen[0], bestgen[1], landmarks_list[3], landmarks_list[4]) #top right corner
    bot=np.array([0,0])
    top=np.array([1,1])
    diagonal_vector = top - bot
    diagonal_vector_expanded = diagonal_vector[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    bot_expanded = bot[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

    #calculating generalism for each individual
    distance_best = np.sqrt((bestgen[0] - all_fits[:,:,:,0,:]) ** 2 + (bestgen[1] - all_fits[:,:,:,1,:]) ** 2) #want this to be small
    prop_distance_best = 1 - distance_best/max_distance_best

    distance_line = np.cross(diagonal_vector_expanded, all_fits - bot_expanded, axis = 3) / np.linalg.norm(diagonal_vector_expanded) #want this to be small
    prop_distance_line = 1 - distance_line/max_distance_line

    conz_BH = (prop_distance_best + prop_distance_line) / 2
    conz_BH_mean = np.mean(conz_BH, axis = 3)  #average across individuals  

    return conz_BH, conz_BH_mean

def calc_pheno_variation(p, children_locs, num_child, parent_locs, dev_steps, num_cells, where_overlap, where_no_overlap):
    #dev_steps = 0 #NOTE: if only last dev step considered
    child_phenotypes = p[children_locs] 
    # inner most list: first: first born of each parent, second: second borns of each parent, etc.
    # so it is NOT all kids of 1 parent, then the other parent, etc.
    reshaped=np.reshape(child_phenotypes, (num_child, len(parent_locs), (dev_steps+1)*num_cells))
    #reshaped is num child per parent, num parents, (dev_steps+1)*num_cells shaped. 
    # so [:,0,:] is all kids of one parent
    pheno_std=np.std(reshaped,axis=0) #one std for each of the parents, so pop_size*trunc_prop now 10
    pheno_std = pheno_std.mean(1).mean() #first averaged across cells, then averaged across individuals in the population
    # generic phenotypic variation among offspring of the same parent

    #looking for more sophisticated phenotypic variation:
    if True: #NOTE: False if only last dev step considered
        reshaped2D=np.reshape(reshaped, (num_child, len(parent_locs), dev_steps+1, num_cells))

        values_they_should_match = reshaped2D[:,:,where_overlap[0],where_overlap[1]]
        #values_they_should_match.shape #4 kids, 2 parents, N values, where N is the number of cells where they overlap
        matching_std = np.std(values_they_should_match, axis=0) #among the 4 kids of 1 parent, output is an N long list for each of the 2 parents
        matching_std = matching_std.mean(axis=1) #average across the N overlaps, to get 1 value for each parent

        #repeat for non-overlap
        values_they_shouldnt_match = reshaped2D[:,:,where_no_overlap[0],where_no_overlap[1]]
        #values_they_should_match.shape #4 kids, 2 parents, N values, where N is the number of cells where they don't overlap
        nonmatching_std = np.std(values_they_shouldnt_match, axis=0) #among the 4 kids of 1 parent, output is an N long list for each of the 2 parents
        nonmatching_std = nonmatching_std.mean(axis=1) #average across the N non overlaps, to get 1 value for each parent

        #minimum std is 0, max is 0.5 in the case of values that range between 0 and 1
        combined_std = nonmatching_std - matching_std
        averaged_combined_std = np.mean(combined_std)
        best_std_id = np.argmax(combined_std)

        return pheno_std, np.max(combined_std), best_std_id, averaged_combined_std
    #return pheno_std

@njit("f8[:,:](f8[:,:],i8, i8)")
def sigmoid(x,a,c):
  return 1/(1 + np.exp(-a*x+c))

def fitness_function_ca(phenos, targ):
  """
  Takes phenos which is pop_size * iters+1 * num_cells and targ which is iters+1 * num_cells
  Returns 1 fitness value for each individual, np array of size pop_size
  """
  return -np.abs(phenos - targ).sum(axis=1).sum(axis=1)

def seedID2string(seed_int, num_cells):
  #takes an integer, turns it into a starting pattern
  binary_string = bin(int(seed_int))[2:]
  binary_list = [int(digit) for digit in binary_string]
  start_pattern = np.array(binary_list)
  start_pattern=np.pad(start_pattern, (num_cells-len(start_pattern),0), 'constant', constant_values=(0))
  return start_pattern

def seed2expression(start_pattern, pop_size, num_cells, grn_size, geneid):
  #takes a starting pattern and makes a population of starting gene expressions
  start_gene_values = np.zeros((pop_size, int(num_cells * grn_size)))
  start_gene_values[:,geneid::grn_size] = start_pattern
  start_padded_gene_values = np.pad(start_gene_values, [(0,0),(1,1)], "wrap")
  start_padded_gene_values = np.float64(start_padded_gene_values)
  return start_padded_gene_values

#DO THE MULTICELLULAR DEVELOPMENT
@njit("f8[:](f8[:], f8[:,:], i8, i8)")
#Make sure that numpy imput in foat64! Otherwise this code breaks
def update_with_grn(padded_gene_values, grn, num_cells, grn_size):
  """
  Gene expression pattern + grn of a single individual -> Next gene expression pattern
  Takes
  - padded_gene_values: np array with num_genes * num_cells + 2 values
  Gene 1 in cell 1, gene 2 in cell 1, etc then gene 1 in cell 2, gene 2 in cell 2... plus left-right padding
  - grn: np array with num_genes * num_genes +2 values, shape of the GRN
  """
  #This makes it so that each cell is updated simultaneously
  #Accessing gene values in current cell and neighbors
  windows = np.lib.stride_tricks.as_strided(
      padded_gene_values, shape=(num_cells, grn_size + 2), strides=(8 * grn_size, 8)
  )
  #Updating with the grn
  next_step = windows.dot(grn)
  c = ALPHA/2
  next_step = sigmoid(next_step,ALPHA,c)

  #Returns same shape as padded_gene_values
  return next_step.flatten()

@njit("f8[:](f8[:], f8[:,:], i8, i8)")
#Make sure that numpy imput in foat64! Otherwise this code breaks
def update_internal(padded_gene_values, grn, num_cells, grn_size):
  """
  Gene expression pattern + grn of a single individual -> Next gene expression pattern
  Takes
  - padded_gene_values: np array with num_genes * num_cells + 2 values
  Gene 1 in cell 1, gene 2 in cell 1, etc then gene 1 in cell 2, gene 2 in cell 2... plus left-right padding
  - grn: np array with num_genes * num_genes +2 values, shape of the GRN
  """
  #Updating with the internal grn
  internal_grn = grn[1:-1,:]
  gene_vals = padded_gene_values[1:-1].copy()
  gene_vals = gene_vals.reshape(num_cells,grn_size)
  next_step = gene_vals.dot(internal_grn)
  c = ALPHA/2
  next_step = sigmoid(next_step,ALPHA,c)

  #Returns same shape as padded_gene_values
  return next_step.flatten()

#Might be faster non-parallel depending on how long computing each individual takes!
@njit(f"f8[:,:,:](f8[:,:], f8[:,:,:], i8, i8, i8, i8)", parallel=True)
def develop(
    padded_gene_values,
    grns,
    iters,
    pop_size,
    grn_size,
    num_cells
):
  """
  Starting gene expression pattern + all grns in the population ->
  expression pattern throughout development for each cell for each individual
  DOES NOT assume that the starting gene expression pattern is the same for everyone
  returns tensor of shape: [POP_SIZE, N_ITERS+1, num_cellsxgrn_size]
  N_ITERS in num developmental steps not including the initial step
  """
  NCxNGplus2 = padded_gene_values.shape[1]
  history = np.zeros((pop_size, iters+1, NCxNGplus2 - 2), dtype=np.float64)

  #For each individual in parallel
  for i in prange(pop_size):
    #IMPORTANT: ARRAYS IN PROGRMING when "copied" just by assigning to a new variable (eg a=[1,2,3], b = a)
    #Copies location and so b[0]=5 overwrites a[0] too! Need .copy() to copy variable into new memory location
    grn = grns[i]
    state = padded_gene_values[i].copy()
    history[i, 0, :] = state[1:-1].copy() #saving the initial condition
    #For each developmental step
    for t in range(iters):
      #INTERNAL
      state[1:-1] = update_internal(state, grn, num_cells, grn_size)
      #To wrap around, change what the pads are
      state[0] = state[-2] #the last element of the output of update_with_grn
      state[-1] = state[1]
      #EXTERNAL
      state[1:-1] = update_with_grn(state, grn, num_cells, grn_size)
      #To wrap around, change what the pads are
      state[0] = state[-2] #the last element of the output of update_with_grn
      state[-1] = state[1]
      history[i, t+1, :] = state[1:-1].copy()
  return history

#Torch implementation

def update_pop_torch(state, grns, NC, NG):
    """
    Receives:
        - state of shape (POP, NCxNG)
        - grns of shape (POP, NG+2, NG)

    Updates the state applying each individual's grn
    to windows that include one communication gene from
    the immediate neighbors (see below for explanation)

    Returns:
        - new state od shape (POP, NCxNG)

    e.g.

    POP = 2 # ind1, ind2
    NC = 3  # cell1 cell2
    NG = 4  # g1, g2, g3, g4

    state:
           g1 g2 g3 g4   g1 g2 g3 g4
           [1, 2, 3, 4]  [5, 6, 7, 8]   ...

               cell1       cell2      cell3
            ----------  ----------  ----------
    ind1 [[ 1  2  3  4  5  6  7  8  9 10 11 12]
    ind2  [13 14 15 16 17 18 19 20 21 22 23 24]]

    padded w/ zeros:

        [[ 0  1  2  3  4  5  6  7  8  9 10 11 12  0]
         [ 0 13 14 15 16 17 18 19 20 21 22 23 24  0]]

    windows:

        [[[ 0  1  2  3  4  5]
          [ 4  5  6  7  8  9]
          [ 8  9 10 11 12  0]]

         [[12  0  0 13 14 15]
          [14 15 16 17 18 19]
          [18 19 20 21 22 23]]]

    assuming dtype is the size of a single entry in state

    state.shape   = (POP, NC * NG)
    state.strides = (NC * NG * dtype, dtype)

    windows.shape   = (POP, NC, NG+2)
    windows.strides = (NC * NG * dtype, NG * dtype, dtype)
    """
    device="cpu"
    POP, _ = state.shape
    #padded = np.pad(state, pad_width=[(0, 0), (1, 1)])
    padded = state.copy()
    view_shape = (POP, NC, NG + 2)
    strides = [padded.strides[0], state.strides[0] // NC, state.strides[1]]
    windows = np.lib.stride_tricks.as_strided(padded, shape=view_shape, strides=strides)
    tgrns = torch.from_numpy(grns).to(device)
    tgrns.requires_grad_(True)
    twins = torch.from_numpy(windows).to(device)
    #with torch.no_grad():
    res = torch.matmul(twins, tgrns).cpu()
    res = torch.clip(res, 0, 1).reshape(POP, NC * NG)
    return res.cpu().numpy()
    # new_state = np.clip(np.matmul(windows, grns), 0, 1)
    # return new_state.reshape(POP, NC * NG)


def develop_torch(
    state,
    grns,
    iters,
    pop_size,
    grn_size,
    num_cells,
):
    _, NCxNG = state.shape
    history = np.zeros((iters+1, pop_size, NCxNG-2), dtype=np.float64)

    # for i in prange(pop_size):
    #     state = gene_values[i].copy()
    #     grn = grns[i]
    #     for t in range(iters):
    #         state[1:-1] = update_with_grn(state, grn, num_cells, grn_size)
    #         history[i, t, :] = state[1:-1].copy()

    history[0] = state[:,1:-1].copy()
    for t in range(iters):
        print(state.shape)
        state = update_pop_torch(state, grns, num_cells, grn_size)
        print(state.shape)
        history[t+1] = state.copy()

    return history.transpose(1, 0, 2)

#MAKE TARGET DEVELOPMENTAL PATTERN OUT OF CA RULE
#ONE HOT START
def rule2targets_wrapped_onehot(r, L, N):
  """
  We need 2 flips:

  1) from value order to wolfram order

    | value | wolfram
  N | order | order
  --|-------|---------
  0 |  000  |  111
  1 |  001  |  110
  2 |  010  |  101
  3 |  011  |  100
  4 |  100  |  011
  5 |  101  |  010
  6 |  110  |  001
  7 |  111  |  000

  so we do:
      rule = rule[::-1]

  2) from array order to base order

  array order: np.arange(3) = [0, 1, 2]

  but base2 orders digits left-to-right

  e.g.
  110 = (1        1        0)    [base2]
         *        *        *
   (2^2) 4  (2^1) 2  (2^0) 1
        ---------------------
      =  4 +      2 +      0  = 6 [base10]

  so we do:
    2 ** np.arange(2)[::-1] = [4 2 1]

  """
  base = 2 ** np.arange(3)[::-1]
  rule = np.array([int(v) for v in f"{r:08b}"])[::-1]

  targets = np.zeros((L, N), dtype=np.int32)
  targets[0][int(N/2)] = 1

  for i in range(1, L):
    s = np.pad(targets[i - 1], (1, 1), "wrap")
    s = sliding_window_view(s, 3)
    s = (s * base).sum(axis=1)
    s = rule[s]
    targets[i] = s

  return targets.astype(np.float64)

#ONE HOT WITH MOVEBY
def rule2targets_wrapped_wmoveby(r, moveby, L, N):
  
  base = 2 ** np.arange(3)[::-1]
  rule = np.array([int(v) for v in f"{r:08b}"])[::-1]

  targets = np.zeros((L, N), dtype=np.int32)
  targets[0][int(N/2)+moveby] = 1

  for i in range(1, L):
    s = np.pad(targets[i - 1], (1, 1), "wrap")
    s = sliding_window_view(s, 3)
    s = (s * base).sum(axis=1)
    s = rule[s]
    targets[i] = s

  return targets.astype(np.float64)

#WITH CUSTOM START
def rule2targets_wrapped_wstart(r, L, N, start_pattern):
  base = 2 ** np.arange(3)[::-1]
  rule = np.array([int(v) for v in f"{r:08b}"])[::-1]
  targets = np.zeros((L, N), dtype=np.int32)
  
  targets[0] = start_pattern

  for i in range(1, L):
    s = np.pad(targets[i - 1], (1, 1), "wrap")
    s = sliding_window_view(s, 3)
    s = (s * base).sum(axis=1)
    s = rule[s]
    targets[i] = s

  return targets.astype(np.float64)

#FOR PLOTTING
#------------------------

def show_effectors(states, targets, M, ax):
    preds = np.where(states[:, M:] > 0.5, 1, 0)

    correct = np.where(np.abs(targets - preds) > 0, 1, 0)
    correct_mask = np.ma.array(targets, mask=correct)

    reds = np.dstack(
        [np.ones_like(targets) * 255, np.zeros_like(targets), np.zeros_like(targets)]
    )

    # Create the figure
    ax.imshow(reds, label="errors")  # red background
    ax.imshow(correct_mask)

    # a bit of a mindfuck, correct is used as a mask so it's like the inverse... xD
    error_perc = correct.sum() / targets.size * 100

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=f"Errors:\n{error_perc:.1f}%",
            markerfacecolor="r",
            markersize=10,
        )
    ]

    # print(targets.size())

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.27, 1.0))

    ax.set_title("Effector genes")

def imshow_ca(grid, ax):
    rocket_cmap = sns.color_palette("rocket", as_cmap=True)
    # im = ax.imshow(grid, cmap="magma")
    im = ax.imshow(grid, cmap=rocket_cmap,interpolation="nearest")

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)

    # And a corresponding grid
    ax.grid(which="minor", alpha=0.3)
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    return im

def plot_three_line(ax, rule, data1, data2, data3, season_len=300, legend=False):
    #Plots the fitness over generations for 3 datasets (data1=static 1, data2=static 2, data3=variable)
    
    # Calculate mean and standard error for each list
    mean1 = np.mean(data1, axis=0)
    mean2 = np.mean(data2, axis=0)
    mean3 = np.mean(data3, axis=0)
    stderr1 = np.std(data1, axis=0) / np.sqrt(len(data1))
    stderr2 = np.std(data2, axis=0) / np.sqrt(len(data2))
    stderr3 = np.std(data3, axis=0) / np.sqrt(len(data3))

    for j in range(0, len(mean1), season_len):
        if j % (season_len * 2) == 0:
            ax.axvline(j, linestyle="--", color="gray", alpha=0.3)
        else:
            ax.axvline(j, linestyle=":", color="gray", alpha=0.3)
    
    # Plot data
    ax.plot(mean1, label='Static T1', color='blue')
    ax.tick_params(right=True, labelright=False)
    ax.plot(mean2, label='Static T2', color='orange')
    ax.plot(mean3, label='Variable env', color='red')
    
    # Fill the area between the lines and the error bars
    ax.fill_between(range(len(mean1)), mean1 - stderr1, mean1 + stderr1, color='blue', alpha=0.3)
    ax.fill_between(range(len(mean2)), mean2 - stderr2, mean2 + stderr2, color='orange', alpha=0.3)
    ax.fill_between(range(len(mean3)), mean3 - stderr3, mean3 + stderr3, color='red', alpha=0.3)
    
    ax.set_title("Rule "+str(rule))
    #ax.grid(axis="y")
    #ax.set_ylabel("Fitness")
    #plt.savefig("rule_"+str(rule)+"_lines.png")
    if legend:
        ax.legend(fontsize=14)
        height = 0.62
        base = season_len/2
        kwargs = {"ha":"center", "va":"center", "fontsize":12, "color":"gray"}
        ax.text(base, height, "T1", **kwargs)
        ax.text(base + season_len, height, "T2", **kwargs)
        ax.text(base + season_len*2, height, "T1", **kwargs)
        ax.text(base + season_len*3, height, "T2", **kwargs)

def get_pop_TPF(pop, pop_size, num_cells, grn_size, dev_steps, geneid, rule, seed_int_target, seed_int_dev):
  start_pattern = seedID2string(seed_int_target, num_cells)
  target = rule2targets_wrapped_wstart(int(rule), L=dev_steps+1, N=num_cells, start_pattern=start_pattern)

  start_pattern = seedID2string(seed_int_dev, num_cells)
  start_expression = seed2expression(start_pattern, pop_size, num_cells, grn_size, geneid)
   
  all_phenos = develop(start_expression, pop, dev_steps, pop_size, grn_size, num_cells)
  phenos = all_phenos[:,:,geneid::grn_size]
   
  worst= -num_cells*dev_steps
  prefitnesses = fitness_function_ca(phenos, target)
  fitnesses=1-(prefitnesses/worst) #0-1 scaling

  return target, phenos, fitnesses

def get_pop_TPF_torch(pop, pop_size, num_cells, grn_size, dev_steps, geneid, rule, seed_int):
  start_pattern = seedID2string(seed_int, num_cells)
  start_expression = seed2expression(start_pattern, pop_size, num_cells, grn_size, geneid)
   
  target = rule2targets_wrapped_wstart(int(rule), L=dev_steps+1, N=num_cells, start_pattern=start_pattern)
   
  #all_phenos = develop(start_expression, pop, dev_steps, pop_size, grn_size, num_cells)
  all_phenos = develop_torch(start_expression, pop, dev_steps, pop_size, grn_size, num_cells)
  phenos = all_phenos[:,:,geneid::grn_size]
   
  worst= -num_cells*dev_steps
  prefitnesses = fitness_function_ca(phenos, target)
  fitnesses=1-(prefitnesses/worst) #0-1 scaling

  return target, phenos, fitnesses

def get_fits(rules, seed_ints, metric, root, season_len, num_reps, id_start, extrapolate=True):
    vari_maxs=[np.loadtxt(os.path.expanduser(root+f"variable/stats_{season_len}_{rules[0]}-{rules[1]}_{seed_ints[0]}-{seed_ints[1]}_{i+1+id_start}_{metric}.txt")) for i in range(num_reps)]
    if rules[0] == rules[1]:
        env1_maxs=[np.loadtxt(os.path.expanduser(root+f"static/stats_100000_{rules[0]}_{seed_ints[0]}_{i+1+id_start}_{metric}.txt")) for i in range(num_reps)]
        env2_maxs=[np.loadtxt(os.path.expanduser(root+f"static/stats_100000_{rules[0]}_{seed_ints[1]}_{i+1+id_start}_{metric}.txt")) for i in range(num_reps)]
    else:
        print("scenario not yet implemented")

    if extrapolate:
        diff_len = len(vari_maxs[0]) - len(env1_maxs[0])
        if diff_len > 1:
            env1_maxs=np.array(env1_maxs)
            env2_maxs=np.array(env2_maxs)
            last_elements = env1_maxs[:,-1]
            last_elements=np.tile(last_elements, (diff_len, 1)).T
            env1_maxs = np.hstack((env1_maxs, last_elements))
            last_elements = env2_maxs[:,-1]
            last_elements=np.tile(last_elements, (diff_len, 1)).T
            env2_maxs = np.hstack((env2_maxs, last_elements))

    return vari_maxs, env1_maxs, env2_maxs

def get_fits_dr(rules, seed_int, metric, root_var, root_stat, season_len, num_reps, id_start, extrapolate=True):
    vari_maxs=[np.loadtxt(os.path.expanduser(root_var+f"stats_{season_len}_{rules[0]}-{rules[1]}_{seed_int}-{seed_int}_{i+1+id_start}_{metric}.txt")) for i in range(num_reps)]
    env1_maxs=[np.loadtxt(os.path.expanduser(root_stat+f"static/stats_100000_{rules[0]}_{seed_int}_{i+1+id_start}_{metric}.txt")) for i in range(num_reps)]
    env2_maxs=[np.loadtxt(os.path.expanduser(root_stat+f"static/stats_100000_{rules[1]}_{seed_int}_{i+1+id_start}_{metric}.txt")) for i in range(num_reps)]

    return vari_maxs, env1_maxs, env2_maxs

def get_fits_alt(rules, seed_ints, metric, root, season_len, num_reps, exp_type):
    vari_maxs=[np.loadtxt(os.path.expanduser(root+f"variable/stats_{season_len}_{rules[0]}-{rules[1]}_{seed_ints[0]}-{seed_ints[1]}_{i+1}_{metric}.txt")) for i in range(num_reps)]
    
    static_maxs=[np.loadtxt(os.path.expanduser(root+f"static/stats_100000_{rules[0]}_{149796}_{i+1}_{metric}.txt")) for i in range(num_reps)]
        
    special_maxs=[np.loadtxt(os.path.expanduser(root+f"{exp_type}/stats_{season_len}_{rules[0]}-{rules[1]}_{149796}-{149796}_{i+1}_{metric}.txt")) for i in range(5)]

    return vari_maxs, static_maxs, special_maxs    

def chunker(runs, season_len = 300):
    florp = np.array(runs).mean(axis=0) # average runs
    all_gens = np.arange(0,np.array(runs).shape[1])
    n_seasons = int(np.floor(florp.shape[0]/season_len))
    chunked_seasons = np.array([florp[i*season_len:(i+1)*season_len] for i in range(n_seasons)])
    chunked_gens = np.array([all_gens[i*season_len:(i+1)*season_len] for i in range(n_seasons)])
    assert chunked_seasons.size == season_len * n_seasons #safety check
    chunked_season1, chunked_season2 = chunked_seasons[0::2], chunked_seasons[1::2]
    chunked_season1_g, chunked_season2_g = chunked_gens[0::2].flatten(), chunked_gens[1::2].flatten()
    # Get maximum for each repeat season:
    max_chunked_season1, max_chunked_season2 = chunked_season1.max(axis=1),chunked_season2.max(axis=1)
    # Get maximum among repeat seasons:
    a = max_chunked_season1.max()
    b = max_chunked_season2.max()

    argmax1 = chunked_season1_g[np.argmax(np.array(chunked_season1))]
    argmax2 = chunked_season2_g[np.argmax(np.array(chunked_season2))]
    std1 = np.array(runs)[:,argmax1].std()
    std2 = np.array(runs)[:,argmax2].std()

    return a,b, std1, std2, np.array(runs)[:,argmax1], np.array(runs)[:,argmax2]

    #experimental, ave of maxs, decided against
    #florp = np.array(runs)
    #n_seasons = int(np.floor(florp.shape[1]/season_len))
    #chunked_seasons = np.array([florp[:, i*season_len:(i+1)*season_len] for i in range(n_seasons)])
    #assert (chunked_seasons.shape[0] * chunked_seasons.shape[-1]) == season_len * n_seasons #safety check
    #chunked_season1, chunked_season2 = chunked_seasons[0::2], chunked_seasons[1::2]
    # Get maximum for each replicate in each repeat season:
    #max_chunked_season1, max_chunked_season2 = chunked_season1.max(axis=2),chunked_season2.max(axis=2)
    # Get maximum for each replicate among repeat seasons, then average
    #a = max_chunked_season1.max(axis = 0)
    #b = max_chunked_season2.max(axis = 0)
    #return a,b 

def scatter_value(variable, season1, season2, season_len):
    vari_env1, vari_env2, std1, std2, list1, list2 = chunker(variable, season_len=season_len)
    
    season1 = np.array(season1)
    season2 = np.array(season2)
    M_env1 = season1.mean(axis=0).max()
    M_env2 = season2.mean(axis=0).max()
    static1 = season1[:,np.argmax(season1.mean(axis=0))]
    env1_std = static1.std()
    static2 = season2[:,np.argmax(season2.mean(axis=0))]
    env2_std = static2.std()
    
    cohen_d1 = (vari_env1- M_env1) / np.sqrt((std1+env1_std)/2)
    cohen_d2 = (vari_env2- M_env2) / np.sqrt((std2+env2_std)/2)

    #t_stat1, p_value1 = ttest_ind(list1, static1)
    #t_stat2, p_value2 = ttest_ind(list2, static2)
    t_stat1, p_value1 = mannwhitneyu(list1, static1, alternative='two-sided')
    t_stat2, p_value2 = mannwhitneyu(list2, static2, alternative='two-sided')


    diffs = (vari_env1 - M_env1, vari_env2 - M_env2)
    return diffs, (cohen_d1, cohen_d2), (p_value1,p_value2), (list1, list2, static1, static2)

    #experimental, decided against
    #vari_env1, vari_env2 = chunker(variable, season_len=season_len)
    #M_env1 = np.array(season1).max(axis=1)
    #M_env2 = np.array(season2).max(axis=1)
    #cohen_d1 = (vari_env1.mean()- M_env1.mean()) / np.sqrt((vari_env1.std()+M_env1.std())/2)
    #cohen_d2 = (vari_env2.mean()- M_env2.mean()) / np.sqrt((vari_env2.std()+M_env2.std())/2)
    #CI_1 = [(vari_env1.mean()- M_env1.mean()) + z * np.sqrt(((vari_env1.std()/len(vari_env1))+(M_env1.std()/len(M_env1)))) for z in [1.96, -1.96] ]
    #CI_2 = [(vari_env2.mean()- M_env2.mean()) + z * np.sqrt(((vari_env2.std()/len(vari_env2))+(M_env2.std()/len(M_env2)))) for z in [1.96, -1.96] ]
    #diffs = (vari_env1.mean() - M_env1.mean(), vari_env2.mean() - M_env2.mean())
    #return diffs, cohen_d1, cohen_d2
    

def scatter_value_alt_specfocus(variable, special, season2, season_len):
    vari_env1, vari_env2, std1, std2, list1, list2 = chunker(variable, season_len=season_len)
    M_special = np.array(special).mean(axis=0).max()
    M_env2 = np.array(season2).mean(axis=0).max()
    #diffs = (vari_env2 - M_special, vari_env2 - M_env2)
    #diffs = (M_special - vari_env2, M_special - M_env2)
    diffs = (M_env2 - vari_env2, M_env2 - M_special)
    return diffs

def scatter_value_alt_varifocus(variable, special, season2, season_len):
    vari_env1, vari_env2, std1, std2, list1, list2 = chunker(variable, season_len=season_len)
    M_special = np.array(special).mean(axis=0).max()
    M_env2 = np.array(season2).mean(axis=0).max()
    diffs = (vari_env2 - M_special, vari_env2 - M_env2)
    #diffs = (M_special - vari_env2, M_special - M_env2)
    return diffs

def main_plt(xs, ys, rules, ax):
  ax.scatter(xs, ys, s=40, zorder=3, color="red", edgecolors="black")
  fontsize = 18

  for i, label in enumerate(rules):
      if label == 254:
          ax.annotate(
              label,
              fontsize=fontsize,
              xy=(xs[i], ys[i]),
              xytext=(xs[i] - 0.03, ys[i] + 0.02),
              arrowprops=dict(
                  facecolor="black", shrink=0.05, width=0.2, headwidth=3, headlength=5
              ),
          )
      elif label == 50:
          ax.annotate(
              label,
              fontsize=fontsize,
              xy=(xs[i], ys[i]),
              xytext=(xs[i] + 0.01, ys[i] + 0.02),
              arrowprops=dict(
                  facecolor="black", shrink=0.05, width=0.2, headwidth=3, headlength=5
              ),
          )
      else:
          ax.text(
              xs[i],
              ys[i],
              label,
              fontsize=fontsize,
              ha="right",
              va="bottom",
              color="black",
          )

  #ax.set_xlim(-0.06, 0.12)
  #ax.set_ylim(-0.06, 0.12)
  #ax.set_xlim(-0.01, 0.01)
  #ax.set_ylim(-0.01, 0.01)
  #plt.gca().set_aspect('equal', adjustable='box')
  circle = Circle((0, 0), 0.01, color='blue', fill=True, linewidth=0, alpha = 0.2)
  plt.gca().add_patch(circle)
  ax.axvline(0, lw=1, color="black")
  ax.axhline(0, lw=1, color="black")
  ax.set_xlabel("Max fit of variable - Max fit of static T1",fontsize=22)
  ax.set_ylabel("Max fit of variable - Max fit of static T2",fontsize=22)
  ax.grid(zorder=0)

def chunker_plotting(run, season_len = 300):
    gens=list(range(len(run)))
    n_seasons = int(np.floor(run.shape[0]/season_len))
    chunked_seasons = np.array([run[i*300:(i+1)*300] for i in range(n_seasons)])
    chunked_gens = np.array([gens[i*300:(i+1)*300] for i in range(n_seasons)])

    assert chunked_seasons.size == season_len * n_seasons #safety check

    chunked_season1, chunked_season2 = chunked_seasons[0::2], chunked_seasons[1::2]
    chunked_gens1, chunked_gens2 = chunked_gens[0::2], chunked_gens[1::2]
    
    return chunked_season1, chunked_season2, chunked_gens1, chunked_gens2

def try_grn(variable, rule, run_seedints, try_seedints, grn_size, geneid, root, num_cells, dev_steps):
    last_grns=[]
    for i in range(15):
        if variable:
            filename = os.path.expanduser(f"{root}/variable/stats_300_{rule}-{rule}_{run_seedints[0]}-{run_seedints[1]}_{i+1}" + "_best_grn.txt")
        else:
            filename = os.path.expanduser(f"{root}/static/stats_100000_{rule}_{run_seedints}_{i+1}" + "_best_grn.txt")
        grns = np.loadtxt(filename)
        num_grns = int(grns.shape[0]/(grn_size+2)/grn_size)
        grns = grns.reshape(num_grns,grn_size+2,grn_size)
        grn = grns[-1,:,:]
        last_grns.append(grn)
    last_grns = np.array(last_grns)

    last_phenos=[]
    fits = []
    for s in try_seedints:
        targets, phenos, fitnesses = get_pop_TPF(last_grns, len(last_grns), num_cells, grn_size, dev_steps, geneid, rule, s,s)
        last_phenos.append(phenos)
        fits.append(fitnesses)
    last_phenos = np.array(last_phenos)
    fits = np.array(fits)
    return last_phenos, fits, last_grns

def make_restricted_plot(all_targs, num_cells, dev_steps, dot_xs, dot_ys, labelled=True):
    
    worst= -num_cells*(dev_steps+1)
    oritargs = np.array([all_targs[0],all_targs[1]])

    where_overlap = np.where(all_targs[0]==all_targs[1])
    where_no_overlap = np.where(all_targs[0]!=all_targs[1])

    bestgen=all_targs[0].copy()
    bestgen[where_no_overlap] = 0.5
    bestgen = np.expand_dims(bestgen, axis=0)

    half= int(len(where_no_overlap[0])/2)

    a = all_targs[0].copy()
    a[tuple(idx[:half] for idx in where_no_overlap)] = 0.5
    a = np.expand_dims(a, axis=0)

    b = all_targs[1].copy()
    b[tuple(idx[:half] for idx in where_no_overlap)] = 0.5
    b = np.expand_dims(b, axis=0)

    inperfa = 1 - all_targs[0].copy()
    inperfa = np.expand_dims(inperfa, axis=0)
    inperfb = 1 - all_targs[1].copy()
    inperfb = np.expand_dims(inperfb, axis=0)

    worstgen=inperfa[0].copy()
    worstgen[where_no_overlap] = 0.5
    worstgen = np.expand_dims(worstgen, axis=0)

    c= all_targs[0].copy()
    c[where_overlap] = 0.5
    c = np.expand_dims(c, axis=0)

    d= all_targs[1].copy()
    d[where_overlap] = 0.5
    d = np.expand_dims(d, axis=0)

    labels = ["A", "B", "Overlap good, rest 0.5", "Overlap good, rest/2 0.5, A", "Overlap good, rest/2 0.5, B", "A inverse","B inverse", "Overlap inverse, rest 0.5"]
    labels.append("A but overlap 0.5")
    labels.append("B but overlap 0.5")

    pop = np.concatenate((oritargs, bestgen,a,b,inperfa,inperfb,worstgen,c,d), axis=0) #0,1, 4,5

    fitnesses1 = -np.abs(pop - all_targs[0]).sum(axis=1).sum(axis=1)
    fitnesses1=1-(fitnesses1/worst) #0-1 scaling
    fitnesses2 = -np.abs(pop - all_targs[1]).sum(axis=1).sum(axis=1)
    fitnesses2=1-(fitnesses2/worst) #0-1 scaling

    pop_df = pd.DataFrame()
    pop_df["x"]=fitnesses1
    pop_df["y"]=fitnesses2
    xs=fitnesses1
    ys=fitnesses2

    if labelled:
        labels = list(zip(fitnesses1,fitnesses2))
        plt.scatter(pop_df["x"], pop_df["y"])
        for i, label in enumerate(labels): 
            plt.text(
                xs[i],
                ys[i],
                label,
                ha="center",
                va="bottom",
                color="black",
            )

    plt.scatter(dot_xs, dot_ys)
    sns.set_style("whitegrid")

    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)

    myhatch='..'
    mycolor="C0"
    triangle = Polygon([[1,1], pop_df.iloc[0], pop_df.iloc[1]], closed=True, alpha=0.5,edgecolor=mycolor, facecolor='none',hatch=myhatch)
    plt.gca().add_patch(triangle)
    triangle = Polygon([[0,0], pop_df.iloc[5], pop_df.iloc[6]], closed=True, alpha=0.5,edgecolor=mycolor, facecolor='none',hatch=myhatch)
    plt.gca().add_patch(triangle)
    triangle = Polygon([[0,1], pop_df.iloc[1], pop_df.iloc[5]], closed=True, alpha=0.5,edgecolor=mycolor, facecolor='none',hatch=myhatch)
    plt.gca().add_patch(triangle)
    triangle = Polygon([[1,0], pop_df.iloc[0], pop_df.iloc[6]], closed=True, alpha=0.5,edgecolor=mycolor, facecolor='none',hatch=myhatch)
    plt.gca().add_patch(triangle)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return pop_df

#coloring
def generate_colors(n):
    # Generate colors in HSL
    colors = []
    for i in range(n):
        # Generate hue, saturation, and lightness
        hue = i / n  # Normalize hue
        saturation = 0.7  # Set saturation to 70%
        lightness = 0.5  # Set lightness to 50%
        
        # Convert HSL to RGB
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors

def make_network(num_gens_show, pop_size, edges):
    # Set up network
    num_rows = num_gens_show+1
    num_columns = pop_size
    G = nx.Graph()
    # Add nodes with specified positions
    pos = {}
    for i in range(num_rows):
        for j in range(num_columns):
            node = (i, j)
            G.add_node(node)
            pos[node] = (j, -i)  # Assigning positions based on rows and columns

    # Add edges from the edges variable
    mydic=defaultdict(list) #make dictionary to keep track of OG where it comes from
    for i in range(len(edges)):
        G.add_edge(edges[i][0],edges[i][1])
        if edges[i][0][0] == 0: #if it is the first generation
            mydic[edges[i][0]].append(edges[i][1])
        else:
            for k in mydic.keys():
                if edges[i][0] in mydic[k]:
                    mydic[k].append(edges[i][1])

    #colors = generate_colors(pop_size)
    colors = list(range(pop_size))
    node_colors = []
    for c in colors:
        node_colors.append(c) #colors for the first generation
    color_dic = {}
    for idx, node in enumerate(G.nodes()):
        if node[0] == 0:
            color_dic[node] = colors[idx] #color assigned to each original parent
    for node in G.nodes():
        if node[0] != 0:
            for k in mydic.keys():
                if node in mydic[k]:
                    node_colors.append(color_dic[k]) #assign color based on original parent
    
    return G, pos, node_colors

'''
TESTING GENERALIST FUNCTION
import math
bestgen = landmarks.iloc[2][0]
p1=[bestgen, bestgen]
p2=[0.488142, 1.000000]
midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

distance=helper.calculate_distance(p1[0], p1[1], midpoint[0], midpoint[1])
new_x = p1[0] - distance / math.sqrt(2)
new_y = p1[1] - distance / math.sqrt(2)
distance=helper.calculate_distance(p1[0], p1[1], new_x, new_y)

plt.xlim(0,1)
plt.ylim(0,1)
plt.scatter(p1[0],p1[1], color="green")
plt.scatter(midpoint[0],midpoint[1], color="orange")
plt.scatter(new_x,new_y,color="red")
plt.scatter(new_x, p1[1], color="blue")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#NOTE: I want a generalist calculator where the midpoint (yellow) has a pretty bad score,
# the red one on the diagonal has a better score (eventhough same distance from green)
# and the blue has exactly and inbetween score
# so we can have a distance from green + distance from line, balanced

orange = calc_conz_BH(midpoint[0], midpoint[1], landmarks)
green = calc_conz_BH(p1[0],p1[1], landmarks)
red = calc_conz_BH(new_x,new_y, landmarks)
blue = calc_conz_BH(new_x, p1[1], landmarks)

labels=[]
i_s = []
j_s = []
fig = plt.figure(figsize=(10,10))
for i in np.arange(0,1,0.1):
    for j in np.arange(0,1,0.1):
        one, two = calc_conz_BH(i, j, landmarks)
        if (one > 0) & (two > 0):
            i_s.append(i)
            j_s.append(j)
            labels.append((round(one, 2),round(two, 2)))
            plt.scatter(i,j, color="blue")

#plt.scatter(pop_df["x"], pop_df["y"])
for i, label in enumerate(labels): 
    plt.text(
        i_s[i],
        j_s[i],
        label,
        ha="center",
        va="bottom",
        color="black",
    )
plt.xlim(0,1)
plt.ylim(0,1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
'''

'''
Renaming
for file in stats_*; do mv "$file" "${file/env_seeded_0_/}"; done

for file in stats_300_*_*_*; do mv "$file" "$(echo "$file" | sed -E 's/(stats_[0-9]+)_([0-9]+)_/\1_\2-\2_/')"; done

for file in *0-0*.txt; do mv "$file" "${file//0-0/102-102}"; done

for file in *1-1*.txt; do mv "$file" "${file//1-1/150-150}"; done

for file in *2-2*.txt; do mv "$file" "${file//2-2/90-90}"; done

for file in *0.5*.txt; do mv "$file" "${file//0.5/69904-149796}"; done

for file in stats_300_*-*_5_[0-9]*.txt; do mv "$file" "$(echo "$file" | sed -E 's/_5_([0-9])/_69904-149796_\1/')"; done

for file in stats_100000_*_5_[0-9]*.txt; do mv "$file" "$(echo "$file" | sed -E 's/_5_([0-9])/_69904_\1/')"; done

for file in stats_100000_*_6_[0-9]*.txt; do mv "$file" "$(echo "$file" | sed -E 's/_6_([0-9])/_149796_\1/')"; done

#-----

for file in *_600_*.txt; do mv "$file" "${file//_600_/_100000_}"; done

for file in stats_0_*.txt; do mv "$file" "${file//stats_0_/stats_100000_}"; done
'''