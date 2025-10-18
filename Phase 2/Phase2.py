import numpy as np
from random import uniform
from math import sqrt, degrees
from functools import partial
import matplotlib.pyplot as plt
from Phase1Methods import bounding_phase_method, golden_section_method

hashmap = {}
fneval = 0
func_name = {
        1:"Sum of Squares",
        2:"Rosenbrock",
        3:"Dixon-Price",
        4:"Trid",
        5:"Zakharov",
}

def reset_globals():
    global hashmap, fneval
    hashmap = {}
    fneval = 0

def evaluate_objective(varArr, func_index):
    global hashmap, fneval
    var_key = str(varArr)
    if var_key in hashmap:
        return hashmap[var_key]
    fneval += 1
    d = len(varArr)
    total = 0

    if func_index == 1:  #Sum of Squares Function
        for i, var in enumerate(varArr): 
            total += (i + 1) * var * var
    elif func_index == 2:  # rosenbrock Function
        for i in range(d - 1): 
            total += 100 * (varArr[i+1] - varArr[i]**2)**2 + (varArr[i] - 1)**2
    elif func_index == 3:  # Dixon-price Function
        total = (varArr[0] - 1)**2
        for i in range(1, d): 
            total += (i + 1) * (2 * varArr[i]**2 - varArr[i-1])**2
    elif func_index == 4:  #Trid Function
        sum1 = (varArr[0]-1)**2
        sum2 = 0
        for i in range(1,d):
            sum1 += (varArr[i] - 1)**2
            sum2 += varArr[i] * varArr[i-1]
        total = sum1 - sum2
    elif func_index == 5:  #Zakharov Function
        sum1,sum2 = 0,0
        for i, var in enumerate(varArr):
            sum1 += var**2
            sum2 += 0.5 * (i + 1) * var 
        total = sum1 + sum2**2 + sum2**4


    hashmap[var_key] = total
    return total

def magnitude(v):
    total = 0
    for x in v:
        total+=x*x
    return sqrt(total)

def normalize(v):
    mag = magnitude(v)
    if mag == 0: return v
    return v / mag

def calculate_alpha_range(x_current, direction, lower_bounds, upper_bounds):
    alpha_min_final = -float('inf') #negative infinity
    alpha_max_final = float('inf') #positive infinity
    for i in range(len(x_current)):
        d_i = direction[i]
        if abs(d_i) < 1e-12: continue #this is to check if the component of direction is not zero(or very small)
        t_lower = (lower_bounds[i] - x_current[i]) / d_i 
        t_upper = (upper_bounds[i] - x_current[i]) / d_i
        alpha_min_i = min(t_lower, t_upper)
        alpha_max_i = max(t_lower, t_upper)
        #taking intersection of current interval with all the previous intervals
        if alpha_min_i > alpha_min_final: 
            alpha_min_final = alpha_min_i 
        if alpha_max_i < alpha_max_final: 
            alpha_max_final = alpha_max_i
    if alpha_min_final > alpha_max_final: #not valid interval, no alpha
        return 0,0
    return alpha_min_final, alpha_max_final


def unidirectional_search(x, direction, func, a,b):
    #defining the function for easier execution of boundign phase and golden section search method
    line_func = lambda alpha: func(x + alpha * direction)
    
    lower_box = np.full_like(x, a) #this makes a constant vector will all components as "a"
    upper_box = np.full_like(x, b)
    safe_min_alpha, safe_max_alpha = calculate_alpha_range(x, direction, lower_box, upper_box)
    #if no valid alpha range then return the current point without unidirectional search
    if safe_min_alpha >= safe_max_alpha:
        return x

    bracket_min_alpha, bracket_max_alpha = bounding_phase_method(line_func, safe_min_alpha, safe_max_alpha)

    #taking intersection just in case bounding phase gives an interval outside of safe interval
    final_min_alpha = max(safe_min_alpha, bracket_min_alpha)
    final_max_alpha = min(safe_max_alpha, bracket_max_alpha)
    if final_min_alpha >= final_max_alpha:
        return x
    
    final_min_alpha, final_max_alpha = golden_section_method(line_func, final_min_alpha, final_max_alpha)
    alpha_star = (final_min_alpha+final_max_alpha)/2
    return x + alpha_star * direction

def plot_search_path(path, func_index, a,b,d):
    func_obj = partial(evaluate_objective, func_index=func_index)
    bounds = (a,b)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a grid to evaluate the function for the contour plot
    x_range = np.linspace(bounds[0], bounds[1], 300)
    y_range = np.linspace(bounds[0], bounds[1], 300)
    X, Y = np.meshgrid(x_range, y_range)
    # Evaluate the function on the grid
    Z = np.array([func_obj(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    # Plot the contour lines
    levels = np.logspace(0, 5, 15) # Logarithmic levels for better visualization
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8, fmt='%d')
    path_points = np.array(path)
    if d == 2:
        ax.plot(path_points[:, 0], path_points[:, 1], 'r-o', markersize=6, linewidth=1.5, label='Search Path')
    elif d == 1:
        ax.plot(path_points[:,0], np.full_like(path_points[:,0],0), 'r-o', markersize=6, linewidth=1.5, label='Search Path')
    
    # Add text labels for each iteration number
    for i, point in enumerate(path):
        if d == 2:
            ax.text(point[0] + 0.3, point[1] + 0.3, str(i+1), fontsize=10, color='black', ha='center', va='center')
        elif d ==1:
            ax.text(point[0] + 0.3,0.3, str(i+1), fontsize=10, color='black', ha='center', va='center')


    ax.set_title(f"Powell's Method Search Path on {func_name[func_index]} for d={d}", fontsize=16, pad=20)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.show()


def powells_conjugate_direction_method(func_obj, d, a,b, max_iter=1000, acc=1e-6,linDepAc = 5):
    x = np.array([uniform(a,b) for i in range(d)])
    initial_guess = x
    dir_matrix = np.identity(d)
    x_values = np.array([])
    x_values = np.append(x_values, x)
    for i in range(max_iter):
        for j in range(d+1):
            x = unidirectional_search(x, dir_matrix[j%d], func_obj, a,b)
            if  j == 0:
                x_start = x
        x_values = np.vstack([x_values, x])
        new_dir = x - x_start

        #Cecking if the magnitude of new direction is less than the accuracy
        norm_new_dir = magnitude(new_dir)
        if norm_new_dir < acc:
            return x, func_obj(x), fneval, initial_guess,x_values
            
        #checking for linear dependency of new direction with the other directions
        reset_needed = False
        for j, retained_dir in enumerate(dir_matrix[1:]):
            if norm_new_dir > 1e-9:
                cos_angle = np.dot(new_dir, retained_dir) / (norm_new_dir * magnitude(retained_dir))
                angle_deg = degrees(np.clip(np.arccos(cos_angle),-1,1))
                if angle_deg < linDepAc:#5 deg
                    reset_needed = True
                    break
        
        if reset_needed:
            dir_matrix = np.identity(d)
        else:
            dir_matrix = np.vstack([ normalize(new_dir),dir_matrix[0:d-1]])
    return x, func_obj(x), fneval, initial_guess,x_values

def main():
    with open("Phase 2/input.txt", "r") as f:
        lines = f.readlines()
        func_index = int(lines[0].strip())
        d = int(lines[1].strip())
        a, b = map(float, lines[2].strip().split(','))
    
    f = open(r"output.txt", "w", encoding="utf-8")
    f.write(f"Objective Function Index: {func_index}\n")
    f.write(f"Number of Variables (d): {d}\n")
    f.write(f"Limits: [{a}, {b}]\n")
    num_runs = 1
    f.write(f"Num of test runs = {num_runs} \n")
    f.write(f"{'Run':<5}{'Initial Guess(x0)':<70}{'Optimal Point (x*)':<70}{'f(x*)':<20}{'nEval':<10}\n")
    f.write("-" * 180)
    f.write("\n")
    path = None
    objective_function_with_index = partial(evaluate_objective, func_index=func_index)
    for run in range(num_runs):
        reset_globals()
        x_opt, f_opt, n_eval, initial_guess, path = powells_conjugate_direction_method(func_obj=objective_function_with_index, d=d, a=a, b=b)
        x_opt_str = np.array2string(x_opt, formatter={'float_kind':lambda val: f"{val:8.4f}"})
        initial_guess_str = np.array2string(initial_guess, formatter={'float_kind':lambda val: f"{val:8.4f}"})
        f.write(f"{run+1:<5}{initial_guess_str:<70}{x_opt_str:<70}{f_opt:<20.6e}{n_eval:<10}\n")
    f.close()

    #plot the last iteration search path
    #This is only supported if d <= 2
    if d <= 2:
        plot_search_path(path,func_index,a,b,d)

main()