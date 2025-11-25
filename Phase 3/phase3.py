import numpy as np
from random import uniform
from math import sin, pi
from functools import partial
from phase2 import powells_conjugate_direction_method

hashmap = {}
fneval = 0

def reset_globals():
    global hashmap, fneval
    hashmap = {}
    fneval = 0

#Problem definition functions
def f1(x):
    return (x[0] - 10)**3 + (x[1] - 20)**3

def g1(x):
    return [
        (x[0] - 5)**2 + (x[1] - 5)**2 - 100,  # g1 >= 0
        (x[0] - 6)**2 + (x[1] - 5)**2 - 82.81 # g2 <= 0
    ]

def constraint_types1():
    return ['ge', 'le']

def bounds1():
    return [(13, 20), (0, 4)]


def f2(x):
    num = ((sin(2 * pi * x[0]))**3) * sin(2 * pi * x[1])
    den = (x[0]**3) * (x[0] + x[1])
    if abs(den) < 1e-12: 
        return -float('inf') 
    return - (num / den) #chaning to minimization problem

def g2(x):
    return [
        x[0]**2 - x[1] + 1, # g1 <= 0
        1 - x[0] + (x[1] - 4)**2 # g2 <= 0
    ]

def constraint_types2():
    return ['le', 'le']

def bounds2():
    return [(0, 10), (0, 10)]

def f3(x):
    return x[0] + x[1] + x[2] 

def g3(x):
    return [
        -1 + 0.0025 * (x[3] + x[5]), # g1 <= 0
        -1 + 0.0025 * (-x[3] + x[4] + x[6]), # g2 <= 0
        -1 + 0.01 * (-x[5] + x[7]), # g3 <= 0
        100 * x[0] - x[0] * x[5] + 833.33252 * x[3] - 83333.333, # g4 <= 0
        x[1] * x[3] - x[1] * x[6] - 1250 * x[3] + 1250 * x[4], # g5 <= 0
        x[2] * x[4] - x[2] * x[7] - 2500 * x[4] + 1250000 # g6 <= 0
    ]

def constraint_types3():
    return ['le', 'le', 'le', 'le', 'le', 'le']

def bounds3():
    return [
        (100, 10000),    
        (1000, 10000),   
        (1000, 10000),   
        (10, 1000),      
        (10, 1000),      
        (10, 1000),      
        (10, 1000),      
        (10, 1000)       
    ]


def evaluate_penalized(varArr, f_unconstrained, g_constraints, constraint_types, R):
    global hashmap, fneval
    var_key = (str(varArr), R)
    if var_key in hashmap:
        return hashmap[var_key]
    
    fneval += 1
    f_x = f_unconstrained(varArr)
    penalty = 0
    g_values = g_constraints(varArr)
    
    for g_val, g_type in zip(g_values, constraint_types):
        if g_type == 'le': # g(x) <= 0
            violation = max(0, g_val)
        elif g_type == 'ge': # g(x) >= 0
            violation = max(0, -g_val)
        else: # h(x) = 0 (equality)
            violation = 0
        penalty += violation**2
            
    total = f_x + R * penalty
    
    hashmap[var_key] = total
    return total


def solve_constrained_problem(f_unconstrained, g_constraints, constraint_types, bounds, d, initial_guess,
                              R_init=1.0, C=10.0, epsilon=1e-6, 
                              powell_acc=1e-6, powell_linDepAc=5):
    global fneval
    R = R_init           
    max_outer_iter = 10 # Max iterations for penalty update
    
    x_current = np.array(initial_guess)
    
    # Extract box bounds (lower, upper)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    total_evals_so_far = 0
    
    print(f"  Starting Run... Initial Guess: {np.array2string(x_current, formatter={'float_kind':lambda val: f'{val:8.4f}'})}")

    for k in range(max_outer_iter):
        # Penalized function
        penalized_func = partial(evaluate_penalized, 
                                 f_unconstrained=f_unconstrained, 
                                 g_constraints=g_constraints, 
                                 constraint_types=constraint_types, 
                                 R=R)
        
    
        reset_globals()
        
        #unconstrained solver
        x_opt, f_penalized_opt = powells_conjugate_direction_method(
            func_obj=penalized_func,
            d=d,
            initial_guess=x_current,
            a=lower_bounds,
            b=upper_bounds,
            max_iter=1000, 
            acc=powell_acc,      
            linDepAc=powell_linDepAc
        )
        
        # Store convergence data
        f_unconstrained_opt = f_unconstrained(x_opt)
        total_evals_so_far += fneval

        #Checking for the convergence
        p_new = penalized_func(x_opt)
        if k!=0 and abs(p_new-p_prev)<epsilon:
            print(f"  Run Converged. Total Penalty {p_new:.2e} < {epsilon:.2e}")
            break
        else:
            p_prev = p_new
        print(f"    Outer Iter {k+1}: R={R:8.1e}, f(x)={f_unconstrained_opt:12.6e}")
        
        R = C * R
        x_current = x_opt 
    
    if k == max_outer_iter - 1:
        print("  Run stopped: Max outer iterations reached.")
        
    final_f_opt = f_unconstrained(x_opt)
    return x_opt, final_f_opt, total_evals_so_far

def get_random_initial_guess(bounds):
    return np.array([uniform(b[0], b[1]) for b in bounds])

def main():
    R_init=1.0 
    C=10.0 
    epsilon=1e-6 
    powell_acc=1e-6 
    powell_linDepAc=5
    # Store problem definitions in a list
    problems = [
        {'index': 1, 'd': 2, 'f': f1, 'g': g1, 'types': constraint_types1, 'bounds': bounds1, 'minima': (14.095, 0.84296)},
        {'index': 2, 'd': 2, 'f': f2, 'g': g2, 'types': constraint_types2, 'bounds': bounds2, 'minima': (1.227, 4.245)},
        # {'index': 3, 'd': 8, 'f': f3, 'g': g3, 'types': constraint_types3, 'bounds': bounds3, 'minima': None},
    ]
    #output file
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write("Project Phase 3 Results: Constrained Optimization\n")
        f.write("Method: Bracket-Operator Penalty Method\n")
        f.write("Unconstrained Solver: Powell's Conjugate Direction Method (from Phase 2)\n")
        f.write("\n--- Parameter Settings ---\n")
        f.write(f"R_init: {R_init}\n")
        f.write(f"C: {C}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Powell Accuracy: {powell_acc}\n")
        f.write(f"Powell LinDepAc: {powell_linDepAc}\n")
        f.write("=" * 80 + "\n\n")

        for prob in problems:
            prob_index = prob['index']
            d = prob['d']
            f_func = prob['f']
            g_func = prob['g']
            types_func = prob['types']
            bounds_func = prob['bounds']
            
            bounds = bounds_func()
            
            print(f"\n--- Solving Problem {prob_index} ---")
            f.write(f"--- Solving Problem {prob_index} ---\n")
            f.write(f"Number of Variables (d): {d}\n")
            f.write(f"Box Bounds: {bounds}\n")
            
            num_runs = 10
            f.write(f"Num of test runs = {num_runs}\n")
            f.write(f"{'Run':<5}{'Initial Guess(x0)':<70}{'Optimal Point (x*)':<85}{'f(x*)':<20}{'nEval':<10}\n")
            f.write("-" * 200)
            f.write("\n")

            results_f_opt = []
            
            for run in range(num_runs):
                print(f"Starting Run {run+1}/{num_runs} for Problem {prob_index}...")
                initial_guess = get_random_initial_guess(bounds)
            
                x_opt, f_opt, n_eval = solve_constrained_problem(
                    f_unconstrained=f_func,
                    g_constraints=g_func,
                    constraint_types=types_func(),
                    bounds=bounds,
                    d=d,
                    initial_guess=initial_guess,
                    R_init=R_init,
                    C=C,
                    epsilon=epsilon,
                    powell_acc=powell_acc,
                    powell_linDepAc=powell_linDepAc
                )
                
                results_f_opt.append(f_opt)

                x_opt_str = np.array2string(x_opt, formatter={'float_kind':lambda val: f"{val:10.6f}"})
                initial_guess_str = np.array2string(initial_guess, formatter={'float_kind':lambda val: f"{val:10.6f}"})
                f.write(f"{run+1:<5}{initial_guess_str:<70}{x_opt_str:<85}{f_opt:<20.6e}{n_eval:<10}\n")
            
            f_best = np.min(results_f_opt)
            f_worst = np.max(results_f_opt)
            f_mean = np.mean(results_f_opt)
            f_median = np.median(results_f_opt)
            f_std_dev = np.std(results_f_opt)

            f.write("-" * 200)
            f.write(f"\nStatistics for Problem {prob_index} over {num_runs} runs:\n")
            f.write(f"  Best f(x*):   {f_best:20.6e}\n")
            f.write(f"  Worst f(x*):  {f_worst:20.6e}\n")
            f.write(f"  Mean f(x*):   {f_mean:20.6e}\n")
            f.write(f"  Median f(x*): {f_median:20.6e}\n")
            f.write(f"  Std Dev:      {f_std_dev:20.6e}\n")
            f.write("\n\n")

                
main()