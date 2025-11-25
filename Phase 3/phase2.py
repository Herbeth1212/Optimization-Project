import numpy as np
from math import sqrt, degrees
from random import uniform

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
        
        if abs(d_i) < 1e-12: # checking if d_i is not zero or very small
            continue
            
        t_lower = (lower_bounds[i] - x_current[i]) / d_i 
        t_upper = (upper_bounds[i] - x_current[i]) / d_i
        
        alpha_min_i = min(t_lower, t_upper)
        alpha_max_i = max(t_lower, t_upper)
        
        #taking intersection of current interval with all the previous intervals
        if alpha_min_i > alpha_min_final: 
            alpha_min_final = alpha_min_i 
        if alpha_max_i < alpha_max_final: 
            alpha_max_final = alpha_max_i
            
    if alpha_min_final > alpha_max_final: #not valid interval
        return 0, -1
        
    return alpha_min_final, alpha_max_final


def bounding_phase_method(func, a, b, alpha0= None, delta=0.5):

    if alpha0 == None:
        alpha0 = uniform(a,b)
    elif alpha0 < a or alpha0 > b:
        alpha0 = uniform(a,b)
      
    while True:
        f0 = func(alpha0)
        f_minus = func(alpha0 - abs(delta))
        f_plus = func(alpha0 + abs(delta))
        
        if f_minus >= f0 >= f_plus: 
            delta = abs(delta)
            break
        elif f_minus <= f0 <= f_plus: 
            delta = -abs(delta)
            break
        else: 
            if f_minus < f0 and f_plus < f0:
                alpha0 = uniform(a,b) 
                continue
            return tuple(sorted((alpha0 - abs(delta), alpha0 + abs(delta))))

    k, alpha_current = 0, alpha0
    max_k = 30 # Prevent infinite loop and large values of k
    while k < max_k:
        alpha_prev = alpha_current
        alpha_current = alpha_prev + (2**k) * delta
        
        # Clamp to bounds
        if alpha_current < a: alpha_current = a
        if alpha_current > b: alpha_current = b

        if func(alpha_current) >= func(alpha_prev):
            return tuple(sorted((alpha_prev - (2**(k-1))*delta, alpha_current)))
        
        if alpha_current == a or alpha_current == b:
            # Reached the edge of the allowed range
            return tuple(sorted((alpha_prev, alpha_current)))
            
        k += 1
        
    # Fallback max_k is reached
    return tuple(sorted((a,b)))


def golden_section_method(func, a, b, accuracy=1e-5):
    golden_ratio = (sqrt(5) - 1) / 2
    alpha1 = b - golden_ratio * (b - a)
    alpha2 = a + golden_ratio * (b - a)
    f1, f2 = func(alpha1), func(alpha2)
    
    while (b - a) > accuracy:
        if f1 < f2:
            b, alpha2, f2 = alpha2, alpha1, f1
            alpha1 = b - golden_ratio * (b - a)
            f1 = func(alpha1)
        else:
            a, alpha1, f1 = alpha1, alpha2, f2
            alpha2 = a + golden_ratio * (b - a)
            f2 = func(alpha2)
    return a,b

def unidirectional_search(x, direction, func, lower_bounds, upper_bounds):
    
    #defining the function for easier execution of boundign phase and golden section search method
    line_func = lambda alpha: func(x + alpha * direction)
    
    #Get feasible alpha range from box constraints
    safe_min_alpha, safe_max_alpha = calculate_alpha_range(x, direction, lower_bounds, upper_bounds)

    #if no valid alpha range then return the current point without unidirectional search
    if safe_min_alpha > safe_max_alpha or abs(safe_max_alpha - safe_min_alpha) < 1e-12:
        return x

    # Find a bracket using the bounding phase 
    bracket_min_alpha, bracket_max_alpha = bounding_phase_method(line_func, safe_min_alpha, safe_max_alpha)
   

    # Intersection with the overall interval
    final_min_alpha = max(safe_min_alpha, bracket_min_alpha)
    final_max_alpha = min(safe_max_alpha, bracket_max_alpha)

    if final_min_alpha >= final_max_alpha:
        # Bracket is invalid
        f_min = line_func(safe_min_alpha)
        f_max = line_func(safe_max_alpha)
        if f_min < f_max:
            return x + safe_min_alpha * direction
        else:
            return x + safe_max_alpha * direction
    
    # using golden section searchmethod
    final_min_alpha, final_max_alpha = golden_section_method(line_func, final_min_alpha, final_max_alpha)
    alpha_star = (final_min_alpha + final_max_alpha) / 2
    
    x_new = x + alpha_star * direction
    
    return x_new

def powells_conjugate_direction_method(func_obj, d,initial_guess, a,b, max_iter=1000, acc=1e-6,linDepAc = 5):
    x = np.copy(initial_guess)
    dir_matrix = np.identity(d)
    for i in range(max_iter):
        for j in range(d+1):
            x = unidirectional_search(x, dir_matrix[j%d], func_obj, a,b)
            if  j == 0:
                x_start = x
        new_dir = x - x_start

        #Cecking if the magnitude of new direction is less than the accuracy
        norm_new_dir = magnitude(new_dir)
        if norm_new_dir < acc:
            return x, func_obj(x)
            
        #checking for linear dependency of new direction with the other directions
        reset_needed = False
        for j, retained_dir in enumerate(dir_matrix[1:]):
            if norm_new_dir > 1e-9:
                cos_angle = np.dot(new_dir, retained_dir) / (norm_new_dir * magnitude(retained_dir))
                angle_deg = degrees(np.arccos(np.clip(cos_angle,-1,1)))
                if angle_deg < linDepAc:#5 deg
                    reset_needed = True
                    break
        
        if reset_needed:
            dir_matrix = np.identity(d)
        else:
            dir_matrix = np.vstack([ normalize(new_dir),dir_matrix[0:d-1]])
    return x, func_obj(x)
