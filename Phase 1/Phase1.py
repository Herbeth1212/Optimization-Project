from math import sin, exp, sqrt
import random
from functools import partial
# import matplotlib.pyplot as plt
# import numpy as np

FUNCT_INDEX = 6
fneval = 0
hash_map = {}

def objective_function(x:float):
    global hash_map, fneval
    if x in hash_map:
        return hash_map[x]
    fneval += 1
    match FUNCT_INDEX:
        case 1:
            hash_map[x] = -(pow(2*x - 5,4) - pow(x*x -1,3))
        case 2:
            hash_map[x] = -(8 + pow(x,3) - 2*x - 2*exp(x))
        case 3:
            hash_map[x] = -(4*x*sin(x))
        case 4:
            hash_map[x] = (2*pow(x-3,2) + exp(0.5*pow(x,2)))
        case 5:
            hash_map[x] = x*x - 10*exp(0.1*x)
        case 6:
            hash_map[x] = -(20*sin(x) - 15*x*x)
        case 7:
            hash_map[x] =  x*x + 54/x
    # print(x, hash_map[x])
    return hash_map[x]

def give_interval_in_correct_order(x1,x2):
    if(x1<x2):
        return x1,x2
    else:
        return x2,x1
    

def normalized_function(lowerLimit:float, upperLimit:float, x:float):
    return objective_function((upperLimit-lowerLimit)*x + lowerLimit)

def bounding_phase_method(lowerLimit:float,upperLimit:float,delta:float, initialGuess = None):
    global fneval
    a, b= lowerLimit, upperLimit
    x_old = None
    f_old = None
    fneval = 0
    function_values = []
    x_values = []

    f = open(r'BoundingPhaseiteration.txt', "w",encoding="utf-8")
    f.write("\n**\n")
    f.write("Bounding Phase Method\n\n")

    #Step 1
    x_mid = initialGuess
    k = 0
    if x_mid == None: #if the initial guess is not given then choose a random initial guess
        x_mid = random.uniform(a,b)
        f.write(f"Initial guess randomly choosen x0 = {x_mid}")
    else:
        f.write(f"Proceeding with initial guess x0 = {x_mid}")

    #Step 2: Deciding the sign of delta
    while True:
        x_ub, x_lb = x_mid + abs(delta),x_mid - abs(delta)
        f_ub, f_mid, f_lb = objective_function(x_ub),objective_function(x_mid),objective_function(x_lb)    
        f.write(f"\nCurrently we have:\n\tx0 - |Δ| = {x_lb}\n\tx0 = {x_mid}\n\tx0 +|Δ| = {x_ub}\n")
        f.write(f"function values are:\n\tf(x0 - |Δ|) = {f_lb}\n\tf(x0) = {f_mid}\n\tf(x0 + |Δ|) = {f_ub}\n\n")
        if f_lb>=f_mid and f_mid>=f_ub:
            delta = abs(delta)
            f.write(f"Value of delta = {delta}\n\n")
            break
        elif f_lb <= f_mid and f_mid<=f_ub:
            delta = -abs(delta)
            f.write(f"Value of delta = {delta}\n\n")
            break
        else:
            f.write("The current initial guess is not suitable.\nChoosing another point at random....\n")
            x_mid = random.uniform(a,b)
            f.write(f"new guess is x0 = {x_mid}\n")

    f.write(f'{"#It":<15}{"x_k":<15}{"x_k+1":<15}{"f(x_k)":<15}{"f(x_k+1)":<15}\n')
    while True:
        #Step 3
        x_new = x_mid + pow(2,k)*delta
        f_new = objective_function(x_new)
        k+=1
        # function_values.append(-1*f_new)
        # x_values.append(x_new)
        f.write(f"{k:<15}{round(x_mid,4):<15}{round(x_new,4):<15}{round(f_mid,4):<15}{round(f_new,4):<15}\n")
        if f_new >= f_mid: #Terminating condition
            break
        x_old = x_mid
        x_mid = x_new
        f_mid = f_new

    if k == 1:
        #This happens when f(x0 - |Δ|) = f(x0) or f(x0) = f(x0 + |Δ|)
        x1,x2 = give_interval_in_correct_order(x_mid,x_new)  
    else:
        x1,x2 = give_interval_in_correct_order(x_old,x_new)
    
    f.write(f"\nThe minimum point lies between {round(x1,4)} and {round(x2,4)} \n")
    f.write(f"Total number of function evaluations: {fneval}\n")
    f.close()
    # x = np.linspace(a, b, 1000)
    # y = [(-1*objective_function(val)) for val in x]
    # plt.figure(figsize=(8,5))
    # plt.scatter(x_values,function_values, color = "red", label = "Function iteration through Bounding Phase method")
    # plt.scatter(x1,-1*objective_function(x1),color = "green", label = "End interval given by bounding phase method")
    # plt.scatter(x2,-1*objective_function(x2),color = "green")
    # plt.plot(x, y, color="blue", label="Objective Function")
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.legend()
    # plt.show()

    # plt.plot([i for i in range(1,k+1)],function_values)
    # plt.title("f(x) vs no. of iterations in bounding phase method")
    # plt.xlabel("No. of iterations")
    # plt.ylabel("f(x)")
    # plt.savefig('BP.png')
    return x1,x2

                
def golden_section_method(lowerLimit:float, upperLimit:float, accuracy:float = 1e-3):
    #method "partial" returns a function which has parametes a and b binded to "normalized_function"
    #This means only the variable x is needed to be passed to the returned function
    #Step 1
    global fneval
    normalizedFunction = partial(normalized_function, lowerLimit, upperLimit)
    a,b,lengthOfInterval,k = 0,1,1,0
    goldenRatio = -1 + (1+sqrt(5))/2 #0.618.....
    fneval = 0
    #x_values=[]
    # function_values = []
    f = open(r"Golden_Section_Search_Mehtod_iterations.txt","w",encoding="utf-8")
    f.write("\n")
    f.write("Golden Section Search Method\n")
    f.write("The values of a and b below are shown after the region elimination.\n\n")
    f.write(f'{"It#":<15}{"w1":<15}{"w2":<15}{"f(w1)":<15}{"f(w2)":<15}{"a":<15}{"b":<15}{"L":<15}\n')
    while abs(lengthOfInterval) > accuracy: #Termination Condition
        #Step 2
        w1 = a + goldenRatio*lengthOfInterval
        w2 = b - goldenRatio*lengthOfInterval
        f1 = normalizedFunction(w1)
        f2 = normalizedFunction(w2)
        # x_values.append(w2)
        # function_values.append(-1*f2)
        #Decision of eliminating the region
        if f1 > f2:
            b = w1
            lengthOfInterval = b-a
        elif f2 > f1:
            a = w2
            lengthOfInterval = b-a
        else:
            a = w2
            b = w1
            lengthOfInterval = b-a
        k+=1
        
        # f.write(f"{k:<15}{w1:<15}{w2:<15}{f1:<15}{f2:<15}{a:<15}{b:<15}{lengthOfInterval:<15}\n")
        f.write(f"{k:<15}{round(w1,4):<15}{round(w2,4):<15}{round(f1,4):<15}{round(f2,4):<15}{round(a,4):<15}{round(b,4):<15}{round(lengthOfInterval,4):<15}\n")
    # x = np.linspace(0, 1, 500)
    # y = [(-1*normalizedFunction(val)) for val in x]
    # plt.figure(figsize=(8,5))
    # plt.scatter(x_values,function_values, color = "red", label = "Function iteration through Golden section search method")
    # plt.scatter(a,-1*normalizedFunction(a),color = "green", label = "End interval given by Golden section search method")
    # plt.scatter(b,-1*normalizedFunction(b),color = "green")
    # plt.plot(x, y, color="blue", label="Normalized Function")
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.legend()
    # plt.show()
    f.write(f"\nthe final region(normalized): ({a},{b}) found in {k} iterations") 
    a = a*(upperLimit-lowerLimit) + lowerLimit
    b = b*(upperLimit-lowerLimit) + lowerLimit
    f.write(f"\nthe final region(unnormalized): ({a},{b})") 
    f.write(f"\nNumber of function evaluations: {fneval}")
    f.close()
    return a,b

    
    

def main_function():
    with open("input.txt") as f:
        inputParam = f.readline().split()
            
    a = float(inputParam[0])
    b = float(inputParam[1])
    delta = float(inputParam[2])
    try: 
        initialGuess = float(inputParam[3])
    except ValueError:
        initialGuess = None
    accuracy = float(inputParam[4])
    a,b = bounding_phase_method(a,b, delta,initialGuess)
    golden_section_method(a,b,accuracy)
    
main_function()
