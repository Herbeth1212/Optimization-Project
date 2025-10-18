from random import uniform
from math import sqrt

def bounding_phase_method(func, a, b, alpha0= None, delta=0.5):
    if alpha0 == None:
        alpha0 = uniform(a,b)
    while True:
        f0 = func(alpha0)
        if func(alpha0 - abs(delta)) >= f0 >= func(alpha0 + abs(delta)): 
            delta = abs(delta)
            break
        elif func(alpha0 - abs(delta)) <= f0 <= func(alpha0 + abs(delta)): 
            delta = -abs(delta)
            break
        else: 
            alpha0 = uniform(a,b)

    k, alpha_current = 0, alpha0
    while True:
        alpha_prev = alpha_current
        alpha_current = alpha_prev + (2**k) * delta
        if func(alpha_current) >= func(alpha_prev):
            return tuple(sorted((alpha_prev - (2**(k-1))*delta, alpha_current)))
        k += 1

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
