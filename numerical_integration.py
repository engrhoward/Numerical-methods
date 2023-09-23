'''
import matplotlib.pyplot as plt

x_values = [1,2,3,4,5]
y_values = [2,4,9,16,25]

x_values_segments = []
y_values_segments = []

for i in range(len(x_values)):
    x_values_segment = [x_values[i], x_values[i+1]]
    y_values_segment = [y_values[i], y_values[i+1]]

    plt.plot(x_values_segment, y_values_segment, 
             color = 'red', linestyle = "--")

plt.show()
'''

'''
def rectangular_integration(func, lower_limit, upper_limit, bins):
    integral = 0.0

    x_midpoints = []
    
    # Define a default behavior when no function is provided
    def default_function(x):
        return 0.0

    if func:  # Check if a function is provided
        for i in range(bins):
            x_midpoint = lower_limit + (i + 0.5) * delta_x
            integral = integral + func(x_midpoint) * delta_x
            x_midpoints.append(x_midpoint)
    else:
        # Use the default function if no function is provided
        integral = default_function(lower_limit) * (upper_limit - lower_limit)

    return integral, x_midpoints
'''
import numpy as np

function_3 = lambda x: 9*(x**3) - 4*x + 3/x
lower_limit = 1
upper_limit = 6
bins = 10

delta_x = (upper_limit - lower_limit) / bins


def simpsons_rule(func, lower_limit, upper_limit, bins):

    x_values = np.linspace(lower_limit, upper_limit, bins+1)
    y_values = [func(x) for x in x_values]

    integral = 0.0

    area_1 = func(x_values[0])
    area_last = func(x_values[-1])

    #get odd summation
    odd_area = 0.0
    odd_indiv_area = [func(x) for x in x_values[1::2]]

    for i in odd_indiv_area:
        odd_area = odd_area + i


    # get even summation

    even_area = 0.0
    even_indiv_area = [func(x) for x in x_values[2:-1:2]]

    for i in even_indiv_area:
        even_area = even_area + i

    integral = (delta_x/3) * (area_1 + 4*(odd_area) + 2*(even_area) +area_last)

    return x_values, y_values, area_1, odd_indiv_area, odd_area, even_area, integral


x_values, y_values, area_1, odd_indiv_area, odd_area, even_area, integral= simpsons_rule(function_3, lower_limit, upper_limit, bins)

even_values = x_values[1::2]
odd_values = x_values[2:-1:2]


