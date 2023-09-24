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
import matplotlib.pyplot as plt

function_3 = lambda x: 9*(x**3) - 4*x + 3/x
lower_limit = 1
upper_limit = 6
bins = 10

delta_x = (upper_limit - lower_limit) / bins


def simpsons_rule(func, lower_limit, upper_limit, bins):
    x_values_exact = np.linspace(lower_limit, upper_limit, 1000)
    y_values_exact = [func(x) for x in x_values_exact]


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

    return x_values, y_values, area_1, odd_indiv_area, odd_area, even_area, integral, x_values_exact, y_values_exact


x_values, y_values, area_1, odd_indiv_area, odd_area, even_area, integral, x_values_exact, y_values_exact = simpsons_rule(function_3, lower_limit, upper_limit, bins)

window_size = 3

# Create lists to store coefficients and corresponding x values for each subset
subset_coefficients = []
subset_x_values = []

# Iterate through the data to create subsets and fit polynomials
for i in range(len(x_values) - window_size + 1):
    x_subset = x_values[i:i + window_size]
    y_subset = y_values[i:i + window_size]

    # Fit a polynomial to the current subset
    coefficients = np.polyfit(x_subset, y_subset, 2)  # You can change the degree as needed
    subset_coefficients.append(coefficients)
    subset_x_values.append(x_subset)

# Plot the data points and the fitted polynomials for each subset
plt.plot(x_values_exact, y_values_exact, label='Data Points', color='blue', alpha = 0.5)

for coefficients, x_subset in zip(subset_coefficients, subset_x_values):
    x_fit = np.linspace(min(x_subset), max(x_subset), 100)
    y_fit = np.polyval(coefficients, x_fit)
    plt.plot(x_fit, y_fit, linestyle='--', color='red', alpha = 0.5)


plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Polynomial Fits for Subsets of Data')
plt.legend()
plt.grid(True)
plt.show()



