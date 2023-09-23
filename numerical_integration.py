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
