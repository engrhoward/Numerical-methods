def trapezoidal_integration(func,lower_limit,upper_limit,bins):
    integral = 0.0
    
    x_values = np.linspace(lower_limit, upper_limit, bins +1)
    y_values = [func]


            