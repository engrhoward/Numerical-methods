def rectangular_integration(func, lower_limit, upper_limit, bins):

    delta_x = (upper_limit-lower_limit)/bins
    integral = 0


    for i in range(bins):
        x_midpoint = lower_limit +(i+0.5)*delta_x

        integral = integral +func(x_midpoint)*delta_x

    return integral

def f(x):
    return x**2

lower_limit = 0
upper_limit = 2
bins = 1000

print(rectangular_integration(f,lower_limit,upper_limit,bins))