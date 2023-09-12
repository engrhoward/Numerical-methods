import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Numerical Methods")

non_linear, linear, integ = st.tabs(['Non-Linear Equation',
                            'Linear Equations',
                            'Numerical Integration'])

with non_linear:
    tab_a, tab_b, tab_c = st.tabs(["Bisection Method",
                                "Fixed Point Problem",
                                "Newton's Rhapson Method"])
with linear:
    tab_a, tab_b, tab_c = st.tabs(['Gaussian Elimination Method',
                                   'LU-Decomposition',
                                   'Gauss-Seidel Method'])
with integ:
    tab_a, tab_b, tab_c = st.tabs(['Rectangular (Midpoint)','Trapezoidal',"Simpson's"])

    
    with tab_a:
        
        #input function
        function = st.text_input("Function")

        # create a function that converts ^ to **
        def convert_syntax(input_str):
            output_str = input_str.replace('^','**')
            return output_str
        
        
        col1, col2 = st.columns(2) # create column for upper and lower inpt
        with col1:
            upper_limit = st.number_input("Upper Limit", step = 1)
        with col2:
            lower_limit = st.number_input("Lower Limit", step = 1)

        #slider for number of divisions
        bins = st.slider("Enter number of divisions", 2,10,20)
        
        #computation of delta_x
        delta_x = (float(upper_limit) - float(lower_limit))/bins
        
        def f(x):
            return eval(convert_syntax(function))
        
        #latex of function
        if function and lower_limit is not None and upper_limit is not None:
            definite_integral_latex =rf'\int_{{{lower_limit}}}^{{{upper_limit}}} {function} \, dx'
            st.latex(definite_integral_latex)

        st.latex(r'\delta_x = \frac{\text{Upper Limit} - \text{Lower Limit}}{\text{Bins}} = \frac{%d - %d}{%d} = %.2f' % (upper_limit, lower_limit, bins, delta_x))

        
        def rectangular_integration(func,lower_limit,upper_limit,bins):
            integral = 0.0

            x_midpoints = []
            for i in range(bins):
                x_midpoint = lower_limit + (i+0.5)*delta_x
                integral = integral + func(x_midpoint)*delta_x

                x_midpoints.append(x_midpoint)
                
            return integral, x_midpoints
        integral, x_midpoints = rectangular_integration(f, lower_limit, upper_limit, bins)

        if function and lower_limit is not None and upper_limit is not None:
            result = integral
            st.write("Approximate integral:", result)

        
        if function and lower_limit is not None and upper_limit is not None:
            x_values = np.linspace(lower_limit, upper_limit, 100)
            y_values = [f(x) for x in x_values]

            plt.figure(figsize=(10, 6))
            plt.plot(x_values, y_values, label=f'Function: {function}')
            plt.fill_between(x_midpoints, 0, [f(x) for x in x_midpoints], alpha=0.2, label='Rectangular Area')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('Function and Rectangular Approximation')
            plt.legend()
            st.pyplot(plt)
