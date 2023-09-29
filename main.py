import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re

st.title("Numerical Methods")


non_linear, linear, integ = st.tabs(['Non-Linear Equation',
                            'Linear Equations',
                            'Numerical Integration'])

# create a function that converts ^ to **

def convert_syntax(input_str):
    #accomodating exponents
    output_str = input_str.replace('^','**')
        
    #accomodating trig functions

    trig_functions = ['sin','cos', 'tan']

    for trig_func in trig_functions:
        output_str = output_str.replace(trig_func + '(', 'np.' + trig_func + '(')

    # Adding * for multiplication
    output_str = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', output_str)
    output_str = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', output_str)

    return output_str
    
    

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
        st.markdown(
            """
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                <p style="font-size: 18px; font-weight: bold; color: #333;">Rectangular Summation (Midpoint)</p>
                <p style="font-size: 16px; line-height: 1.4; color: #555;">
                    Rectangular summation (midpoint) is a numerical integration technique used to estimate the area under the curve 
                    by dividing it into smaller rectangles and adding all of them together.
                </p>
                <p style="font-size: 16px; line-height: 1.4; color: #555;">
                    The summation form of this technique is expressed as:
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.latex(r'\int_a^b f(x) \, dx \approx \sum_{i=1}^{n} f\left(\frac{x_{i-1}+x_i}{2}\right) \cdot \Delta x')

        #input function
        function = st.text_input("Function", key = 'rectangular')
        
        
        col1, col2 = st.columns(2) # create column for upper and lower inpt
        with col1:
            upper_limit = st.number_input("Upper Limit", step = 1)
        with col2:
            lower_limit = st.number_input("Lower Limit", step = 1)

        #slider for number of divisions
        bins = st.slider("Enter number of divisions", 2,100,20)
        
        #computation of delta_x
        delta_x = (float(upper_limit) - float(lower_limit))/bins
        
        def f(x):
            return eval(convert_syntax(function))
        

        #st.latex(r'\delta_x = \frac{\text{Upper Limit} - \text{Lower Limit}}{\text{Bins}} = \frac{%d - %d}{%d} = %.2f' % (upper_limit, lower_limit, bins, delta_x))

        def rectangular_integration(func,lower_limit, upper_limit ,bins):
            integral = 0.0

            x_midpoints = []
            for i in range(bins):
                x_midpoint = lower_limit + (i+0.5)*delta_x
                integral = integral + func(x_midpoint)*delta_x
                
                x_midpoints.append(x_midpoint)
                
            return integral, x_midpoints

        # latex of function and answer
        if function and lower_limit is not None and upper_limit is not None:
            try:
                integral, x_midpoints = rectangular_integration(f, lower_limit, upper_limit, bins)
                definite_integral_latex = rf'\int_{{{lower_limit}}}^{{{upper_limit}}} {function} \, dx = {integral}'
                st.latex(definite_integral_latex)

                # graph of function and approximate integral

                x_values = np.linspace(lower_limit, upper_limit, 100)
                y_values = [f(x) for x in x_values]

                plt.figure(figsize=(10, 6))
                plt.plot(x_values, y_values, label=f'Function: {function}')
                plt.bar(x_midpoints, [f(x) for x in x_midpoints], width=delta_x, alpha=0.5,
                        color = 'blue', label='Rectangular Approximation', edgecolor = 'black',
                        linewidth = 2)
                plt.scatter(x_midpoints, [f(x) for x in x_midpoints], color = 'black')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.title('Function and Rectangular Approximation')
                plt.legend()
                
                st.pyplot(plt)
            
            except Exception as e:
                st.error(f"Error: Invalid input")

    with tab_b:
        st.markdown(
            """
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                <p style="font-size: 18px; font-weight: bold; color: #333;">Trapezoidal Summation</p>
                <p style="font-size: 16px; line-height: 1.4; color: #555;">
                    Trapezoidal Summation is a numerical integration method used to approximate
                    the area under a function within a specified range by dividing it into 
                    trapezoids and summing their areas. This technique provides an estimation 
                    of definite integrals for various applications.
                </p>
                <p style="font-size: 16px; line-height: 1.4; color: #555;">
                    The summation form of this technique is expressed as:
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.latex(r'\int_a^b f(x) \, dx \approx \frac{\Delta x}{2} \left[ f(a) + 2\sum_{i=1}^{n-1} f(x_i) + f(b) \right]')

        function_2 = st.text_input('Function', key='Trapezoidal')
        

        trap_upper, trap_lower = st.columns(2)

        with trap_upper:
            upper_limit = st.number_input("Upper Limit", step=1, key='upper_trap')
        with trap_lower:
            lower_limit = st.number_input("Lower Limit", step=1, key='lower_trap')

        #slider for number of divisions
        bins = st.slider("Enter number of divisions", 2, 100, 20, key='trapezoidal')

        #computation of delta_x
        delta_x = (float(upper_limit) - float(lower_limit)) / bins
        
        def f_2(x):
            return eval(convert_syntax(function_2))

        #st.latex(r'\delta_x = \frac{\text{Upper Limit} - \text{Lower Limit}}{\text{Bins}} = \frac{%d - %d}{%d} = %.2f' % (upper_limit, lower_limit, bins, delta_x))
        
        def trapezoidal_integration(func, lower_limit, upper_limit, bins):
            integral = 0.0

            x_values = np.linspace(lower_limit, upper_limit, bins + 1)
            y_values = [func(x) for x in x_values]

            # Create a figure for plotting
            plt.figure(figsize=(10, 6))

            for i in range(1, bins):
                integral += y_values[i]

                # Plot each trapezoid
                plt.fill_between([x_values[i - 1], x_values[i]], [0, 0], [y_values[i - 1], y_values[i]],
                                alpha=0.5, edgecolor='black', facecolor='none')

            integral += (y_values[0] + y_values[-1]) / 2
            integral *= delta_x

            #latex of function
            if function_2 and lower_limit is not None and upper_limit is not None:
                definite_integral_latex =rf'\int_{{{lower_limit}}}^{{{upper_limit}}} {function_2} \, dx= {integral}'
                st.latex(definite_integral_latex)

            # Plot the function curve
            x_values_exact = np.linspace(lower_limit, upper_limit, 100)
            y_values_exact = [func(x) for x in x_values_exact]
            plt.plot(x_values_exact, y_values_exact, label=f'Function: {function_2}',
                     linewidth = 1, color = 'black')
            plt.scatter(x_values,y_values)

            for i in range(len(x_values) - 1):
                x_segment = [x_values[i], x_values[i + 1]]
                y_segment = [y_values[i], y_values[i + 1]]
                plt.plot(x_segment, y_segment, color='red', linewidth = 1)
                plt.fill_between(x_segment, y_segment, color = 'blue',
                                 alpha = 0.5)

            # Customize the plot
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('Function and Trapezoidal Approximation')
            plt.legend()

            return integral

        if function_2 and lower_limit is not None and upper_limit is not None:
            try:
                result = trapezoidal_integration(f_2, lower_limit, upper_limit, bins)
                st.pyplot(plt)

            except Exception as e:
                st.error(f"Error: Invalid Input")
        
    with tab_c:
        st.markdown(
            """
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                <p style="font-size: 18px; font-weight: bold; color: #333;">Simpson's 1/3 rule Summation</p>
                <p style="font-size: 16px; line-height: 1.4; color: #555;">
                    Simpson's 1/3 rule is a numerical integration approach that calculates 
                    the area under a curve by dividing it into smaller segments and using 
                    parabolic approximations to estimate the integral. This method provides 
                    accurate results for definite integrals and is particularly effective for 
                    smooth, continuous functions.
                </p>
                <p style="font-size: 16px; line-height: 1.4; color: #555;">
                    The summation form of this technique is expressed as:
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        #insert latex of the algorithm
        st.latex(r'\int_a^b f(x) \, dx \approx \frac{\Delta x}{3} \left[ f(a) + 4\sum_{i=1}^{n-1} f(x_{2i-1}) + 2\sum_{i=1}^{n-2} f(x_{2i}) + f(b) \right]')
        function_3 = st.text_input('Function', key='Simpsons')

        simp_upper, simp_lower = st.columns(2)

        with simp_upper:
            upper_limit = st.number_input("Upper Limit", step=1, key='upper_simp')
        with simp_lower:
            lower_limit = st.number_input("Lower Limit", step=1, key='lower_simp')

        # Slider for the number of divisions
        bins = st.slider("Enter number of divisions (Even numbers only)"
                        , 2, 100, 20, step = 2, key='simpsons')

        # Computation of delta_x
        delta_x = (float(upper_limit) - float(lower_limit)) / bins

        def f_3(x):

            return eval(convert_syntax(function_3))


        def simpsons_rule(func, lower_limit, upper_limit, bins):
            x_values_exact = np.linspace(lower_limit, upper_limit, 1000)
            y_values_exact = [func(x) for x in x_values_exact]

            x_values_simp = np.linspace(lower_limit, upper_limit, bins + 1)
            y_values_simp = [func(x) for x in x_values_simp]

            integral = 0.0

            area_1 = func(x_values[0])
            area_last = func(x_values[-1])

            # Get odd summation
            odd_area = 0.0
            odd_indiv_area = [func(x) for x in x_values_simp[1::2]]

            for i in odd_indiv_area:
                odd_area = odd_area + i

            # Get even summation
            even_area = 0.0
            even_indiv_area = [func(x) for x in x_values_simp[2:-1:2]]

            for i in even_indiv_area:
                even_area = even_area + i
            
            integral = (delta_x / 3) * (area_1 + 4 * (odd_area) + 2 * (even_area) + area_last)

            definite_integral_latex =rf'\int_{{{lower_limit}}}^{{{upper_limit}}} {function_3} \, dx= {integral}'
            st.latex(definite_integral_latex)

            # Plot the data points and the fitted polynomials for each subset
            plt.figure(figsize=(10, 6))
            plt.plot(x_values_exact, y_values_exact, label='Data Points', color='blue', alpha=0.5)
            plt.scatter(x_values_simp, y_values_simp, label = 'Simpson Approximation')

            window_size = 3

            # Create lists to store coefficients and corresponding x values for each subset
            subset_coefficients = []
            subset_x_values = []

            # Iterate through the data to create subsets and fit polynomials
            for i in range(len(x_values_simp) - window_size + 1):
                x_subset = x_values_simp[i:i + window_size]
                y_subset = y_values_simp[i:i + window_size]

                # Fit a polynomial to the current subset
                coefficients = np.polyfit(x_subset, y_subset, 2)  # You can change the degree as needed
                subset_coefficients.append(coefficients)
                subset_x_values.append(x_subset)


            for coefficients, x_subset in zip(subset_coefficients, subset_x_values):
                x_fit = np.linspace(min(x_subset), max(x_subset), 100)
                y_fit = np.polyval(coefficients, x_fit)
                plt.plot(x_fit, y_fit, linestyle='--', color='red')
            for coefficients, x_subset in zip(subset_coefficients, subset_x_values):
                
                x_fit = np.linspace(min(x_subset), max(x_subset), 100)
                y_fit = np.polyval(coefficients, x_fit)

                # Plot the data points and the fitted polynomials for each subset
                plt.plot(x_fit, y_fit, linestyle='--', color='red', alpha=0.5)

                # Draw vertical lines from x-axis to the points
                plt.vlines(x_subset, 0, [np.polyval(coefficients, x) for x in x_subset], colors='green', linestyle=':', alpha=0.5)

                # Fill the area under the parabolic segments
                plt.fill_between(x_fit, 0, y_fit, where=(y_fit > 0), color='blue', alpha=0.5)

            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('Polynomial Fits for Subsets of Data')
            plt.legend()

            return x_values_simp, y_values_simp, area_1, odd_indiv_area, odd_area, even_area, integral, x_values_exact, y_values_exact
        
        if function_3 and lower_limit is not None and upper_limit is not None:
            try:

                result = simpsons_rule(f_3, lower_limit, upper_limit, bins)
                st.pyplot(plt)
            
            except Exception as e:
                st.error(f"Error: Invalid Input")
