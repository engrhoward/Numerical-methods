import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Numerical Methods")


non_linear, linear, integ = st.tabs(['Non-Linear Equation',
                            'Linear Equations',
                            'Numerical Integration'])

# create a function that converts ^ to **
def convert_syntax(input_str):
    output_str = input_str.replace('^','**')
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
        function = st.text_input("Function")
        
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
        
        #latex of function
        if function and lower_limit is not None and upper_limit is not None:
            definite_integral_latex =rf'\int_{{{lower_limit}}}^{{{upper_limit}}} {function} \, dx'
            st.latex(definite_integral_latex)

        #st.latex(r'\delta_x = \frac{\text{Upper Limit} - \text{Lower Limit}}{\text{Bins}} = \frac{%d - %d}{%d} = %.2f' % (upper_limit, lower_limit, bins, delta_x))

        
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
            result = rectangular_integration(f, lower_limit, upper_limit, bins)
            st.markdown(
                f"<div style=padding: 10px; border-radius: 5px;'>"
                f"<p style='font-size: 18px; font-weight: bold; color: #333;'>The Approximate integral is:</p>"
                f"<p style='font-size: 16px; line-height: 1.4; color: #555;'>"
                f"<p style='font-size: 20px; font-weight: bold;'>{integral}</p>"
                f"</div>",
                unsafe_allow_html=True
            )


        
        if function and lower_limit is not None and upper_limit is not None:
            x_values = np.linspace(lower_limit, upper_limit, 100)
            y_values = [f(x) for x in x_values]

            plt.figure(figsize=(10, 6))
            plt.plot(x_values, y_values, label=f'Function: {function}')
            plt.bar(x_midpoints, [f(x) for x in x_midpoints], width=delta_x, alpha=0.5, 
                    label='Rectangular Approximation', edgecolor = 'black',
                    linewidth = 2)
            plt.scatter(x_midpoints, [f(x) for x in x_midpoints], color = 'black')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('Function and Rectangular Approximation')
            plt.legend()
            st.pyplot(plt)

#adding comment here to try to push
print("hello")

print("hello")

    with tab_b:
        function_2 = st.text_input('Function', key = 'Trapezoidal')

        trap_upper, trap_lower = st.columns(2)

        with trap_upper:
            upper_limit = st.number_input("Upper Limit", step = 1, key = 'upper_trap')
        with trap_lower:
            lower_limit = st.number_input("Lower Limit", step = 1, key = 'lower_trap')

        #slider for number of divisions
        bins = st.slider("Enter number of divisions", 2,100,20, key ='trapezoidal')

        #computation of delta_x
        delta_x = (float(upper_limit) - float(lower_limit))/bins
        
        def f(x):
            return eval(convert_syntax(function_2))
        
        #latex of function
        if function and lower_limit is not None and upper_limit is not None:
            definite_integral_latex =rf'\int_{{{lower_limit}}}^{{{upper_limit}}} {function_2} \, dx'
            st.latex(definite_integral_latex)

        st.latex(r'\delta_x = \frac{\text{Upper Limit} - \text{Lower Limit}}{\text{Bins}} = \frac{%d - %d}{%d} = %.2f' % (upper_limit, lower_limit, bins, delta_x))
        
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

            # Plot the function curve
            x_values = np.linspace(lower_limit, upper_limit, 100)
            y_values = [func(x) for x in x_values]
            plt.plot(x_values, y_values, label=f'Function: {function_2}')

            # Customize the plot
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('Function and Trapezoidal Approximation')
            plt.legend()

            return integral

        if function_2 and lower_limit is not None and upper_limit is not None:
            result = trapezoidal_integration(f, lower_limit, upper_limit, bins)
            st.markdown(f"Approximate integral (Trapezoidal Method): {result}")

        st.pyplot(plt)