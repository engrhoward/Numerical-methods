dimport streamlit as st

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
        function = st.text_input("Function")
        col1, col2 = st.columns(2)
        with col1:
            lower_limit = st.text_input("Upper Limit")
        with col2:
            upper_limit = st.text_input("Lower Limit")

        #insert latex of function
        
        if function and lower_limit is not None and upper_limit is not None:
            definite_integral_latex =rf'\int_{{{lower_limit}}}^{{{upper_limit}}} {function} \, dx'
            st.latex(definite_integral_latex)

        bins = st.slider("Enter number of divisions", 2,10,20)

        

