import numpy as np
import altair as alt
import pandas as pd
import streamlit as st

st.header('양도경 사랑해')

# Example 1

st.write('*Hello*, *Dokyeong!* :heart: :heart: :heart:')

# Example 2

st.write('3000'+' '+'만큼 사랑해 내 아가 도경이!')

# Example 3

df = pd.DataFrame({
     'first column': [1, 2, 3, 4],
     'second column': [10, 20, 30, 40]
     })
st.write(df)

# Example 4

st.write('Below is a DataFrame:', df, 'Above is a dataframe.')

# Example 5

df2 = pd.DataFrame(
     np.random.randn(200, 3),
     columns=['a', 'b', 'c'])

c = alt.Chart(df2).mark_circle().encode(
     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
st.write(c)


code = '''def hello():
    print("Hello, Streamlit!")'''
st.code(code, language='python')
