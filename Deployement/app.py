
import streamlit as st
import pickle as pk

st.title('Understanding the factors influencing US housing Prices')
st.header('Factors considered')

st.write('''__Ques.__ Find publicly available data for key factors that influence US home prices nationally. Then, build a data science model that explains how these factors impacted home prices over the last 20 years.

Use the S&P Case-Schiller Home Price Index as a proxy for home prices :(fred.stlouisfed.org/series/CSUSHPISA).''')

st.write('''__Understanding the problem:__

First of all, we need to figure out what are the key factors which are influcening US home price nationally. After that, we need to get the valid and trustable sources so that we can use them in the data.


### Factors influencing the US home price nationally

1. GDP
2. Unemployement rate
3. Inflation
4. Population growth
5. Mortgages rate
6. Federal fund rate
7. Housing supply


### Note:

There are other factors which influnces the house price such as the average size of the house, immigration population, marriages, locality parameters and various other factors. But, either suitable data is not present or due to time constraint, it could be processed.''')


st.text('Note: as the data extracted is monthly, \n  so we also have month and year column for our analysis!')

st.header('Bi-variate analysis')
st.text('This is what we understood from the data and the patterns')

st.image(r"C:\Users\LENOVO\US Housing price\Uemployemet Rate.png", caption='cost_index tends to rise with decrease in the employement rate')
st.image(r"C:\Users\LENOVO\US Housing price\mortgage.png", caption= 'No clear relation between cost_index and mortgage fund rate')
st.image(r"C:\Users\LENOVO\US Housing price\federal_rate.png", caption= 'No clear relation netween cost_index and federal rate')
st.image(r"C:\Users\LENOVO\US Housing price\housing_supply.png", caption= 'Initially decrease in housing supply reduced the cost_index, but then pattern changed')
st.image(r"C:\Users\LENOVO\US Housing price\Year.png", caption= 'As we are moving forward in time, the cost_index is increasing')
st.image(r"C:\Users\LENOVO\US Housing price\Month.png", caption= 'There is no clear relation between cost_index and month')
st.image(r"C:\Users\LENOVO\US Housing price\gdp_pCapita.png", caption= 'As the per capita GDP is increasing, the cost_index is also increasing')
st.image(r"C:\Users\LENOVO\US Housing price\cpi.png", caption= 'As the Consumer price inflation is increasing, cost_index is also increasing')
st.image(r"C:\Users\LENOVO\US Housing price\population.png", caption= 'As the population is increasing, cost_index is also increasing')

st.text('Let us understand the data further!')

st.image(r"C:\Users\LENOVO\US Housing price\corr.png")

st.write('''__Conclusion:__

- Like observed through the scatter plot above, the cost_index is linearly correlated to "per capita GDP", "Year", "CPI", and "Population"
- Unemployement has a negative correlation value for cost_index
- And mortage funds rate is also negatively correlated.
- But, federal funds rate and month have very low correlation with the cost_index.


__Note:__

- The features are also correlated with each other, so let us find out, because if we are to build model, then we might consider dropping them.''')
st.image(r"C:\Users\LENOVO\US Housing price\corr2.png")

st.write('''__Note:__

We can also see one more thing here, the other features are also correlated with each other. All the four features which are linearly correlated to our cost_index, as also correlated with each other. 

- As we can see here, all the four features "per capita GDP", "CPI", "Population", and "Year" all are highly correlated with each other. So, we might consider using only using one of these features instead of all of them in the model building!''')

st.write('Let us take a closer look at the month index')

st.image(r"C:\Users\LENOVO\US Housing price\month_close.png")

st.write('''So, can we assume from here that the prices of the house took a dip in the 2nd month of the year and after that it increased up until the end of the year!''')

st.write('Let us also understand the frequency distribution of the cost_index')
st.image(r"C:\Users\LENOVO\US Housing price\distribution.png")

st.write('So, we dropped the following columns: Population, CPI and per capita GDP. We kept the Year column because it is easy to interpret.')

st.header('Model building')

st.image(r"C:\Users\LENOVO\US Housing price\Models.png")

st.write('So, we would use the Random forest regressor for our prediction')

st.header('Get an estimated cost_index!')
st.write('Input values below')

unemployement_rate = st.number_input('Enter unemployent Rate')
mortgage = st.number_input('Enter Mortgage fund rate')
federal_fund = st.number_input('Enter federal fund rate')
housing = st.number_input('Enter housing supply value')
year = st.number_input('Enter year', min_value= 2000, max_value=3000, step= 1)

model = pk.load(open(r'C:\Users\LENOVO\US Housing price\modelRF.pkl', 'rb'))

if st.button('Predict'):
    val = model.predict([[unemployement_rate,mortgage, federal_fund, housing, year]])
    st.write(val)
