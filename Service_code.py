import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt

header = st.beta_container()
dataset = st.beta_container()
corrmatrix = st.beta_container()
features = st.beta_container()
targets = st.beta_container()
sliders = st.beta_container()
model_training = st.beta_container()
mape = st.beta_container()
adstock = st.beta_container()
for_del = st.beta_container()
new_corr_filter = st.beta_container()
linear_reg = st.beta_container()
graph = st.beta_container()


with header:
    st.title('Statistical modelling by Yndx')
    st.text('This instrument will help you to find the key factors which affect the target value.')
    st.text('If you want to try this service, download this file or use your own data: ')
    st.text('https://disk.yandex.ru/d/CWhDVoEYLO-yXg')

with dataset:
    st.header('Here you can upload your dataset')
    st.text('Just choose the csv file')
    uploaded_file = st.file_uploader("Choose a file")
    df = pd.read_csv(uploaded_file, sep = ';')
    df_up = df.copy(deep = True)
    st.write(df)

with corrmatrix:
    df = df.loc[:, (df != 0).any(axis=0)]
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    corr_andr_df = df.corr()
    def correlation_matrix(data):
        def highlight_vals(val):
            if 0.7 <= np.abs(val) < 0.9999999:
                color = 'red'
            elif 0.6 <= np.abs(val) < 0.7:
                color = 'pink'
            else:
                return ''
            return 'background-color: %s' % color
        df = corr_andr_df
        return df.style.applymap(highlight_vals)
    df_paint = correlation_matrix(corr_andr_df)
    df_paint = correlation_matrix(corr_andr_df)
    st.header('**Correlation matrix**')
    st.text('If correlation is red - make a sum of factors or throw away one of them')
    st.write(df_paint)

with features:
    st.header('Correlation filter')
    st.text('These factors have large correlation.')
    st.text('Please, make new factors by using SUM of these columns')
    df_c = df.corr()
    names = df_c.columns
    li = []
    for i in range(len(df_c)):
        for j in range(len(df_c)):
            if ( (df_c.iloc[i,j] >= 0.7) or (df_c.iloc[i,j] <= -0.7) ) and (df_c.iloc[i,j] != 1.0):
                a = [names[i], names[j], df_c.iloc[i,j]]
                li.append(a)
            if len(li) >= 1:
                df_out = pd.DataFrame(li, columns=['Factor 1', 'Factor 2', 'Corr'])
            else:
                df_out = 'There are no correlating factors'

    st.write(df_out)

with targets:
    st.header('Target variable')
    st.text('Now let us choose the Target variable')
    option = st.selectbox( 'Which factor are going to model?', (names))
    st.write('You have chosen: ', option)
    Y = df[option]
    df = df.drop(option, axis = 1)
    new_names = df.columns

with mape:
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

with adstock:
    def adstock(data, rate):
        tt = np.empty(len(data))
        tt[0] = data[0]

        for i in range(1, len(data)):
            tt[i] = data[i] + tt[i - 1] * rate
        return tt

#with sliders:
 #   for i in range(1, len(new_names)):
  #      a = st.slider(new_names[i], min_value=0.0, max_value=1.0,  step=0.1)

        #df[new_names[i]] = pd.DataFrame(adstock(df[new_names[i]], a), columns = new_names[i])

#st.write(df)

with for_del:
    options_del = st.multiselect(
    'Which factor do you want to delete?',new_names )
    df.drop(options_del, 1, inplace=True)
    X = df.copy(deep = True)
    st.write(X)

with new_corr_filter:
    st.header('Correlation filter after removing features')
    df_n = X.corr()
    names_n = df_n.columns
    li_n = []
    for k in range(len(df_n)):
        for m in range(len(df_n)):
            if ( (df_n.iloc[k,m] >= 0.7) or (df_n.iloc[k,m] <= -0.7) ) and (df_n.iloc[k,m] != 1.0):
                bre = [names_n[k], names_n[m], df_n.iloc[k,m]]
                li_n.append(bre)
            if len(li_n) >=1 :
                df_out_n = pd.DataFrame(li_n, columns=['Factor 1', 'Factor 2', 'Corr'])
            else:
                df_out_n = 'There are no correlating factors'

    st.write(df_out_n)


with linear_reg:
    st.header('Modelling results')
    model = sm.OLS(Y, sm.add_constant(X)).fit()
    st.write('Your modelling results are: ', model.params)
    st.write('R^2: ', model.rsquared)
    st.write(model.summary())

with graph:
    df_up_names = df_up.columns
    x_time = df_up[df_up_names[0]]

    prstd, iv_l, iv_u = wls_prediction_std(model)
    fig, ax = plt.subplots()
    ax.plot(x_time, Y, label = "Fact")
    ax.plot(x_time, model.fittedvalues, label = "Model")
    ax.plot(x_time, iv_u, 'r--', label="Min/max")
    ax.plot(x_time, iv_l, 'r--')
    ax.legend()

    st.pyplot(fig)




