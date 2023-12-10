import warnings, math, logging, cmdstanpy, os, json, pickle
# Set the logging level for cmdstanpy to WARNING
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import streamlit as st
from sklearn.impute import SimpleImputer
import numpy as np
# from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from bayes_opt import BayesianOptimization
warnings.filterwarnings("ignore")
plt.style.use('default')

def save_parameters(json_file, biofilter, split_date, params):
    with open(json_file) as f:
        parameters_dict = json.load(f)
    dict_para, dict_para1 = {}, {}
    dict_para['split_date'] = split_date.strftime("%Y-%m-%d")
    if len(params)>0: dict_para['best_params'] = params
    dict_para1[biofilter] = dict_para
    parameters_dict.update(dict_para1)
    # Write dictionary to json
    with open(json_file, 'w') as convert_file:
        convert_file.write(json.dumps(parameters_dict))

def load_parameters(json_file):
    with open(json_file) as f:
        parameters_dict = json.load(f)
    return parameters_dict

def read_log(log_path, site, number):
    if not os.path.exists(log_path): return None
    df_log = pd.read_csv(log_path, parse_dates=True, index_col=2)
    df_log = df_log[['site','biofilter']][(df_log['site'] == site)&
                                      (df_log['biofilter'].str.contains(str(number)))]
    if len(df_log) > 0: return df_log
    else: return None

def plot_train_test(df, label, sub=None):
    fig, ax = plt.subplots(figsize=(12,4))
    for i in range(0, len(df)):
        ax.plot(df[i]['ds'].values, df[i]['y'].values, label=label[i])
    ax.set_xlabel('Time')
    ax.set_ylabel('Rotation')
    ax.legend()
    fig.tight_layout()
    # plt.show()
    if sub is not None: st.subheader(sub)
    st.write(fig)

def split_data(df, time_split):
    df_train = df.loc[df['ds'] < pd.to_datetime(time_split)]
    df_test = df.loc[df['ds'] >= pd.to_datetime(time_split)]
    return df_train, df_test


def plot_prediction_train_test(model, train_df, test_df, title=''):
    pred_train, pred_test = model.predict(train_df), model.predict(test_df)
    fig, ax = plt.subplots(2, 1, figsize=(12,8))

    ax[0].plot(pred_train['ds'].values, pred_train['yhat'].values)
    ax[0].plot(pred_test['ds'].values, test_df['y'].values, label='Actual data points')
    ax[0].fill_between(pred_train['ds'].values, pred_train['yhat_lower'].values,
                    pred_train['yhat_upper'].values, color='#0072B2', alpha=0.2)
    ax[0].vlines(pred_train['ds'].values[-1], ymin=train_df['y'].min(),
            ymax=train_df['y'].max(), color='k', linewidth=2, label='Spliting point')
    fig = model.plot(pred_test, ax=ax[0])
    ax[0].title.set_text('Prediction on entire dataset') #, fontdict={'fontsize':15}
    ax[0].legend()

    ax[1].plot(pred_test['ds'].values, test_df['y'].values, label='Measurement')
    ax[1].plot(pred_test['ds'].values, pred_test['yhat'].values, label='Forecast')
    ax[1].fill_between(pred_test['ds'].values, pred_test['yhat_lower'].values,
                    pred_test['yhat_upper'].values, color='#0072B2', alpha=0.2)

    actual = make_imputer(test_df['y'].values, 'median')
    rmse = math.sqrt(mean_squared_error(actual.reshape(actual.shape[0],), pred_test['yhat'].values))
    ax[1].title.set_text(f'Prediction on test dataset (RMSE: {rmse:.2f})')
    ax[1].legend()

    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    # plt.show()
    return fig

def plot_comparison(df):
    fig, ax = plt.subplots(2, 1, figsize=(12,6))
    df_new = df.copy().drop(columns=['actual'])
    columns = df_new.columns
    for i in range(0, len(columns)):
        ax[0].plot(df_new.index, df_new[columns[i]].values, label=df_new[columns[i]].name)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Rotation')
    ax[0].set_title('Comparison between missing data filling methods (Prediction values)', fontdict={'fontsize':15})
    ax[0].legend()

    labels, values = df.columns[1:], []
    actual = df['actual'].values
    for i in range(0, len(labels)):
        values.append(math.sqrt(mean_squared_error(actual, df[labels[i]].values)))
    ax[1].bar(np.arange(len(labels)), values)
    ax[1].set_xlabel('Missing data filling methods')
    ax[1].set_ylabel('Root Mean Square Error')
    ax[1].set_title('Difference in predicion results between filled data and original data', fontdict={'fontsize':10})
    ax[1].set_xticks(np.arange(len(labels)))
    ax[1].set_xticklabels(labels)
    fig.tight_layout()
    return fig

def make_imputer(arr, strategy):
    imputer = SimpleImputer(strategy=strategy)
    y = arr.copy().reshape(-1, 1)
    return imputer.fit_transform(y)

def compare_models(df, split_date, missing_data):
    data = df.copy()
    data = data.rename(columns={data.columns[1]:'y', 'timestamp':'ds'})
    train_df, test_df = split_data(data, split_date)
    model = Prophet().fit(train_df)
    pred = model.predict(test_df)
    final_df = pd.DataFrame(index=pd.to_datetime(test_df['ds'].values),
                            columns=['actual'],
                            data=make_imputer(test_df['y'].values, 'median'))
    final_df['No fill'] = pred['yhat'].values
    for i in range(0, len(missing_data)):
        data_new = data[['ds', missing_data[i]]]
        data_new = data_new.rename(columns={data_new.columns[1]:'y'})
        train_df, test_df = split_data(data_new, split_date)
        model = Prophet().fit(train_df)
        pred = model.predict(test_df)
        final_df[missing_data[i]] = pred['yhat'].values
    return plot_comparison(final_df)

def compute(model, train_df, test_df, title, missing_data=None):
    pred = model.predict(test_df)
    if missing_data is None:
        # Plot the components of the model
        fig2 = model.plot_components(pred)
        return plot_prediction_train_test(model, train_df, test_df, title), fig2
    else:
        final_df = pd.DataFrame(index=pd.to_datetime(test_df['ds'].values),
                                columns=['actual'], data=test_df['y'].values)        
        final_df['No fill'] = pred['yhat'].values
        for i in range(0, len(missing_data)):
            data_new = df[['ds', missing_data[i]]]
            data_new = data_new.rename(columns={data_new.columns[1]:'y'})
            train_df, test_df = split_data(data_new, split_date)
            final_df[missing_data[i]] = pred['yhat'].values
        return plot_comparison(final_df)
    
def explore_data(df):
    df_ = df.copy()
    df_.index = pd.to_datetime(df_['ds'])
    df_ = df_.drop(columns=['ds'])
    # Resample from hourly to daily frequency and calculate the daily mean
    df_ = df_.resample('D').mean()
    df_['date'] = df_.index
    df_['month'] = df_['date'].dt.strftime('%B')
    df_['year'] = df_['date'].dt.strftime('%Y')
    df_['dayofweek'] = df_['date'].dt.strftime('%A')
    df_['quarter'] = df_['date'].dt.quarter
    df_['dayofyear'] = df_['date'].dt.dayofyear
    df_['dayofmonth'] = df_['date'].dt.day
    # df_['weekofyear'] = df_['date'].dt.weekofyear
  
    df_new = df_[['dayofweek','quarter','month','year',
                  'dayofyear','dayofmonth', 'y']]
    fig, ax = plt.subplots(figsize=(12,4))
    ax = sns.barplot(x="month", y="y", hue ='year', data=df_new)
    ax.set_xlabel('Month')
    ax.set_ylabel('Rotation')
    ax.legend()
    fig.tight_layout()
    return fig


def create_model(param):
    model = Prophet(seasonality_mode=param['seasonality_mode'],
                    changepoint_prior_scale=param['changepoint_prior_scale'],
                    holidays_prior_scale=param['holidays_prior_scale'],
                    seasonality_prior_scale=param['seasonality_prior_scale'],
                    weekly_seasonality=True, daily_seasonality=True,
                    yearly_seasonality=True, interval_width=0.95)
    return model


def model_tunning(train_df): #, holiday
    """ Use the hyperopt package
    seasonality_mode = ['multiplicative','additive']
    params = {"holidays_prior_scale": hp.uniform("holidays_prior_scale", 0.01, 10.0),
            "seasonality_prior_scale": hp.uniform("seasonality_prior_scale", 0.01, 10.0),
            "changepoint_prior_scale": hp.uniform("changepoint_prior_scale", 0.001, 0.5),
            'seasonality_mode': hp.choice('seasonality_mode', seasonality_mode)}
    
    def Bayesian_Optimization(param, train_df): #, test_df, holiday
        def objective_function(param):
            model = Prophet(seasonality_mode=param['seasonality_mode'], #holidays=holiday, 
                            changepoint_prior_scale=param['changepoint_prior_scale'],
                            holidays_prior_scale=param['holidays_prior_scale'],
                            seasonality_prior_scale=param['seasonality_prior_scale'],
                            weekly_seasonality=True, daily_seasonality=True,
                            yearly_seasonality=True, interval_width=0.95)
            # Set the changepoint prior scale (adjust the value as needed)
            model.add_seasonality(name="custom_seasonality",
                              period=7,  # Replace with your desired seasonality period
                              fourier_order=10,  # Adjust the Fourier order as needed
                              prior_scale=0.1,  # Adjust this value to control sensitivity (lower values make the model less sensitive)
                              )
            model.fit(train_df)
            pred = model.predict(train_df)
            actual = make_imputer(train_df['y'].values, 'median')
            loss = math.sqrt(mean_squared_error(y_true=actual.reshape(actual.shape[0],), y_pred=pred['yhat'].values))
            print(f'- RMSE: {loss}')
            print(f'Parameters: {param}')
            return {'loss': loss, 'status': STATUS_OK}
        trials, rstate = Trials(), np.random.default_rng(42)
        best_hyperparams = fmin(fn=objective_function, space=param, algo=tpe.suggest,
                                max_evals=50, trials=trials, rstate=rstate)
        return best_hyperparams
    best_param = Bayesian_Optimization(params, train_df) #, test_df, holiday
    best_param['seasonality_mode']= seasonality_mode[best_param['seasonality_mode']]
    model = Prophet(seasonality_mode=best_param['seasonality_mode'], #holidays=holiday, 
                    changepoint_prior_scale=best_param['changepoint_prior_scale'],
                    holidays_prior_scale=best_param['holidays_prior_scale'],
                    seasonality_prior_scale=best_param['seasonality_prior_scale'],
                    weekly_seasonality=True, daily_seasonality=True,
                    yearly_seasonality=True, interval_width=0.95).fit(train_df) """

    # Use the bayes_opt package
    seasonality_mode_list = ['multiplicative','additive']
    params = {"holidays_prior_scale": (0.01, 10.0),
              "seasonality_prior_scale": (0.01, 10.0),
              "changepoint_prior_scale": (0.001, 0.5),
              'seasonality_mode': (0, 1)}
    def objective_function(seasonality_mode, holidays_prior_scale,
                           seasonality_prior_scale, changepoint_prior_scale):
        param = {"holidays_prior_scale": holidays_prior_scale,
                  "seasonality_prior_scale": seasonality_prior_scale,
                  "changepoint_prior_scale": changepoint_prior_scale,
                  'seasonality_mode': seasonality_mode_list[int(seasonality_mode)]}
        model = create_model(param)
        # Set the changepoint prior scale (adjust the value as needed)
        model.add_seasonality(name="custom_seasonality", period=7, fourier_order=10, prior_scale=0.1)
        # model.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')
        # model.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')
        # model.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)




        model.fit(train_df)
        pred = model.predict(train_df)
        actual = make_imputer(train_df['y'].values, 'median')
        score = math.sqrt(mean_squared_error(y_true=actual.reshape(actual.shape[0],), y_pred=pred['yhat'].values))
        return score
    bo = BayesianOptimization(objective_function, params, verbose=2, random_state=42)
    bo.maximize(init_points=10, n_iter=30)
    best_params = bo.max['params']
    best_params['seasonality_mode'] = seasonality_mode_list[int(best_params['seasonality_mode'])]
    model = create_model(best_params).fit(train_df)
    return model, best_params


def plot_tunned_model(model, data, title, log_df=None):
    pred = model.predict(data)
    fig, ax = plt.subplots(figsize=(12,6))
    if log_df is not None:
        ax.vlines(log_df.index, ymin=0, ymax=max(data['y'].values), colors='r', label='Scheduled time')
    ax.plot(data['ds'].values, data['y'].values, label='Actual measurement')
    ax.plot(pred['ds'].values, pred['yhat'].values, label='Forecasted measurement')
    ax.fill_between(pred['ds'].values, pred['yhat_lower'].values, pred['yhat_upper'].values,
                    color='#0072B2', alpha=0.2, label='Uncertainty')
    # ax.set_ylim([-0.2*max(data['y'].fillna(0).values), 1.0*max(data['y'].fillna(0).values)])
    # ax.set_ylim([-0.5*max(data['y'].fillna(0).values), 1.5*max(data['y'].fillna(0).values)])
    ax.legend(loc='best')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig2 = model.plot_components(pred)
    return fig, fig2


# =================== FOCUS FROM HERE ===========================

# Codes for running Streamlit
folder, folder_model = 'Anomaly_Detection', os.path.join('Prophet', 'Models')
folder_file = os.path.join(folder, 'Biofilters')
json_file = os.path.join('Prophet', 'prophet_parameters.json')
log_path = os.path.join(folder, 'log_processed.csv')
site = [i.replace('.csv','') for i in os.listdir(folder_file)]


if os.path.exists(json_file) == False:
    f = open(json_file, "w")
    f.write('{}')
    f.close()


st.set_page_config(page_title='Prophet Model', page_icon=':bar_chart', layout='wide')
st.title('Prophet (Experimental Version)')
st.sidebar.header('Select your data:')

# Add site select box
option_site = st.sidebar.selectbox('Select Site', options=list(site))
# Add biofilter select box
df = pd.read_csv(os.path.join(folder_file, f'{option_site}.csv'),
                    parse_dates=True, index_col=0)
# Resample data
df = df.resample('1H').mean().replace({0: np.nan})
biofilter = [i for i in df.columns.values if 'Biofilter' in i]
option_biofilter = st.sidebar.selectbox('Select Biofilter', options=list(biofilter))

if option_biofilter is not None:
    data = df[[option_biofilter]]
    data = data.reset_index().rename(columns={data.columns[0]:'y', 'timestamp':'ds'})
    plot_train_test([data], ['Measurement'], 'Original data')
    time_split = data['ds'][int(0.7*len(data))]
    split_date = pd.to_datetime(st.sidebar.date_input('Splitting Date', time_split))
    train_df, test_df = split_data(data, split_date)
    plot_train_test([train_df, test_df], ['Train', 'Test'], 'Train-Test Split')

    # Exploratory Data Analysis
    if st.sidebar.button("Explore Data", type="secondary"):
        fig = explore_data(data)
        st.subheader('Rotation of Biofilter by Month')
        plot_placeholder = st.empty()
        plot_placeholder.pyplot(fig)
    
    missing_data = st.sidebar.multiselect('Fill missing data',
                            ['Forward fill','Backward fill','Mean',
                            'Median','Most frequent'])

    # Compare naive models with and without dealing with missing data
    if st.sidebar.button("Compare naive models (missing data filled)", type="secondary"):
        if len(missing_data)>0:
            # Deal with missing data
            for i in range(0, len(missing_data)):
                if missing_data[i] == 'Forward fill':
                    data[missing_data[i]] = data['y'].fillna(method='ffill')
                elif missing_data[i] == 'Backward fill':
                    data[missing_data[i]] = data['y'].fillna(method='bfill')
                elif missing_data[i] == 'Mean':
                    data[missing_data[i]] = make_imputer(data['y'].values, 'mean')
                elif missing_data[i] == 'Median':
                    data[missing_data[i]] = make_imputer(data['y'].values, 'median')
                elif missing_data[i] == 'Most frequent':
                    data[missing_data[i]] = make_imputer(data['y'].values, 'most_frequent')
            
            st.subheader('Prediction comparison')
            fig = compare_models(data, split_date, missing_data)
            plot_placeholder = st.empty()
            plot_placeholder.pyplot(fig)
        else: st.subheader('Missing method(s) must be selected')

    
    # Hyperparameters tuning
    if st.sidebar.button("Tune Hyperparameters", type="secondary"):
        st.subheader('Hyperparameter Tuning')
        # # Get holidays
        # holiday = get_holidays(data)
        # st.write('Holidays in UK')
        # st.dataframe(holiday)

        # Tune model
        train_df, test_df = split_data(data, split_date)
        model, best_params = model_tunning(train_df)

        save_parameters(json_file, option_biofilter, split_date, best_params)
        if len(best_params) > 0:
            st.write(f'Best hyperparameter: {best_params}')
            st.write(f'Parameters are saved in: "{json_file}"')
        else: st.write("Hyperparameters are NOT saved because the model isn't tuned yet.")

        # Save model
        file_name = f'{option_biofilter.replace(" ", "_")}.sav'
        pickle.dump(model, open(os.path.join(folder_model, file_name), 'wb'))
        st.write(f'Tunned model is saved in: "{os.path.join(folder_model, file_name)}"')

        # Plot prediction for entire data
        fig, fig1 = compute(model, train_df, test_df, title=option_biofilter)
        st.subheader('Prediction')
        plot_placeholder = st.empty()
        plot_placeholder.pyplot(fig)
        st.subheader("Model's Components")
        plot_placeholder = st.empty()
        plot_placeholder.pyplot(fig1)

    st.sidebar.subheader(':blue[Using unsaved model (Fit -> Predict)]')
    model_radio = ["Naive model", "***Tuned model***"]
    model_type = st.sidebar.radio("What's your favorite model", model_radio,
                          captions = ["Use default hyperparameters.",
                                      "Use tuned hyperparameters."])

    if model_type == model_radio[1]:
        # Load hyperparameters
        parameters_dict = load_parameters(json_file)
        if option_biofilter in parameters_dict.keys():
            param = parameters_dict[option_biofilter]
            split_date = pd.to_datetime(param["split_date"])
            train_df, test_df = split_data(data, split_date)
            plot_train_test([train_df, test_df], ['Train', 'Test'], 'Train-Test Split (use tunned model)')
            txt = f'Hyperparameters: {param}'
            st.subheader(f":red[{txt}]")
        else:
            txt = f'{option_biofilter} is NOT in database. Model need to be tunned first.'
            st.subheader(f":red[{txt}]")

    # Check button event
    if st.sidebar.button("Predict (using unsaved model)", type="secondary"):
        data = data.rename(columns={data.columns[1]:'y', 'timestamp':'ds'})
        if model_type == model_radio[0]:
            # Split data
            train_df, test_df = split_data(data, split_date)
            # Using Naive model
            model = Prophet().fit(train_df)
            fig, fig1 = compute(model, train_df, test_df, title='Naive model')
        else:
            # Using tunned model
            model = create_model(param["best_params"]).fit(train_df)
            fig, fig1 = compute(model, train_df, test_df, title='Tunned model')
        st.subheader("Prediction")
        plot_placeholder = st.empty()
        plot_placeholder.pyplot(fig)
        st.subheader("Model's Components")
        plot_placeholder = st.empty()
        plot_placeholder.pyplot(fig1)         

    st.sidebar.subheader(':blue[Using saved model]')
    if st.sidebar.button("Predict (using saved model)", type="secondary"):
        # Load model and predict
        file_name = os.path.join(folder_model, f'{option_biofilter.replace(" ", "_")}.sav')
        if os.path.exists(file_name):
            if not os.path.exists(log_path): st.write(':red[Cannot find the log file.]')            
            df_log = read_log(log_path, option_biofilter.split()[0], option_biofilter.split()[-1])
            if df_log is None: st.write(f':red[Cannot find the log file for "{option_biofilter}".]') 
            model = pickle.load(open(file_name, 'rb'))
            fig, fig1 = plot_tunned_model(model, data, option_biofilter, df_log)
            st.subheader('Prediction using trained model')
            plot_placeholder = st.empty()
            plot_placeholder.pyplot(fig)
            st.subheader("Model's Components")
            plot_placeholder = st.empty()
            plot_placeholder.pyplot(fig1)
        else: st.write(':red[Model is not exited. Need to tune model for this site first.]')



# The vitual environment must be actived first and then run below command
# Recommend open new terminal with command promt
# streamlit run Prophet\prophet_model.py --server.port 8888
