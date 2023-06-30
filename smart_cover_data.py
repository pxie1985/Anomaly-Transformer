import pandas as pd

sc_csv = r'C:\Users\C21252\PycharmProjects\Anomaly_Attention\Data\Smart_cover\SmartCover.csv'
sc_df = pd.read_csv(sc_csv)

# get the data for SiteID no. 1
sc_df = sc_df[sc_df['SiteID'] == 33214]

#resample the data to 10 minutes
sc_df = sc_df.set_index('Timestamp')
#parse the timestamp column to datetime
sc_df.index = pd.to_datetime(sc_df.index)

#resample the data to 1 hour
sc_df = sc_df.resample('1H').mean()


#remove the SiteID column
sc_df = sc_df.drop(['SiteID'], axis=1)

#fill the missing values with the forward fill method and backward fill method
sc_df = sc_df.fillna(method='ffill')
sc_df = sc_df.fillna(method='bfill')

#create columns to indicate the day of the week and the hour of the day, month of the year and year
sc_df['day_of_week'] = sc_df.index.dayofweek
sc_df['hour_of_day'] = sc_df.index.hour
sc_df['month_of_year'] = sc_df.index.month
sc_df['year'] = sc_df.index.year



#read the rain data
rain_csv = r'C:\Users\C21252\PycharmProjects\Anomaly_Attention\Data\Smart_cover\rain_for_model.csv'
rain_df = pd.read_csv(rain_csv)
#parse the timestamp column to datetime
rain_df['Timestamp'] = pd.to_datetime(rain_df['Timestamp'])
#drop the GaugeID column
rain_df = rain_df.drop(['GaugeID'], axis=1)
#set the timestamp column as index
rain_df = rain_df.set_index('Timestamp')

#resample the data to 1 hour
rain_df = rain_df.resample('1H').sum()

#join the rain data with the smart cover data
sc_df = sc_df.join(rain_df, how='left')
#fill the missing values with the forward fill method and backward fill method
sc_df = sc_df.fillna(method='ffill')
sc_df = sc_df.fillna(method='bfill')
#reset the index
sc_df = sc_df.reset_index()


#making the first 0.8 as training data and the rest as test data
sc_df['Label'] = 0
sc_df.loc[0:sc_df.shape[0]*0.8, 'Label'] = 1

#export the train and testing data
sc_df_train = sc_df[sc_df['Label'] == 1]
sc_df_test = sc_df[sc_df['Label'] == 0]

#remove the label column
sc_df_train = sc_df_train.drop(['Label'], axis=1)
sc_df_test = sc_df_test.drop(['Label'], axis=1)

#expoert the train and test data
sc_df_train.to_csv(r'C:\Users\C21252\PycharmProjects\Anomaly_Attention\Data\Smart_cover\hourly\SmartCover_train_33214_hourly.csv', index=False)
sc_df_test.to_csv(r'C:\Users\C21252\PycharmProjects\Anomaly_Attention\Data\Smart_cover\hourly\SmartCover_test_33214_hourly.csv', index=False)


#plot the sc_df with plotly
import plotly.graph_objects as go
import plotly.express as px

fig = go.Figure()
fig.add_trace(go.Scatter(x=sc_df_train['Timestamp'], y=sc_df_train['Level'],
                    mode='lines',
                    name='lines'))
fig.show()

fig = px.line(sc_df_train, x='Timestamp', y='level')
fig.show()


