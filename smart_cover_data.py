import pandas as pd

sc_csv = r'C:\Users\C21252\PycharmProjects\Anomaly_Attention\Data\Smart_cover\SmartCover.csv'
sc_df = pd.read_csv(sc_csv)

# get the data for SiteID no. 1
sc_df = sc_df[sc_df['SiteID'] == 33214]

#resample the data to 10 minutes
sc_df['Timestamp'] = pd.to_datetime(sc_df['Timestamp'])
sc_df = sc_df.set_index('Timestamp')
sc_df = sc_df.resample('10T').mean()
sc_df = sc_df.reset_index()

#remove the SiteID column
sc_df = sc_df.drop(['SiteID'], axis=1)

#fill the missing values with the forward fill method and backward fill method
sc_df = sc_df.fillna(method='ffill')
sc_df = sc_df.fillna(method='bfill')

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
sc_df_train.to_csv(r'C:\Users\C21252\PycharmProjects\Anomaly_Attention\Data\Smart_cover\SmartCover_train_33214.csv', index=False)
sc_df_test.to_csv(r'C:\Users\C21252\PycharmProjects\Anomaly_Attention\Data\Smart_cover\SmartCover_test_33214.csv', index=False)



