import pandas as pd
import numpy as np
import plotly.graph_objects as go









# read the csv file into a pandas dataframe
df = pd.read_csv('data_for_model/train_data.csv')
#convert the column of timestamps to datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#set the index of the dataframe to be the column of timestamps
df.set_index('Timestamp', inplace=True)




#plot the data in plotly with x-axis as the index of df and y-axis as the column of Level, and color the points by the column of pred, pred column is discrete
#higher the value of pred, especially when pred is 1, the more likely the point is an anomaly
#making pred=0 points gray and opque, pred=1 points red
fig = go.Figure(data=go.Scatter(x=df.index, y=df['Level'], mode='markers'))
fig.update_layout(title='Anomaly Detection Result', xaxis_title='Timestamp', yaxis_title='Level')

#plot the column Rain on the top of the previous plot using the same x axis but to a different y axis,
#this is to show the relationship between the column Rain and the column Level
fig.add_trace(go.Scatter(x=df.index, y=df['Rain'], mode='lines', name='Rain', yaxis='y2'))
fig.update_layout(yaxis2=dict(title='Rain', overlaying='y', side='right'))



#starting from each pred =1 point, draw a vertical rectangular whose width is about 100 data points and heigh is infinite
#this is to show the time period of each anomaly
# for i in range(len(df)):
#     if df['pred'][i] == 1:
#         fig.add_shape(type='rect', x0=df.index[i], y0=-1, x1=df.index[i+100], y1=1, line=dict(color='Red', width=2), fillcolor='Red', opacity=0.2)



fig.show()

anomalies_test = ['2023-06-08 11:10','2023-06-09 18:20', '2023-06-10 13:20' ]
anomalies_train = ['2023-03-22 16:30', '2023-03-23 16:40', '2023-03-08 18:40', '2023-03-26 19:40', '2023-03-27 16:40',
                   '2023-03-28 19:00', '2023-03-29 18:20', '2023-03-31 14:20', '2023-03-31 15:30', '2023-04-05 10:40',
                   '2023-04-14 08:40', '2023-04-14 08:50', '2023-04-18 15:50', '2023-05-04 15:10', '2023-06-02 14:00',
                   '2023-05-10 07:30', '2023-05-10 07:40', '2023-05-10 07:50', '2023-05-10 08:00', '2023-05-10 08:10',
                   '2023-05-10 08:20', '2023-05-10 08:30', '2023-05-10 08:40', '2023-05-10 08:50',
                   '2023-05-14 08:00', '2023-05-14 08:10', '2023-05-14 08:20', '2023-05-14 08:30', '2023-05-14 08:40',
                   '2023-05-14 08:50', '2023-05-14 09:00', '2023-05-14 09:10', '2023-05-14 09:20', '2023-05-14 09:30',
                   '2023-05-14 09:40', '2023-05-14 09:50', '2023-05-14 10:00', '2023-05-14 10:10', '2023-05-14 10:20',
                   '2023-05-14 10:30', '2023-05-14 10:40', '2023-05-14 10:50', '2023-05-14 11:00', '2023-05-14 11:10',
                   '2023-05-14 11:20', '2023-05-14 11:30', '2023-05-14 11:40', '2023-05-14 11:50', '2023-05-14 12:00',
                   '2023-05-14 12:10', '2023-05-14 12:20', '2023-05-14 12:30', '2023-05-14 12:40', '2023-05-14 12:50',
                   '2023-05-14 13:00', '2023-05-14 13:10', '2023-05-14 13:20', '2023-05-14 13:30', '2023-05-14 13:40',
                   '2023-05-14 13:50', '2023-05-14 14:00', '2023-05-14 14:10', '2023-05-14 14:20', '2023-05-14 14:30',
                   '2023-05-14 14:40', '2023-05-14 14:50', '2023-05-14 15:00', '2023-05-14 15:10', '2023-05-14 15:20',
                   '2023-05-14 15:30', '2023-05-14 15:40', '2023-05-14 15:50', '2023-05-14 16:00', '2023-05-14 16:10',
                   '2023-05-14 16:20', '2023-05-14 16:30', '2023-05-14 16:40', '2023-05-14 16:50', '2023-05-14 17:00',
                   '2023-05-14 17:10', '2023-05-14 17:20', '2023-05-14 17:30', '2023-05-14 17:40', '2023-05-14 17:50']