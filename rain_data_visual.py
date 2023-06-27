import plotly as py
import plotly.graph_objects as go
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html





rain_csv = r'rain_data/400.csv' #change this to the path of the csv file you want to use



########################################################################################################################
rain_data = pd.read_csv(rain_csv)

# Create a scatter plot of the data
#make data show the rain event number when you hover over the data point

fig = go.Figure(data=go.Scatter(x=rain_data['Timestamp'], y=rain_data['value'], name='rainfall',
                                line_shape='hv',fill='tozeroy', line=dict(color='blue', width=1),
                                hovertemplate = 'Rain Event Number: %{text}<br>Timestamp: %{x}<br>Rainfall: %{y}',
                                text = rain_data['sig_event_number']))


#
# add a dropdown menu to select the rain event number
# Create a dictionary of dropdown options which includes the rain event number
# and the corresponding value for the dropdown menu
dropdown_options = [x for x in rain_data['sig_event_number'].unique()]


# Create the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    dcc.Dropdown(
        id='dropdown',
        options=dropdown_options,
        value='option1'
    ),
    dcc.Graph(
        id='graph',
        figure=fig
    )
])


# Define the callback function to update the figure based on the dropdown selection
@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('dropdown', 'value')]
)
def update_figure(selected_option):
    if selected_option is None:
        updated_fig = fig
    else:
     # Filter the data based on the selected options
    # Update the figure based on the selected dropdown option
        rain_data_selected = rain_data[rain_data['sig_event_number'] == selected_option]
        updated_fig = go.Figure(data=go.Scatter(x=rain_data_selected['Timestamp'], y=rain_data_selected['value'],  name='rainfall',
            line_shape='hv',
            fill='tozeroy',
            line=dict(color='blue', width=1)))

    return updated_fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port = 50)