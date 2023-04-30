import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import pandas as pd
from dash.dependencies import Output, Input
import dash_daq as daq
import plotly.express as px
import datetime
import plotly.graph_objs as go

df = pd.read_csv("aguatratadalasirena.csv")

################################################################

datos_soledad = pd.read_csv('ene_feb_mar_2021.csv')
datos_soledad = datos_soledad.rename(columns={
    'Time': 'Hora',
    'Date': 'Fecha',
    'temperatura': 'Temperatura',
    'turbiedad': 'Turbiedad',
    'oxigeno': 'Oxigeno',
    'conduct': 'Conductividad',
    'ph': 'pH',
    'Mes': 'Num_Mes',
    'Dia': 'Dia_Semana',
    'Nombre Mes': 'Mes'
})
datos_soledad['Hora_Minuto'] = datos_soledad.apply(lambda x: f"{x['Fecha']} {x['Hora']}", axis=1)
datos_numericos = datos_soledad.select_dtypes(include=['float'])
estadisticas = datos_numericos.describe()
datos_soledad['Fecha'] = pd.to_datetime(datos_soledad['Fecha'])

#################################################################

# Convierte las columnas de fecha y hora a formato datetime
df['Time'] = pd.to_datetime(df['Time'], errors='ignore')
df['Date'] = pd.to_datetime(df['Date'], errors='ignore')

# Elimina filas con valores nulos o vacíos
df.dropna(inplace=True)

# Renombra las columnas para mayor claridad
df.rename(columns={
    'AP03AT9002TEMP': 'temperatura',
    'AP03AT9002TURB': 'turbiedad',
    'AP03AT9002CL2': 'cloro',
    'AP03AT9002PH': 'ph'
}, inplace=True)

# Redondea los valores de turbiedad, cloro, temperatura y ph a 2 decimales
df['ph'] = pd.Series([round(val, 2) for val in df['ph']])
df['turbiedad'] = pd.Series([round(val, 2) for val in df['turbiedad']])
df['cloro'] = pd.Series([round(val, 3) for val in df['cloro']])
df['temperatura'] = pd.Series([round(val, 3) for val in df['temperatura']])

# Agrupa los datos por fecha y toma el valor máximo de cada variable
dff = df.groupby('Date', as_index=False)[['ph', 'turbiedad', 'cloro', 'temperatura']].max()

# Obtiene los valores máximo, mínimo, promedio y último para cada variable
dff_max = df.groupby('Date', as_index=False)[['ph', 'turbiedad', 'temperatura', 'cloro']].max()
dff2_min = df.groupby('Date', as_index=False)[['ph', 'turbiedad', 'temperatura', 'cloro']].min()
dff3_mean = df.groupby('Date', as_index=False)[['ph', 'turbiedad', 'temperatura', 'cloro']].mean()
dff4_tail = df.groupby('Date', as_index=False)[['ph', 'turbiedad', 'temperatura', 'cloro']].tail(1)

# Agrega una columna para indicar si se trata del valor máximo, mínimo, promedio o último
dff_max['Indicador'] = 'max'
dff2_min['Indicador'] = 'min'
dff3_mean['Indicador'] = 'promedio'
dff4_tail['Indicador'] = 'ultimo'

# Concatena todos los datos en un solo DataFrame
df_max_min = pd.concat([dff_max, dff2_min, dff3_mean])

# Datos daq

daq_max = pd.read_csv("datos_maximos.csv")
daq_min = pd.read_csv("datos_minimos.csv")

# Grafico de linea para Turbiedad2
colors = px.colors.qualitative.Dark24

fig_turbiedad2 = go.Figure()
fig_turbiedad2.add_trace(go.Scatter(x=datos_soledad['Fecha'], y=datos_soledad['Turbiedad'], mode='lines', line=dict(color=colors[0], width=2, dash='solid')))

for i, month in enumerate(datos_soledad['Mes'].unique()):
    month_datos_soledad = datos_soledad[datos_soledad['Mes'] == month]
    start_date_datos_soledad = month_datos_soledad['Fecha'].iloc[0]
    end_date_datos_soledad = month_datos_soledad['Fecha'].iloc[-1]
    fig_turbiedad2.add_vrect(x0=start_date_datos_soledad, x1=end_date_datos_soledad, fillcolor=colors[i % len(colors)], opacity=0.2, layer='below', line_width=0)

fig_turbiedad2.update_layout(plot_bgcolor='#f2f2f2', paper_bgcolor='#f2f2f2', xaxis=dict(gridcolor='#ffffff', linecolor='#000000', linewidth=1), yaxis=dict(gridcolor='#ffffff', linecolor='#000000', linewidth=1))
fig_turbiedad2.update_layout(margin=dict(t=20, b=50, l=50, r=50))
fig_turbiedad2.update_layout(xaxis={"showgrid": False},yaxis={"showgrid": False})

# Grafico de linea para pH2
colors = px.colors.qualitative.Dark24

fig_ph2 = go.Figure()
fig_ph2.add_trace(go.Scatter(x=datos_soledad['Fecha'], y=datos_soledad['pH'], mode='lines', line=dict(color=colors[0], width=2, dash='solid')))

for i, month in enumerate(datos_soledad['Mes'].unique()):
    month_datos_soledad = datos_soledad[datos_soledad['Mes'] == month]
    start_date_datos_soledad = month_datos_soledad['Fecha'].iloc[0]
    end_date_datos_soledad = month_datos_soledad['Fecha'].iloc[-1]
    fig_ph2.add_vrect(x0=start_date_datos_soledad, x1=end_date_datos_soledad, fillcolor=colors[i % len(colors)], opacity=0.2, layer='below', line_width=0)

fig_ph2.update_layout(plot_bgcolor='#f2f2f2', paper_bgcolor='#f2f2f2', xaxis=dict(gridcolor='#ffffff', linecolor='#000000', linewidth=1), yaxis=dict(gridcolor='#ffffff', linecolor='#000000', linewidth=1))
fig_ph2.update_layout(margin=dict(t=20, b=50, l=50, r=50))
fig_ph2.update_layout(xaxis={"showgrid": False},yaxis={"showgrid": False})

# Grafico de linea para Conductividad
colors = px.colors.qualitative.Dark24

fig_condct2 = go.Figure()
fig_condct2.add_trace(go.Scatter(x=datos_soledad['Fecha'], y=datos_soledad['Conductividad'], mode='lines', line=dict(color=colors[0], width=2, dash='solid')))

for i, month in enumerate(datos_soledad['Mes'].unique()):
    month_datos_soledad = datos_soledad[datos_soledad['Mes'] == month]
    start_date_datos_soledad = month_datos_soledad['Fecha'].iloc[0]
    end_date_datos_soledad = month_datos_soledad['Fecha'].iloc[-1]
    fig_condct2.add_vrect(x0=start_date_datos_soledad, x1=end_date_datos_soledad, fillcolor=colors[i % len(colors)], opacity=0.2, layer='below', line_width=0)

fig_condct2.update_layout(plot_bgcolor='#f2f2f2', paper_bgcolor='#f2f2f2', xaxis=dict(gridcolor='#ffffff', linecolor='#000000', linewidth=1), yaxis=dict(gridcolor='#ffffff', linecolor='#000000', linewidth=1))
fig_condct2.update_layout(margin=dict(t=20, b=50, l=50, r=50))
fig_condct2.update_layout(xaxis={"showgrid": False},yaxis={"showgrid": False})


# Gráfico de línea para Temperatura2
colors = px.colors.qualitative.Dark24

fig_tem2 = go.Figure()
fig_tem2.add_trace(go.Scatter(x=datos_soledad['Fecha'], y=datos_soledad['Temperatura'], mode='lines', line=dict(color=colors[0], width=2, dash='solid')))

for i, month in enumerate(datos_soledad['Mes'].unique()):
    month_datos_soledad = datos_soledad[datos_soledad['Mes'] == month]
    start_date_datos_soledad = month_datos_soledad['Fecha'].iloc[0]
    end_date_datos_soledad = month_datos_soledad['Fecha'].iloc[-1]
    fig_condct2.add_vrect(x0=start_date_datos_soledad, x1=end_date_datos_soledad, fillcolor=colors[i % len(colors)], opacity=0.2, layer='below', line_width=0)

fig_tem2.update_layout(plot_bgcolor='#f2f2f2', paper_bgcolor='#f2f2f2', xaxis=dict(gridcolor='#ffffff', linecolor='#000000', linewidth=1), yaxis=dict(gridcolor='#ffffff', linecolor='#000000', linewidth=1))
fig_tem2.update_layout(margin=dict(t=20, b=50, l=50, r=50))
fig_tem2.update_layout(xaxis={"showgrid": False},yaxis={"showgrid": False})


# Crear la aplicación

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Dashboard PTAP La Sirena")

# Crear el sidebar

sidebar = dbc.Container([
    html.H5('PTAP La Soledad', className='bg-primary text-white font-italic p-3'),
    dbc.Row([
        dbc.Col([
            
            dbc.Button('Turbiedad', id="btn-turb", color='primary', className='my-2', style={'width': '100%', 'border-radius': '30px'}),
            dbc.Button('Conductividad', id="btn-cond", color='primary', className='my-2', style={'width': '100%', 'border-radius': '30px'}),
            dbc.Button('Temperatura', id="btn-temp",color='primary', className='my-2', style={'width': '100%', 'border-radius': '30px'}),
            dbc.Button('pH', id="btn-ph",color='primary', className='my-2 mb-0', style={'width': '100%', 'border-radius': '30px'}),
        ], className='mt-3 align-items-center'),
    ]),
    dbc.Row([
        
        dbc.Col([
            dbc.Card([
                dbc.CardImg(id='card-image', top=True, bottom=False,
                    title="Image by Kevin Dinkel", alt='Learn Dash Bootstrap Card Component'),
                dbc.CardBody([
                    html.H4('AGUA CRUDA', className='card-title'),
                ], style={'padding-bottom': '0'})
            ], style={'margin-bottom': '0'})
        ], className='mt-3 align-items-center', style={'margin-bottom': '0',})
        
        ], className='justify-content-between'),
], className='bg-light', style={'height': '100vh', 'padding': '0', 'background-color': '#f5f5f5'}, fluid=True)

# Crear el contenido principal

content = dbc.Container([
    dbc.Row(
            [
                dbc.Col(html.Div([dbc.Alert("Elije un parametro",id="alert1", color="secondary")])),
                
                dbc.Col(html.Div(dbc.Alert("Promedio del parametro", id= 'alert2', color="danger"))),
                
            ]
        ),
    dbc.Row([dcc.Graph(id='line-chart')],),
    
    dbc.Row([
    dbc.Col([
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.Div(daq.LEDDisplay(
                id="led-max",
                label="Máximo registrado",
                value="0",
                size=48,
                color="#FF5E5E"
            )),
                    ]
                )
            ],
            className="mb-3",
        )
    ], width=3),
    dbc.Col([
        dbc.Card(
            [
                dbc.CardBody(
                    [

                        html.Div(daq.LEDDisplay(
                id="led-min",
                label="Mínimo registrado",
                value="0",
                size=48,
                color="#5E5EFF"
            )),
                    ]
                )
            ],
            className="mb-3",
        )
    ], width=3),
    dbc.Col([
        dbc.Card([dbc.CardBody([html.Div(daq.LEDDisplay(label="Numero de Horas de Medicion", value="2662", size=48, color="#00FF00")),])],
                 className="mb-3",)], width=3),
    dbc.Col([
        dbc.Card([dbc.CardBody([html.Div(daq.LEDDisplay(label="Numero de datos registrados", value="42549", size=48, color='#F7DC6F')),])],
                 className="mb-3",), 
    ], width=3)
], className='mt-3')
], fluid=True)

# Combinar el sidebar y el contenido principal

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(sidebar, width=2),
        dbc.Col(content, width=10)
    ])
], fluid=True)

# Función para actualizar el gráfico
@app.callback(
    Output('line-chart', 'figure',),
    Input('btn-ph', 'n_clicks'),
    Input('btn-turb', 'n_clicks'),
    Input('btn-cond', 'n_clicks'),
    Input('btn-temp', 'n_clicks'),
)
def update_chart(btn_ph, btn_turb, btn_cond, btn_temp):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-ph' in changed_id:
        return fig_ph2
    elif 'btn-turb' in changed_id:
        return fig_turbiedad2
    elif 'btn-clor' in changed_id:
        return fig_condct2
    elif 'btn-temp' in changed_id:
        return fig_tem2
    else:
        return fig_ph2


#DAQ

@app.callback(
    [Output("led-max", "value"), Output("led-min", "value")],
    [Input("btn-ph", "n_clicks"), Input("btn-cond", "n_clicks"), 
     Input("btn-turb", "n_clicks"), Input("btn-temp", "n_clicks")])
def update_led(button_ph, button_cond, button_turbiedad, button_temperatura):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "0", "0"
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "btn-ph":
        max_value = daq_max["ph"].values[0]
        min_value = daq_min["ph"].values[0]
    elif button_id == "btn-cond":
        max_value = daq_max["cloro"].values[0]
        min_value = daq_min["cloro"].values[0]
    elif button_id == "btn-turb":
        max_value = daq_max["turbiedad"].values[0]
        min_value = daq_min["turbiedad"].values[0]
    elif button_id == "btn-temp":
        max_value = daq_max["temperatura"].values[0]
        min_value = daq_min["temperatura"].values[0]
    return str(max_value), str(min_value)





@app.callback(
    Output("alert1", "children"),
    Input("btn-turb", "n_clicks"),
    Input("btn-cond", "n_clicks"),
    Input("btn-temp", "n_clicks"),
    Input("btn-ph", "n_clicks")
)
def update_alert(n_clicks_turb, n_clicks_temp, n_clicks_cond, n_clicks_ph):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Elije un parametro"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "btn-turb":
            return "80 veces se colapso la PTAP"
        elif button_id == "btn-cond":
            return "20 oxigeno maximo disuelto en el agua"
        elif button_id == "btn-temp":
            return " la temperatura mas fria 8 veces"
        elif button_id == "btn-ph":
            return "El pH es una muestra de la contaminacion"
        else:
            return "Elije un parametro"


@app.callback(
    Output("alert2", "children"),
    Input("btn-turb", "n_clicks"),
    Input("btn-cond", "n_clicks"),
    Input("btn-temp", "n_clicks"),
    Input("btn-ph", "n_clicks")
)
def update_alert(n_clicks_turb, n_clicks_temp, n_clicks_cond, n_clicks_ph):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Xxxxxxxxxx"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "btn-turb":
            return "xxxxxxxxxx"
        elif button_id == "btn-cond":
            return "yyyyyyyyy"
        elif button_id == "btn-temp":
            return "zzzzzzzzz"
        elif button_id == "btn-ph":
            return "wwwwwwwww"
        else:
            return "Elije un parametro"


# Define la función de callback
@app.callback(Output('card-image', 'src'),
              [Input('btn-turb', 'n_clicks'),
               Input('btn-cond', 'n_clicks'),
               Input('btn-temp', 'n_clicks'),
               Input('btn-ph', 'n_clicks')])
def update_image(n_clicks_turb, n_clicks_cond, n_clicks_temp, n_clicks_ph):
    button_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'btn-turb':
        return '/assets/turbidez_comp.jpg'
    elif button_id == 'btn-cond':
        return '/assets/oxigeno_comp.jpg'
    elif button_id == 'btn-temp':
        return '/assets/tempe_comp.jpg'
    elif button_id == 'btn-ph':
        return '/assets/ph_comp.jpg'
    else:
        return '/assets/ptap.jpeg'


# Iniciar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True, port=3000)