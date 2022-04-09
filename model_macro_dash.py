# import required packages
import pandas as pd 
import dash
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go 
import numpy as np 
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import wbgapi as wb
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, tree, neighbors
import plotly.figure_factory as ff

mpl.style.use(['ggplot'])
pd.set_option('display.max_rows',300)
pd.set_option('display.max_columns',25)

# I- READ DATA INTO PANDAS DATAFRAME

# Countries and associated regions
countries=wb.economy.DataFrame(labels=True)

#1- function get dataframe
def dataframe(indicator):

    df=wb.data.DataFrame(indicator,['all'], mrv=20,labels=True,skipAggs=True)
    #add region inside of dataframe
    df['region']=countries[['region']]  # by index id to dataframe have 266 elements
    df.insert(1,'Region',df['region']) #put region after countries
    df.drop(['region'],axis=1,inplace=True)
    df=df.reset_index().set_index('Country')
    df.rename(columns={i:j for i,j in zip (df.columns[2:],list(map(str,range(2001,2021,1))))}, inplace=True)
    return df

df_gdp=dataframe('NY.GDP.MKTP.CD')
df_gdp_growth=dataframe('NY.GDP.MKTP.KD.ZG')
df_inflation=dataframe('FP.CPI.TOTL.ZG')
df_unemployment=dataframe(  'SL.UEM.TOTL.ZS')

# 2- data wrangling

# 2.1 delete missing values

years=df_gdp.columns[2:] # create list years from 2001 to 2020

#find countries which aren't provided data 
countries_miss=set() # list with unique values for countries
def country_missing(df):
    
    # for countries with a bunch of missing values
        #replace nan value to mean value
    df[years]=df[years].apply(lambda row: row.fillna(row.mean()), axis=1)
   
    # get the list of countries without any data   
    df_miss=df[df[years].isnull().all(axis=1)] 
    countries_miss.update(df_miss.index)
    return countries_miss

country_missing(df_gdp)
country_missing(df_gdp_growth)
country_missing(df_unemployment)
country_missing(df_inflation)
list_countries_miss=list(countries_miss)

# delete countries with missing values
for df in [df_gdp,df_gdp_growth,df_inflation, df_unemployment]:
    df.drop(labels=list_countries_miss, axis=0, inplace=True)

# convert data of gdp from usd to billion usd
df_gdp[years]=df_gdp[years]/100000000

# create bar chart race of gdp 

df_bar_chart=df_gdp[years].transpose()
df=df_bar_chart.unstack().to_frame().reset_index()
df.rename(columns={'level_1':'year',0:'precent'}, inplace=True)


dict_keys=['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen',
           'fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty']


# create new dictionary
n_frame={}

for y, d in zip(years, dict_keys):
    dataframe=df[(df['year']==y)]
    dataframe=dataframe.nlargest(n=5,columns=['precent'])
    dataframe=dataframe.sort_values(by=['year','precent'])
    n_frame[d]=dataframe

#-------------------------------------------
fig01 = go.Figure(
    data=[
        go.Bar(
        x=n_frame['one']['precent'], y=n_frame['one']['Country'],orientation='h',
        text=n_frame['one']['precent'], texttemplate='%{text:.3s}',
        textfont={'size':18}, textposition='inside', insidetextanchor='middle',
        width=0.9, marker_color=['#9d0208','#ffdd00','#ccff33','#006400','#03045e'])
    ],
    layout=go.Layout(
        xaxis=dict(range=[0, 250000], autorange=False, title=dict(text='precent',font=dict(size=18))),
        yaxis=dict(range=[-0.5, 5.5], autorange=False,tickfont=dict(size=14)),
        title=dict(text='GDP in Billions USD 2001',font=dict(size=28),x=0.5,xanchor='right'),
        # Add button
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                        #   , style={'margin-bottom': '10px', 'margin-right': '10px'}
                          args=[None,
                          {"frame": {"duration": 350, "redraw": True},
                          "transition": {"duration":100,
                          "easing": "linear"}}]
            )]
        )]
    ),
    frames=[
            go.Frame(
                data=[
                        go.Bar(x=value['precent'], y=value['Country'],
                        orientation='h',text=value['precent'])
                    ],
                layout=go.Layout(
                        xaxis=dict(range=[0, 250000], autorange=False),
                        yaxis=dict(range=[-0.45, 4.75], autorange=False,tickfont=dict(size=14)),
                        title=dict(text='GDP in Billions USD: '+str(value['year'].values[0]),
                        font=dict(size=28))
                    )
            )
        for key, value in n_frame.items()
    ]
)
#____________________________________________

# combine inside on data frame

def data_melt(df,text):
    df=df.reset_index()
    df=pd.melt(df, id_vars=['Country','economy','Region'])
    df.rename(columns={'variable':'year','value':f'{text}'},inplace=True)
    return df

df_gdp_m=data_melt(df_gdp, 'gdp')
df_gdp_growth=data_melt(df_gdp_growth, 'growth')
df_inflation=data_melt(df_inflation,'inflation')
df_unemployment=data_melt(df_unemployment, 'unemployment')
df_all=df_gdp_m
df_all['growth']=df_gdp_growth.growth
df_all['inflation']=df_inflation.inflation
df_all['unemployment']=df_unemployment.unemployment


#II- VISUALIZATION
# create a dash application
app=Dash(__name__)

app.layout= html.Div(children=[
    html.H1(children='Моделирование взаимосвязей макроэкономических процессов в странах мира', style={'textAlign':'center', 'color':'Green', 'font-size':50}),
    


    #segment 1: bar chart race
    html.Div([
        html.H2(children='I- GDP in billion of country from 2000 in 2020'),
        dcc.Graph(figure=fig01),
        html.Br()
    ]),

    #segment 2: sunburst and map 
    html.Div([
        html.H2('II- GDP of Region And Country'),
        html.Div([
            html.Div(dcc.Graph(id='map')),
            html.Div(dcc.Graph(id='sunburst'))            
        ], style={'display':'flex'}),
        dcc.Slider(int(years.min()),(years.max()),1, value=int(years.max()), id='slider',
        marks={str(year): str(year) for year in years}
        )
    ]),
    
    # segment 3: 3 bar chart of countries
    html.Div(
        [
            html.H2(id='title_of_chart',style={'textAlign':'center', 'color':'Blue', 'font-size':30}),
            dcc.Dropdown(df_gdp.index,id='dropdown_country1',value='Canada',placeholder="Select a city"
                            , style={'width':200, 'height':50}),
            html.Br(),
            html.Div([
                        html.Div( dcc.Graph(id='gdpgrowth_chart')),
                        html.Br(),
                        html.Div( dcc.Graph(id='inflation_chart'))],style={'display':'flex'}),
            html.Br(),
            html.Div([dcc.Graph(id='unemployment_chart')],style={'float':'center','width':'60%'})
        ]),


    # segment 4:  line chart of countries 

    html.Div([
        html.H2(id='relationship_chart'),
        dcc.Dropdown(df_gdp.index, value='Vietnam',id='dropdown_country2', style={'width':'400px'}),
        dcc.Checklist(
            options={'growth':'GDP growth(%)',
                    'inflation':"Inflation rate (%)",
                    'unemployment':'Unemployment rate(%)'},
            value=['inflation'], 
            id ='indicators',
            inline =False),
        html.Div(dcc.Graph(id='relations'))
    ]),

    #segment 5: regression chart and table
    html.Div(
        [
            html.H2('V- Machine learning for analysis'),

            html.Div([
                dcc.Dropdown(df_gdp.index, value='Russian Federation',id='dropdown_country3',
                    style={'width':200, 'height':50, 'float':"center"}
                ),
                html.Br(),
                dcc.Checklist(
                    id='list_models',
                    options=['Regression','Decision Tree', 'k-NN'],
                    value='Regression',
                    inline=False,
                ),
            ], style={'display':'block'}),

            html.H2(id='Heatmap'),

            html.Div(
                [
                dash_table.DataTable(
                    id='create_table', 
                    columns=[{'name':i,'id':i} for i in ['Country','year','growth','inflation','unemployment']],
                    style_table={'height':'300px','overflowY':'auto', 'width':'600px'},
                    fixed_rows={'headers': True}),
                dcc.Graph(id='heatmap')
                ],
                style={'display':'flex'})
        ]), 

    # segment 6: model regression       
        html.Br(),
        html.H2('Model Regression'),
        html.Div([
            html.Div(dcc.Graph(id='growth_inflation')), 

            html.Div(dcc.Graph(id='growth_unemployment')),

            html.Div(dcc.Graph(id='inflation_unemployment')) 

            ], style={'display':'flex'}),

        html.H3('Multiple Regression:'),    
        html.Div([
            html.Div([dcc.Graph     (id='multi_chart_train'),
                    html.P(id='mse_train'),
                    html.P(id='rmse_train'),
                    html.P(id='r_train')]
                    ), 

            html.Div([dcc.Graph(id='multi_chart_test'),
                    html.P(id='mse_test'),
                    html.P(id='rmse_test'),
                    html.P(id='r_test')]
                    ), 

            ], style={'display':'flex'}),

    
    # segment 7: predict indicator depend on years
    html.Div(
            [
                html.H2('Analysis of forecasts of GDP growth and inflation, unemployment',style={'textAlign':'center', 'color':'Blue', 'font-size':30}),
                dcc.Dropdown(df_gdp.index,id='dropdown_country4',value='Canada',placeholder="Select a city"
                                , style={'width':200, 'height':50}),
                html.Div([
                            html.Div( dcc.Graph(id='pred_gdp')),
                            html.Div( dcc.Graph(id='pred_growth'))],style={'display':'flex'}),
                html.Br(),
                html.Div([
                            html.Div( dcc.Graph(id='pred_inflation')),
                            html.Div( dcc.Graph(id='pred_unemployment'))],style={'display':'flex'})
            ])

])


# CALLBACK 

#callback 1: sunburst and map
@app.callback(
    Output('map','figure'),
    Output('sunburst','figure'),
    Input('slider','value'))

def get_sunburst_figure(year):
    df=df_gdp.reset_index()
    fig1= px.choropleth(df, locations='economy',color=f'{year}',hover_name='Country', color_continuous_scale=px.colors.sequential.speed)

    fig2=px.sunburst(df[['Region','Country',f'{year}']],path=['Region',"Country"],values=f'{year}', color=f'{year}')
    return fig1,fig2


#callback 2: create 3 chart
@app.callback([
    Output('title_of_chart','children'),
    Output('gdpgrowth_chart','figure'),
    Output('inflation_chart','figure'),
    Output('unemployment_chart','figure'),
    Input('dropdown_country1','value')
])
def update_output(country):
    df=df_all[df_all.Country==f'{country}']

    fig1=px.bar(df,x='year',y='growth',text_auto='.2s',title='GDP growth(%)')
    fig1.update_traces(marker_color='#03045e')

    fig2=px.bar(df,x='year',y='inflation',text_auto='.2s',title='Inflation rate(%)')
    fig2.update_traces(marker_color='#ae2012')

    fig3=px.bar(df,x='year',y='unemployment',text_auto='.2s',title='Unemployment rate(%)')
    fig3.update_traces(marker_color='#606c38')

    return  f'III- Charts of {country}', fig1,fig2,fig3

# callback 3: line chart about  relationships
@app.callback(Output('relationship_chart','children'),
Output('relations','figure'),
Input('dropdown_country2','value'),
Input('indicators','value')
)
def get_graph_rel(country, indicator):

    df=df_all[df_all.Country==f'{country}']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df['year'],y = df['growth'],mode='lines+markers',name = f'GDP Growth of {country}(%)', visible = ('growth'in indicator)))

    fig.add_trace(go.Scatter(x = df['year'],y = df['inflation'],mode='lines+markers',name = f'Inflation rate of {country}(%)', visible = ('inflation' in indicator)))

    fig.add_trace(go.Scatter(x = df['year'],y = df['unemployment'],mode='lines+markers',name = f'Unemployment rate {country}(%)', visible = ('unemployment' in indicator)))

    fig.update_layout(title=f'Chart about relationships of {country}',
                   xaxis_title='Year',
                   yaxis_title=f'rate (%)',
                   showlegend=True,
                   xaxis_tickangle=-45)
    return f'IV- Visualizing Relationships between Global Indicators of {country}', fig

# callback 4: heatmap
@app.callback(
    Output('Heatmap','children'),
    Output('heatmap','figure'),
    Input('dropdown_country3','value'),
    #Input('list_model','value')
)
def get_models(country):
    df=df_all[df_all.Country==f'{country}']
    df=df.drop('gdp', axis=1)
    df_corr=df.corr()
    fig=px.imshow(df_corr, text_auto='.3f',color_continuous_scale='ylgn')

    return f'Heatmap about correlation of {country}',fig

# callback 5: Simple Linear Regression 
@app.callback(
    Output('growth_inflation','figure'),
    Output('growth_unemployment','figure'),
    Output('inflation_unemployment','figure'),
    Output('create_table','data'),

    Input('dropdown_country3','value'),
)
def get_models(country):
    df=df_all[df_all.Country==f'{country}']

    fig1 = px.scatter(
    df, x='inflation', y='growth', opacity=0.65,
    trendline='ols', trendline_color_override='darkblue',width=450, height=350)

    fig2 = px.scatter(
    df, x='unemployment', y='growth', opacity=0.65,
    trendline='ols', trendline_color_override='red',width=450, height=350)

    fig3 = px.scatter(
    df, x='unemployment', y='inflation', opacity=0.65,
    trendline='ols', trendline_color_override='#bb3e03',width=450, height=350)

    return fig1, fig2, fig3, df[['Country','year','growth','inflation','unemployment']].to_dict('records')

#callback 5: multiple Linear Regression
@app.callback(
    Output('multi_chart_train','figure'),
    Output('multi_chart_test','figure'),

    Output('mse_train','children'),
    Output('rmse_train','children'),
    Output('r_train','children'),
    
    Output('mse_test','children'),
    Output('rmse_test','children'),
    Output('r_test','children'),

    Input('dropdown_country3','value'),
)
def machinelearning(country):
    df=df_all[df_all.Country==f'{country}']
    df.year.astype('float')

    X_train, X_test, y_train, y_test= train_test_split(df[['year','inflation','unemployment']],df['growth'], test_size=0.2, random_state=42)

    def plot_model(x,y):
        model=linear_model.LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        y_actual = y

        y_actual=y_actual.reset_index()
        df_model=pd.DataFrame(y_pred)
        y_actual['predict']=df_model[0]

        mse= mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        rsq = model.score(x,y)

        dft = y_actual[['growth','predict']]
        fig = ff.create_distplot([dft[c] for c in dft.columns], dft.columns, bin_size=0.5)

        return fig,f'MSE={mse}',f'RMSE={rmse}',f'R^2={rsq}'
    fig1,mse1,rmse1,rsq1=plot_model(X_train,y_train)
    fig2,mse2,rmse2,rsq2=plot_model(X_test,y_test)

    return fig1, fig2, mse1,rmse1,rsq1,mse2,rmse2,rsq2


@app.callback(
    Output('pred_gdp','figure'),
    Output('pred_growth','figure'),
    Output('pred_inflation','figure'),
    Output('pred_unemployment','figure'),

    Input('dropdown_country4','value'),
)
def predict_indicator(country):

    def get_model(indicator):
        df=df_all[df_all.Country==f'{country}']
        df.year=df.year.astype('float')
        X = df.year.values.reshape(-1, 1)
        y=df[f'{indicator}']

        model = linear_model.LinearRegression()          
        model.fit(X, y)

        x_range = np.linspace(X.min(), 2025, 25)
        y_range = model.predict(x_range.reshape(-1, 1))
        y_hat=model.predict(X)

        R=model.score(X, y)
        mse=mean_squared_error(y,y_hat)

        fig = px.scatter(df, x='year', y=f'{indicator}')
        fig.add_traces(go.Scatter(x=x_range, y=y_range, name=f'Predict R^2={R:.2f} \n MSE={mse:.2f}'))
        
        fig.update_layout(title=f'Predict {indicator.capitalize()} of {country}',
                    xaxis_title='Year',
                    yaxis_title=f'rate (%)',
                    showlegend=True,
                    xaxis_tickangle=-45, width=630, height=400)
        return fig
    fig1=get_model('gdp')
    fig2=get_model('growth')
    fig3=get_model('inflation')
    fig4=get_model('unemployment')

    return fig1, fig2, fig3, fig4






if __name__ == '__main__':
    app.run_server(debug=False)