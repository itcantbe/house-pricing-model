import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np


def histogramPlot(df,field,show=False):
    fig = px.histogram(df,x=field)
    pio.write_image(fig, 'reports/histogram_'+field+'.svg', format='svg')
    if(show):
        fig.show()

def boxPlot(df,field,show=False):
    fig = px.box(df,y=field,points="all")
    pio.write_image(fig, 'reports/box_'+field+'.svg', format='svg')
    if(show):
        fig.show()

def scatterWRTvar(df,field,primary="median_house_value",show=False):
    fig = px.scatter(df.sort_values(by=primary),x=primary,y=field)
    pio.write_image(fig, 'reports/scatter_'+field+'.svg', format='svg')
    if(show):
        fig.show()

def boxCompWRTvar(df,field,primary="median_house_value",isContinous = False,show=False):
    if(isContinous):
        val = pd.cut(df[field], bins=50, labels=False, include_lowest=True)
        fig = px.box(df,x=val, y=primary,title=field)   
        pio.write_image(fig, 'reports/boxComp_'+field+'.svg', format='svg')
        if(show):
            fig.show()
    else:
        fig = px.box(df,x=field,y=primary,title=field)
        pio.write_image(fig, 'reports/boxComp_'+field+'.svg', format='svg')
        if(show):
            fig.show()
    
def makeScatterMatrix(df,show=False):
    fig = px.scatter_matrix(df,width=1500,height=1500)
    fig.update_traces(diagonal_visible=False,showupperhalf=False)
    fig.update_layout(
        margin=dict(l=50, r=50, t=80, b=50), # Adjust margins if needed
        title_font_size=20 # Adjust title font size
    )
    for i in fig['layout']:
        if i.startswith('xaxis') or i.startswith('yaxis'):
            fig['layout'][i]['automargin'] = True # Let Plotly adjust margins for labels
            fig['layout'][i]['tickfont'] = dict(size=8) # Reduce tick label font size
            # fig['layout'][i]['showticklabels'] = False # Uncomment to hide tick labels
            # fig['layout'][i]['title']['font'] = dict(size=10) # Adjust axis title font size

    pio.write_image(fig, 'reports/scatter_matrix.svg', format='svg')
    if(show):
        fig.show()

def scatterLogPlot(df,primary,secondary,show=False):
    sec = np.log(secondary)

    fig = px.scatter(df, x=primary,y=sec)
    pio.write_image(fig, 'reports/scatter_income_and_price_log_data.svg', format='svg')
    if(show):
        fig.show()

def histroLogPlot(secondary,show=False):
    sec = np.log(secondary)
    fig = px.histogram(sec)
    pio.write_image(fig, 'reports/histogram_price_log_data.svg', format='svg')
    if(show):
        fig.show()