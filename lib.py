import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import colorlover as cl
from IPython.display import HTML
colors = cl.scales['11']['div']['RdYlBu']
colors = colors[:5] + ['rgb(183,188,143)','rgb(143,188,143)'] + colors[8:]
HTML(cl.to_html(colors))




def y_smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



def go_chart(gvalues,title=None, y_smooth=False, y_savgol_filter=False):
    
    ##### chart
    if y_smooth:
        
        fig = go.Figure(data=[
            go.Scatter(
                y=y_smooth(gvalues.loc[gvalues[c].index>=min(gvalues.loc[gvalues[c].notnull()].index),c].dropna(),2),
                name=c,
                mode='lines+markers',
                line=dict(color=colors[i],width=4)
            )
            for i,c in enumerate([x for _,x in sorted(zip(gvalues.iloc[-1,-2],gvalues.columns[:-2]))][::-1])
        ])
        
    elif y_savgol_filter:
                
        fig = go.Figure(data=[
            go.Scatter(
                y=y_savgol_filter(gvalues.loc[gvalues[c].index>=min(gvalues.loc[gvalues[c].notnull()].index),c].dropna(),3,1),
                name=c,
                mode='lines+markers',
                line=dict(color=colors[i],width=4)
            )
            for i,c in enumerate([x for _,x in sorted(zip(gvalues.iloc[-1,:-2],gvalues.columns[:-2]))][::-1])
        ])
        
    else:
        
        fig = go.Figure(data=[
            go.Scatter(
                y=gvalues.loc[gvalues[c].index>=min(gvalues.loc[gvalues[c].notnull()].index),c].dropna(),
                name=c,
                mode='lines+markers',
                line=dict(color=colors[i],width=4)
            )
            for i,c in enumerate([x for _,x in sorted(zip(gvalues.iloc[-1,:-2],gvalues.columns[:-2]))][::-1])
        ])
    
    fig.update_layout(
        title=title,
        xaxis_title='days'
    )

    fig.show()
    
    
def distribution(xdata,ydata,res_print=False):

    from scipy.optimize import leastsq
    import matplotlib.dates as mdates
    from scipy.stats import norm
    import numpy as np

    fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)+p[3]
    errfunc  = lambda p, x, y: (y - fitfunc(p, x))

    init  = [3000, 30, 30, 30]
    out   = leastsq( errfunc, init, args=(xdata, ydata))
    c = out[0]

    x_axis = np.arange(0, 300, 1)
    y_axis = norm.pdf(x_axis,c[1],abs(c[2]))
    
    if res_print:
        print(r'$ A = %.3f\  \mu = %.3f\  \sigma = %.3f\ k = %.3f $' %(c[0],c[1],abs(c[2]),c[3]))        

    return xdata, fitfunc(c,xdata), x_axis, y_axis, c


def chart_distribution(y,ydata,country,ylabel,prediction=False):
    
    fig = go.Figure(data=[
        go.Bar(
            x=y.index,
            y=y,
            name='data',
            marker_color=colors[-1],
        ),
    ])
    
    if prediction:
        
        fig.add_trace(
            go.Scatter(
                x=y.index,
                y=ydata,
                name='prediction',
                line=dict(color=colors[1],width=3),
            )
        )

    fig.update_layout(
        title=r'$%s$'%country.upper(),
        yaxis_title=ylabel,
        xaxis_title='days'
    )

    fig.show()  
    
 