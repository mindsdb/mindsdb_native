import plotly.graph_objects as go


def plotter(time, real, predicted, confa=None, confb=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time,y=confa,
                             name = 'confidence',
                             fill=None,
                             mode='lines',
                             line = dict(color='#919EA5', width=0 )))

    fig.add_trace(go.Scatter(x=time,y=confb,
                             name='confidence',
                             fill='tonexty',
                             mode='lines',
                             line = dict(color='#919EA5', width=0 )))\

    fig.add_trace(go.Scatter(x=time,y=real,
                             mode='lines',
                             name='real',
                             line = dict(color='rgba(0,176,109,1)', width=3)))

    fig.add_trace(go.Scatter(x=time,y=predicted,
                             name='predicted',
                             mode='lines',
                             line = dict(color='rgba(103,81,173,1)', width=3 )))

    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            gridwidth=1,
            gridcolor='rgb(232,232,232)',
            linecolor='rgb(181, 181, 181)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Source Sans Pro',
                size=14,
                color='rgb(44, 38, 63)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            linecolor='rgb(181, 181, 181)',
                    linewidth=2,

            showticklabels=True,
            gridwidth=1,
            gridcolor='rgb(232,232,232)',
            tickfont=dict(
                family='Source Sans Pro',
                size=14,
                color='rgb(44, 38, 63)',
            ),

        ),
        autosize=True,
        margin=dict(
            autoexpand=False,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=False,
        plot_bgcolor='white',
        hovermode = 'x',
    )

    fig.show()
