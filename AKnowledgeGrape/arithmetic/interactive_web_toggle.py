import dash
from dash import dcc, html
import dash_cytoscape as cyto


def run_dash_app(*views, view_titles=None):
    app = dash.Dash(__name__)

    tabs = []
    for i, elements in enumerate(views):
        title = view_titles[i] if view_titles and i < len(view_titles) else f"视图 {i + 1}"
        tabs.append(
            dcc.Tab(label=title, children=[
                cyto.Cytoscape(
                    id=f'cytoscape-{i}',
                    layout={'name': 'cose'},
                    style={'width': '100%', 'height': '600px'},
                    elements=elements,
                    stylesheet=[
                        {
                            'selector': 'node',
                            'style': {
                                'label': 'data(label)',
                                'background-color': 'data(color)',
                                'color': 'black',
                                'font-size': '12px'
                            }
                        },
                        {
                            'selector': 'edge',
                            'style': {
                                'width': 1,
                                'line-color': '#ccc',
                                'target-arrow-color': '#ccc',
                                'target-arrow-shape': 'triangle'
                            }
                        }
                    ]
                )
            ])
        )

    app.layout = html.Div([
        html.H1("学习路径可视化对比"),
        dcc.Tabs(children=tabs)
    ])

    app.run(debug=True)

