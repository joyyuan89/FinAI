import numpy as np
import plotly.express as px

class ClusterVisualizer:

    def __init__(self, clusterable_embedding, labels):

        self.clusterable_embedding = clusterable_embedding
        self.labels = labels
        self.colors = self.labels
        self.weights = np.full_like(self.labels, fill_value=20)
        self.opacities = np.full_like(self.labels, fill_value=1) / 4

        # Highlight the last point
        self.colors[-1] = -1  # set the darkest color for this point
        self.weights[-1] = 80  # set a larger size value for this point
        self.opacities[-1] = 1  # set a higher opacity value for this point


        # Highlight the 1M
        self.colors[-21] = -1  # set the darkest color for this point
        self.weights[-21] = 60  # set a larger size value for this point
        self.opacities[-21] = 1  # set a higher opacity value for this point

        # Highlight the 3M
        self.colors[-65] = -1  # set the darkest color for this point
        self.weights[-65] = 40  # set a larger size value for this point
        self.opacities[-65] = 1  # set a higher opacity value for this point


    def plot_2d(self):
        fig = px.scatter(
            self.clusterable_embedding,
            x=0, y=1,
            title='Clustering Result and Dimension Reduction (2D)',
            color=self.colors,
            size=self.weights,
            height=700,
            width=700,
            opacity=self.opacities,
            color_continuous_scale='plasma',
            # rename axis
            labels={
                "0": "Dimension 1",
                "1": "Dimension 2",
            }
            # not show axis
        )
        fig.update_traces(
            marker=dict(line=dict(width=0, color='DarkSlateGrey')),
            selector=dict(mode='markers')
        )

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        #fig.show()

        return fig

    def plot_3d(self):
        fig = px.scatter_3d(
            self.clusterable_embedding,
            x=0, y=1, z=2,
            title='3D Plot',
            color=self.colors,
            size=self.weights,
            height=700,
            width=700,
            opacity=self.opacities,
        )
        fig.update_traces(
            marker=dict(line=dict(width=0, color='DarkSlateGrey')),
            selector=dict(mode='markers')
        )
        #fig.show()

        return fig
