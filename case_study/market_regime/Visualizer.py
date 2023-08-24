import numpy as np
import plotly.express as px

class KMeansVisualizer:

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

    def plot_2d(self):
        fig = px.scatter(
            self.clusterable_embedding,
            x=0, y=1,
            title='2D Plot',
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
        fig.show()

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
        fig.show()
