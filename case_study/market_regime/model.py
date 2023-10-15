# Model libraries
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap


class HighDimensionalClustering:

    '''
    Clustering on High Dimensional Data
    1. Dimension Reduction (TSNE or UMAP)
    2. Clustering (kMeans)


    reducer_name: TSNE or UMAP
    '''

    def __init__(self, reducer_name,dimension,Nneighbor,clustering_model_name, Ncluster):

        # dimension reduction
        self.reducer_name = reducer_name 
        self.reducer = None
        self.dimension = dimension
        self.Nneighbor = Nneighbor
        
        self.low_dimension_embedding = None
        
        # clustering
        self.clustering_model_name = clustering_model_name
        self.clustering_model = None
        self.Ncluster = Ncluster
        
        self.labels = None
   

        if self.reducer_name == 'TSNE':
            self.reducer = TSNE(
            n_components=self.dimension,
            perplexity=self.Nneighbor,
            )

        elif self.reducer_name == 'UMAP':
            self.reducer = umap.UMAP(
            n_components=self.dimension,
            n_neighbors=self.Nneighbor,
            min_dist=0.0,
            )        
        else:
            raise ValueError('Invalid reducer name')


        if self.clustering_model_name == 'kMeans':
            self.clustering_model = KMeans(
                                init="random",
                                n_clusters=self.Ncluster,
                                n_init=30,
                                max_iter=300,
                                #random_state=0
                                )
        else:
            raise ValueError('Invalid cluster name')
    

    def clustering(self, data):

        '''
        data: hign dimensional dataset, market_data
        '''
        # dimension reduction
        low_dimension_embedding = self.reducer.fit_transform(data)
        self.low_dimension_embedding = low_dimension_embedding


        # clustering
        labels = self.clustering_model.fit_predict(self.low_dimension_embedding)
        self.labels = labels
    

    

