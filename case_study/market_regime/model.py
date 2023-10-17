# Model libraries
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


class HighDimensionalClustering:

    '''
    Clustering on High Dimensional Data
    1. Dimension Reduction (TSNE or UMAP or PCA)
    2. Clustering (kMeans, DBSCAN)


    reducer_name: TSNE or UMAP or PCA
    '''

    def __init__(self, reducer_name,dimension,Nneighbor,clustering_model_name, Ncluster = None, eps = None):

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
        self.eps = eps
        
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

        elif self.reducer_name == 'PCA':
            self.reducer = PCA(n_components=self.dimension)

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
        
        elif self.clustering_model_name == 'DBSCAN':
            self.clustering_model = DBSCAN(
                                eps=self.eps, # The maximum distance between two samples for one to be considered as in the neighborhood of the other.
                                min_samples=10,
                                metric="euclidean",
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
    

    

