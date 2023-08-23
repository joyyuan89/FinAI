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

    def __init__(self, reducer_name,dimension,Nneighbor,cluster_name, Ncluster):

        # dimension reduction
        self.reducer_name = reducer_name 
        self.reducer = None
        self.dimension = dimension
        self.Nneighbor = Nneighbor
        
        self.low_dimension_embedding = None
        
        # clustering
        self.clustering_model_name = clustering_model_name
        self.clutering_model = None
        self.Ncluster = Ncluster
   

        if self.reducer_name == 'TSNE':
            self.reducer = TSNE
        elif self.reducer_name == 'UMAP':
            self.reducer = umap.UMAP        
        else:
            raise ValueError('Invalid reducer name')
        

        if self.cluster_name == 'kMeans':
            self.cluter_model = KMeans(
                                init="random",
                                n_clusters=self.Ncluster,
                                n_init=30,
                                max_iter=300,
                                #random_state=0
                                )
        else:
            raise ValueError('Invalid cluster name')


    def embedding(self, data):

        '''
        data: hign dimensional dataset, market_data
        '''

        low_dimension_embedding = self.reducer(
                                n_components=self.dimension,
                                perplexity=self.Nneighbor,
                                ).fit_transform(data)
    
        self.low_dimension_embedding = low_dimension_embedding

        return low_dimension_embedding
    

    def clustering(self, data):

        '''
        data: low dimensional dataset, low_dimension_embedding
        '''
        if self.clustering_model_name == 'kMeans':
            labels = self.clustering_model.fit_predict(self.low_dimension_embedding)
        
        # add labels to the data
        df_labeled = data.copy()
        df_labeled['label'] = labels
       
        return labels
    

