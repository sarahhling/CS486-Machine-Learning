#Written By Sarah Ling
# 9/27/2021

import numpy as np
import random

class KMeans_SL():
    
    def __init__(self, n_clusters=4, max_iter=1000, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter=max_iter
        self.random_state=random_state
        self.cluster_centers_ = None 
        self.labels_ = None 

    def init_centroids(self, num_features: int, input: np.ndarray) -> np.array:
        #creates and returns a centroid numpy array with the shape (number of clusters, number of features)

        #if self.random_state:
        #random.seed(self.random_state)
    
        N = self.n_clusters
        M = num_features
        
        #initialize cluster_centers to random values with n_clusters rows and num_features columns
        self.cluster_centers_= np.random.rand(N,M)
        
        #old code: I tried to create the array by using random values using the lower and upper bounds of the input
        #however, the cluster centers did not match as well as just using random.rand(N,M)
        #for i in range(N):
        #    col = []
        #    for j in range(M):
        #        col.append(np.random.randint(min(input[:,j]), max(input[:,j])))
        #    self.cluster_centers_.append(col)
        #self.cluster_centers_ = np.array(self.cluster_centers_)
        
        return self.cluster_centers_

    def calculate_distance(self, d_features, c_features) -> int:
        #calculates and returns the euclidian distance between two arrays of the same length
        
        return np.sum((d_features - c_features)**2)

    def recenter_centroids(self, input: np.array) -> None:
        #recenters centroids by finding the mean column value of each datapoint associated with each
        #cluster center and setting it to the coordinates of the cluster center.
        
        for row in range(len(self.cluster_centers_)):
            #create a new list that specifies indices of labels_ where value is equal to cluster_centers_ index
            cluster_indices = np.where(self.labels_ == row)[0]
            #if cluster center has no datapoints associated with it, no change is made.
            if len(cluster_indices) > 0:    
                self.cluster_centers_[row] = input[cluster_indices].mean(axis=0)
                    
    def group_clusters(self, input: np.ndarray) -> None:
        #group clusters by matching the datapoints to the cluster center. The distance between datapoint in the input
        #and each cluster_center_ will be calculated with calculate_distance(), and the datapoint will be associated
        #with the cluster center with the smallest distance. The association is created by changing the value of the
        #label_ at the index of the datapoint to the index of the cluster.
        
        #matches datapoints with clusters
        for row in range(np.shape(input)[0]):

            #compare datapoint distance to every centroid, will associate with centroid with smallest distance

            smallest_distance = self.calculate_distance(np.array(input[row,:]),self.cluster_centers_[0])
            for cluster in range(self.n_clusters):
                if self.calculate_distance(np.array(input[row,:]),self.cluster_centers_[cluster]) < smallest_distance:
                    smallest_distance = self.calculate_distance(np.array(input[row,:]),self.cluster_centers_[cluster])
                    self.labels_[row] = cluster
                    #datapoint associates with cluster by setting the label index to cluster index
                    
    def fit(self, input: np.ndarray) -> np.array: 
        #initializes cluster_centers. Groups datapoints to clusters and calculates their average distance to
        #recenter centroids. The grouping and recentering will be called max_iter number of times or until
        #centroids no longer recenter.
        
        #gives number of features from column length of input array
        num_features = np.shape(input)[1]

        #create an empty label list that is same length as input dataset. Each element represents 
        #the index of cluster that the specific row from input is connected to
        self.labels_ = np.array([0] * np.shape(input)[0])

        #initialize cluster centers to random values
        self.cluster_centers_ = self.init_centroids(num_features, input)
        for i in range(self.max_iter):
            old_cluster_centers = self.cluster_centers_.copy()

            #recenters centroids using mean coordinates of its associated datapoints
            self.group_clusters(input)
            self.recenter_centroids(input)
            
            #will break from loop if cluster_centers_ does not change after recentering
            #Reduce time efficiency of program
            if np.array_equal(old_cluster_centers, self.cluster_centers_):
                break

        return self.cluster_centers_
