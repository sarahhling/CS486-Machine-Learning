#Author: Sarah Ling

import numpy as np

class KNN_SL:

	def __init__(self, n_neighbors, random_state=None):
		self.n_neighbors = n_neighbors
		self.random_state = random_state
		self.train_labels_ = None
		self.predict_labels = list()
		self.train_features_ = None


	def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        # Saves train labels and features for later use
        
		self.train_labels_ = labels.tolist()
		self.train_features_ = features

	def calculate_distance(self, d_features, c_features) -> int:
		# Calculates and returns the euclidian distance between two points with the same number of features
        
		return np.sum((d_features - c_features)**2)
    
	def predict(self, features: np.ndarray) -> np.array:
		# Predicts the labels for the input features given the training instances.

		for test_num in range(features.shape[0]):
			distances = []
			target = {}
			for train_num in range(len(self.train_features_)):
				distances.append([train_num, self.calculate_distance(features[test_num], self.train_features_[train_num])])
           
			# Sort the values
			distances.sort(key=lambda x:x[1])
            
            # get only the first "k" number of entries
			k_distances = distances[:self.n_neighbors]
            
            # Use dictionary keep keep track of the number of rows in k_distances that have a certain train_y label
			for item in k_distances:
				if self.train_labels_[item[0]] not in target:
					target[self.train_labels_[item[0]]] = 0
				else:
					target[self.train_labels_[item[0]]] += 1
             
            #add label associated with the most rows into predict_labels
			self.predict_labels.append(max(target, key=target.get))

		return self.predict_labels        