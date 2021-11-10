import numpy as np

class KNN_SL:

	def __init__(self, n_neighbors, random_state=None):
		self.n_neighbors = n_neighbors
		self.random_state = random_state
		self.train_labels_ = None
		self.predict_labels = list()
		self.train_features_ = None


	def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
		'''
		Create the KNN model (or not)
		'''
		self.train_labels_ = labels.tolist()
		self.train_features_ = features

	def calculate_distance(self, d_features, c_features) -> int:
		#calculates and returns the euclidian distance between two points with the same number of features
		return np.sum((d_features - c_features)**2)
    
	def predict(self, features: np.ndarray) -> np.array:
		'''
		Predict the labels for the input features given the
		training instances.
		'''

		for test_num in range(features.shape[0]):
			distances = {}
			target = {}
			for train_num in range(len(self.train_features_)):
				distances[train_num] = self.calculate_distance(features[test_num], self.train_features_[train_num])
           
			sorted_distances = sorted(distances.values()) # Sort the values
            
            #put sorted values back in dictionary
			sorted_distances_dict = {} 
			for i in sorted_distances:
				for key in distances.keys():
					if distances[key] == i:
						sorted_distances_dict[key] = distances[key]
						break

            #convert sorted dictionary to list, and get only the first "k" number of entries
			k_distances = list(sorted_distances_dict.items())[:self.n_neighbors]
			for item in k_distances:
				if self.train_labels_[item[0]] not in target:
					target[self.train_labels_[item[0]]] = 0
				else:
					target[self.train_labels_[item[0]]] += 1
                    
			self.predict_labels.append(max(target, key=target.get))

		return self.predict_labels        
            
       





