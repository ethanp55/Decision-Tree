import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

class DTClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,counts=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            counts = how many types for each attribute
        Example:
            DT  = DTClassifier()
        """
        self.counts = counts

    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        feature_names = np.arange(np.shape(X)[1])

        self.tree = self._make_tree(X, y, feature_names)

        return self

    def _make_tree(self, X, y, feature_names):
        n_instances = np.shape(X)[0]
        n_features = len(feature_names)

        unique, counts = np.unique(y, return_counts=True)
        unique_map = dict(zip(unique, counts))
        max_key = max(unique_map, key=unique_map.get)

        # Base case when there are no more instances or features/attributes
        if n_instances == 0 or n_features == 0:
            # Return the most frequent class
            return max_key

        # Base case when there is only 1 class/target left
        elif unique_map[y[0][0]] == n_instances:
            # Return that 1 class
            return y[0][0]

        # Normal case
        else:
            # Find the feature that gives the most information gain
            gains = [0] * n_features

            for feature in range(n_features):
                g = self._calc_info_gain(X, y, feature)

                gains[feature] = g

            best_feature = np.argmax(gains)

            # Initialize the tree
            tree = {feature_names[best_feature]: {}}

            # Find all the possible values for the best feature we found
            feature_vals = self._find_unique_feature_values(X, best_feature)

            # Iterate through each possible feature value
            for feature_val in feature_vals:
                # Initialize variables for our new subset of data
                new_data = []
                new_classes = []
                new_names = []
                i = 0

                # Iterate through all of the data
                for instance in X:
                    # Grab the instance if the value of the best feature is the current feature value
                    if instance[best_feature] == feature_val:
                        # If the best feature is the first
                        if best_feature == 0:
                            new_instance = instance[1:]
                            new_names = feature_names[1:]

                        # If the best feature is the last
                        elif best_feature == n_features - 1:
                            new_instance = instance[:-1]
                            new_names = feature_names[:-1]

                        # If the best feature is somewhere in the middle
                        else:
                            new_instance = list(instance[:best_feature])

                            new_instance.extend(list(instance[best_feature + 1:]))

                            new_names = list(feature_names[:best_feature])

                            new_names.extend(list(feature_names[best_feature + 1:]))

                        # Update our new subset of data
                        new_data.append(new_instance)
                        new_classes.append(y[i])

                    i += 1

                # Get the subtree for our subset of data and add that to our current tree
                subtree = self._make_tree(new_data, new_classes, new_names)
                tree[feature_names[best_feature]][feature_val] = subtree

            # Return the tree
            return tree

    # Helper function for calculating information gain
    def _calc_info_gain(self, X, y, feature):
        # Get the number of data points
        n_instances = np.shape(X)[0]

        ## CALCULATE TOTAL ENTROPY
        # List of unique target class values
        total_class_vals = self._find_unique_class_values(X, y)

        # Counts of each unique target class value
        total_class_counts = [0] * len(total_class_vals)

        for i in range(len(total_class_vals)):
            for j in range(n_instances):
                if y[j] == total_class_vals[i]:
                    total_class_counts[i] += 1

        # Total entropy
        total_entropy = 0

        # Use the target class counts to calculate total entropy
        for total_class_count in total_class_counts:
            total_entropy += self._calc_entropy(total_class_count / n_instances)

        ## CALCULATE ENTROPY FOR GIVEN FEATURE
        # List of unique values that the feature can take
        feature_vals = self._find_unique_feature_values(X, feature)

        # Entropy for given feature
        info_S_A = 0

        # Iterate through each unique feature value
        for feature_val in feature_vals:
            # Number of data instances where the given feature is equal to the current unique feature value
            S_j = 0

            for instance in X:
                if instance[feature] == feature_val:
                    S_j += 1

            # List of unique target class values for the current unique feature value
            class_vals = self._find_unique_class_values(X, y, feature, feature_val)

            # Counts of each unique target class value for the current unique feature value
            class_counts = [0] * len(class_vals)

            for i in range(len(class_vals)):
                for j in range(n_instances):
                    if X[j][feature] == feature_val and y[j] == class_vals[i]:
                        class_counts[i] += 1

            # Calculate entropy
            info_S = 0

            for class_count in class_counts:
                info_S += self._calc_entropy(class_count / S_j)

            # Update the given feature entropy
            info_S_A += (S_j / n_instances) * info_S

        # Return the information gain (total entropy - entropy for the given feature)
        return total_entropy - info_S_A

    # Helper function for calculating entropy
    def _calc_entropy(self, p):
        if p != 0:
            return -p * np.log2(p)

        else:
            return 0

    # Helper function for finding all the unique values a feature can have
    def _find_unique_feature_values(self, X, feature):
        feature_vals = []

        for instance in X:
            if instance[feature] not in feature_vals:
                feature_vals.append(instance[feature])

        return feature_vals

    # Helper function for finding all the unique class/target values
    def _find_unique_class_values(self, X, y, feature=None, feature_val=None):
        class_vals = []

        for i in range(np.shape(y)[0]):
            if feature is None and y[i] not in class_vals:
                class_vals.append(y[i])

            elif feature is not None and X[i][feature] == feature_val and y[i] not in class_vals:
                class_vals.append(y[i])

        return class_vals

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        output = []

        for instance in X:
            curr_node = self.tree

            while True:
                curr_feature = list(curr_node.keys())[0]
                feature_val = instance[curr_feature]

                if feature_val in curr_node[curr_feature].keys():
                    curr_node = curr_node[curr_feature][feature_val]

                    if not isinstance(curr_node, dict):
                        output.append([curr_node])
                        break

                else:
                    most_frequent_class = max(curr_node[curr_feature].values(), key=lambda x: list(curr_node[curr_feature].values()).count(x))
                    output.append([most_frequent_class])
                    break

        return output


    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-li    def _shuffle_data(self, X, y):
        """
        # Use our predict method
        output = self.predict(X)

        # Find the total number of correct predictions and the total number of data instances
        num_correct = np.sum(np.all(np.array(output) == np.array(y), axis=1))
        num_instances = np.shape(X)[0]

        # Calculate and return the accuracy
        return num_correct / num_instances

