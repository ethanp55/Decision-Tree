from desiciontree import *
from arff import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz

def cross_validation_split(X, y, n_folds=10):
    # Shuffle the data
    p = np.random.permutation(np.shape(X)[0])
    X = X[p]
    y = y[p]

    splits = []
    fold_size = int(np.shape(X)[0] / n_folds)
    start_index = 0

    for _ in range(n_folds):
        test_indices = np.arange(start_index, start_index + fold_size)
        X_train = np.delete(X, test_indices, axis=0)
        y_train = np.delete(y, test_indices, axis=0)
        X_test = X[test_indices, :]
        y_test = y[test_indices, :]

        splits.append((X_train, y_train, X_test, y_test))

        start_index += fold_size

    return splits

def fill_empty_vals(X):
    for i in range(np.shape(X)[1]):
         if np.any(np.isnan(X[:, i])):
             unique_vals = list(np.unique(X[:, i]))

             unknown_val = int(max(unique_vals) + 1)

             X[:, i][np.isnan(X[:, i])] = unknown_val

    return X

# Part 1
# DEBUGGING DATASET RESULTS
mat = Arff("datasets/lenses.arff",label_count=1)
counts = [] ## this is so you know how many types for each column

for i in range(mat.data.shape[1]):
       counts += [mat.unique_value_count(i)]

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
DTClass = DTClassifier(counts)
DTClass.fit(data,labels)
mat2 = Arff("datasets/all_lenses.arff")
data2 = mat2.data[:,0:-1]
labels2 = mat2.data[:,-1].reshape(-1,1)
pred = DTClass.predict(data2)
Acc = DTClass.score(data2,labels2)
np.savetxt("pred_lenses.csv", pred, delimiter=",")
print("DEBUG DATASET")
print("Accuracy = [{:.2f}]".format(Acc))
print()

# EVALUATION DATASET RESULTS
mat = Arff("datasets/zoo.arff",label_count=1)
counts = [] ## this is so you know how many types for each column

for i in range(mat.data.shape[1]):
       counts += [mat.unique_value_count(i)]

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
DTClass = DTClassifier(counts)
DTClass.fit(data,labels)
mat2 = Arff("datasets/all_zoo.arff")
data2 = mat2.data[:,0:-1]
labels2 = mat2.data[:,-1].reshape(-1,1)
pred = DTClass.predict(data2)
np.savetxt("pred_zoo.csv",pred,delimiter=",")
Acc = DTClass.score(data2,labels2)
print("EVALUATION DATASET")
print("Accuracy = [{:.2f}]".format(Acc))
print()

# Part 2
# CARS DATASET RESULTS
mat = Arff("datasets/cars.arff",label_count=1)
counts = [] ## this is so you know how many types for each column

for i in range(mat.data.shape[1]):
       counts += [mat.unique_value_count(i)]

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    DTClass = DTClassifier(counts)
    DTClass.fit(split[0], split[1])
    train_accuracy = DTClass.score(split[0], split[1])
    test_accuracy = DTClass.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("CARS DATASET")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print(DTClass.tree)
print()

# VOTING DATASET RESULTS
mat = Arff("datasets/voting.arff",label_count=1)
counts = [] ## this is so you know how many types for each column

for i in range(mat.data.shape[1]):
       counts += [mat.unique_value_count(i)]

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    DTClass = DTClassifier(counts)
    DTClass.fit(split[0], split[1])
    train_accuracy = DTClass.score(split[0], split[1])
    test_accuracy = DTClass.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("VOTING DATASET")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print(DTClass.tree)
print()


# Part 5
# CARS DATASET RESULTS - DEFAULT
mat = Arff("datasets/cars.arff",label_count=1)

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    sklearn_tree = DecisionTreeClassifier(criterion="entropy")
    sklearn_tree.fit(split[0], split[1])
    train_accuracy = sklearn_tree.score(split[0], split[1])
    test_accuracy = sklearn_tree.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("CARS DATASET - DEFAULT")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print()

# CARS DATASET RESULTS - MAX DEPTH = 4
mat = Arff("datasets/cars.arff",label_count=1)

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    sklearn_tree.fit(split[0], split[1])
    train_accuracy = sklearn_tree.score(split[0], split[1])
    test_accuracy = sklearn_tree.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("CARS DATASET - MAX DEPTH = 4")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print()

# CARS DATASET RESULTS - MAX DEPTH = 5
mat = Arff("datasets/cars.arff",label_count=1)

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    sklearn_tree.fit(split[0], split[1])
    train_accuracy = sklearn_tree.score(split[0], split[1])
    test_accuracy = sklearn_tree.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("CARS DATASET - MAX DEPTH = 5")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print()

# CARS DATASET RESULTS - MIN SAMPLES SPLIT = 5
mat = Arff("datasets/cars.arff",label_count=1)

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    sklearn_tree = DecisionTreeClassifier(criterion="entropy", min_samples_split=5)
    sklearn_tree.fit(split[0], split[1])
    train_accuracy = sklearn_tree.score(split[0], split[1])
    test_accuracy = sklearn_tree.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("CARS DATASET - MIN SAMPLES SPLIT = 5")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print()

# CARS DATASET RESULTS - MIN SAMPLES LEAF = 3
mat = Arff("datasets/cars.arff",label_count=1)

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    sklearn_tree = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3)
    sklearn_tree.fit(split[0], split[1])
    train_accuracy = sklearn_tree.score(split[0], split[1])
    test_accuracy = sklearn_tree.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("CARS DATASET - MIN SAMPLES LEAF = 3")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print()

# CARS DATASET RESULTS - MAX DEPTH = 5, MIN SAMPLES SPLIT = 5, MIN SAMPLES LEAF = 3
mat = Arff("datasets/cars.arff",label_count=1)

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=5, min_samples_leaf=3)
    sklearn_tree.fit(split[0], split[1])
    train_accuracy = sklearn_tree.score(split[0], split[1])
    test_accuracy = sklearn_tree.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("CARS DATASET - MAX DEPTH = 5, MIN SAMPLES SPLIT = 5, MIN SAMPLES LEAF = 3")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print()

# VOTING DATASET RESULTS - DEFAULT
mat = Arff("datasets/voting.arff",label_count=1)

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    sklearn_tree = DecisionTreeClassifier(criterion="entropy")
    sklearn_tree.fit(split[0], split[1])
    train_accuracy = sklearn_tree.score(split[0], split[1])
    test_accuracy = sklearn_tree.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("VOTING DATASET - DEFAULT")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print()

# VOTING DATASET RESULTS - MAX DEPTH = 4
mat = Arff("datasets/voting.arff",label_count=1)

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    sklearn_tree.fit(split[0], split[1])
    train_accuracy = sklearn_tree.score(split[0], split[1])
    test_accuracy = sklearn_tree.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("VOTING DATASET - MAX DEPTH = 4")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print()

# VOTING DATASET RESULTS - MAX DEPTH = 5
mat = Arff("datasets/voting.arff",label_count=1)

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    sklearn_tree.fit(split[0], split[1])
    train_accuracy = sklearn_tree.score(split[0], split[1])
    test_accuracy = sklearn_tree.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("VOTING DATASET - MAX DEPTH = 5")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print()

# VOTING DATASET RESULTS - MIN SAMPLES SPLIT = 5
mat = Arff("datasets/voting.arff",label_count=1)

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    sklearn_tree = DecisionTreeClassifier(criterion="entropy", min_samples_split=5)
    sklearn_tree.fit(split[0], split[1])
    train_accuracy = sklearn_tree.score(split[0], split[1])
    test_accuracy = sklearn_tree.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("VOTING DATASET - MIN SAMPLES SPLIT = 5")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print()

# VOTING DATASET RESULTS - MIN SAMPLES LEAF = 3
mat = Arff("datasets/voting.arff",label_count=1)

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    sklearn_tree = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3)
    sklearn_tree.fit(split[0], split[1])
    train_accuracy = sklearn_tree.score(split[0], split[1])
    test_accuracy = sklearn_tree.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("VOTING DATASET - MIN SAMPLES LEAF = 3")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print()

# VOTING DATASET RESULTS - MAX DEPTH = 5, MIN SAMPLES SPLIT = 5, MIN SAMPLES LEAF = 3
mat = Arff("datasets/voting.arff",label_count=1)

data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

data = fill_empty_vals(data)

cv_splits = cross_validation_split(data, labels)
cv_train_accuracies = []
cv_test_accuracies = []

for split in cv_splits:
    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=5, min_samples_leaf=3)
    sklearn_tree.fit(split[0], split[1])
    train_accuracy = sklearn_tree.score(split[0], split[1])
    test_accuracy = sklearn_tree.score(split[2], split[3])
    cv_train_accuracies.append(train_accuracy)
    cv_test_accuracies.append(test_accuracy)

print("VOTING DATASET - MAX DEPTH = 5, MIN SAMPLES SPLIT = 5, MIN SAMPLES LEAF = 3")
print("Train accuracies = " + str(cv_train_accuracies))
print("Test accuracies = " + str(cv_test_accuracies))
print("Average train accuracy = " + str(sum(cv_train_accuracies) / len(cv_train_accuracies)))
print("Average test accuracy = " + str(sum(cv_test_accuracies) / len(cv_test_accuracies)))
print("Decision tree:")
print()

# SKLEARN ON OWN DATASET
mat = Arff("datasets/desharnais.arff",label_count=1)

data = mat.data[:,1:-1]
labels = mat.data[:,-1].reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

param_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              'min_samples_split':[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              'criterion': ['entropy', 'gini']}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=10, return_train_score=True)
grid_search.fit(X_train, y_train)

print("Best score: {:.4f}".format(grid_search.best_score_))
print("Best parameters: {}".format(grid_search.best_params_))

bestdt = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=7, min_samples_split=2)
bestdt.fit(X_train, y_train)
print("Best tree train accuracy:", bestdt.score(X_train, y_train))
print("Best tree test accuracy:", bestdt.score(X_test, y_test))

# Part 6
# Use the command "dot -Tpng bestdt.dot -o bestdt.png" to generate an image
export_graphviz(bestdt, out_file="bestdt.dot", feature_names=["TeamExp", "ManagerExp", "YearEnd", "Length", "Effort",
                                                              "Transactions", "Entities", "PointsNonAdjust",
                                                              "Adjustment", "PointsAdjust"])
