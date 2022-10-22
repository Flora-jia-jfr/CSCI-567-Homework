import numpy as np
import json
from matplotlib import pyplot as plt


def data_processing(data):
    train_set, valid_set, test_set = data['train'], data['valid'], data['test']
    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xval = valid_set[0]
    yval = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    Xtrain = np.array(Xtrain)
    Xval = np.array(Xval)
    Xtest = np.array(Xtest)

    ytrain = np.array(ytrain)
    yval = np.array(yval)
    ytest = np.array(ytest)

    return Xtrain, ytrain, Xval, yval, Xtest, ytest


def data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False):
    train_set, valid_set, test_set = data['train'], data['valid'], data['test']
    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xval = valid_set[0]
    yval = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    Xtrain = np.array(Xtrain)
    Xval = np.array(Xval)
    Xtest = np.array(Xtest)

    ytrain = np.array(ytrain)
    yval = np.array(yval)
    ytest = np.array(ytest)

    # We load data from json here and turn the data into numpy array
    # You can further perform data transformation on Xtrain, Xval, Xtest

    #####################################################
    #				 YOUR CODE HERE					    #
    #####################################################

    def minmax(Xtrain, Xval, Xtest):
        """
        linearly normalize by columns (features)
        z = (x-min)/(max-min)
        :param x:
        :return: minmax-normalized x
        """
        # print("minmax: ")
        Xtrain_processed = np.zeros(Xtrain.shape)
        Xtest_processed = np.zeros(Xtest.shape)
        Xval_processed = np.zeros(Xval.shape)
        for i in range(Xtrain.shape[1]):
            col = Xtrain[:, i]
            max = np.amax(col)
            min = np.amin(col)
            # minmax_processed[:, i] = (x[:, i] - min)/(max - min)
            Xtrain_processed[:, i] = (Xtrain[:, i] - min)/(max - min)
            Xtest_processed[:, i] = (Xtest[:, i] - min)/(max - min)
            Xval_processed[:, i] = (Xval[:, i] - min)/(max - min)
        # print(minmax_processed)
        return Xtrain_processed, Xval_processed, Xtest_processed

    # Min-Max scaling
    if do_minmax_scaling:
        Xtrain, Xval, Xtest = minmax(Xtrain, Xval, Xtest)
        # Xtrain = minmax(Xtrain)
        # Xval = minmax(Xval)
        # Xtest = minmax(Xtest)


    # Normalization
    def normalization(x):
        #####################################################
        #				 YOUR CODE HERE					    #
        #####################################################
        # print("normalization: ")
        # print(x.shape)
        normalized = np.zeros(x.shape)
        for i in range(x.shape[0]):
            l2_norm = np.sqrt(np.sum(np.square(x[i])))
            is_all_zero = not np.any(x[i])
            if is_all_zero:
                normalized[i] = x[i]
            else:
                # print("x: ", x)
                # print("normalized: ", x/l2_norm)
                normalized[i] = x[i]/l2_norm
                # print(x[i]/l2_norm)
        return normalized

    if do_normalization:
        Xtrain = normalization(Xtrain)
        Xval = normalization(Xval)
        Xtest = normalization(Xtest)

    return Xtrain, ytrain, Xval, yval, Xtest, ytest


def compute_l2_distances(Xtrain, X):
    """
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	"""
    #####################################################
    #				 YOUR CODE HERE					    #
    #####################################################
    # print("compute l2 distances")
    # print(Xtrain.shape)  # (618, 8)
    # print(X.shape)  # (75, 8)
    dists = np.zeros((X.shape[0], Xtrain.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Xtrain.shape[0]):
            dists[i][j] = np.sqrt(np.sum(np.square(X[i] - Xtrain[j])))
    return dists


def compute_cosine_distances(Xtrain, X):
    """
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Cosine distance between the ith test point and the jth training
	  point.
	"""
    #####################################################
    #				 YOUR CODE HERE					    #
    #####################################################
    # print("compute cosine distance:")
    dists = np.zeros((X.shape[0], Xtrain.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Xtrain.shape[0]):
            test = X[i]
            train = Xtrain[j]
            xtest_norm = np.sqrt(np.sum(np.square(test)))
            xtrain_norm = np.sqrt(np.sum(np.square(train)))
            if xtest_norm == 0 or xtrain_norm == 0:
                dists[i][j] = 1
            else:
                dists[i][j] = 1 - (np.dot(test, train))/(xtest_norm * xtrain_norm)
    return dists


def predict_labels(k, ytrain, dists):
    """
	Given a matrix of distances between test points and training points,
	predict a label for each test point.
	Inputs:
	- k: The number of nearest neighbors used for prediction.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  gives the distance betwen the ith test point and the jth training point.
	Returns:
	- ypred: A numpy array of shape (num_test,) containing predicted labels for the
	  test data, where y[i] is the predicted label for the test point X[i].
	"""
    #####################################################
    #				 YOUR CODE HERE					    #
    #####################################################
    # print("perdict labels")
    ypred = np.zeros(dists.shape[0])
    for index in range(dists.shape[0]):
        # arg sort all distances between training labels and the corresponding test label (index)
        # take the first k elements after sorting
        # bincount find the number of occurences of labels 0 and 1
        # argmax find the label with more occurences
        ypred[index] = np.argmax(np.bincount(ytrain[np.argsort(dists[index])[:k]]))
    return ypred
    # ypred = np.zeros(dists.shape[0])
    # for i in range(ypred.shape[0]):  # test[i]
    #     # print("i: ", i)
    #     # print(k)
    #     curr = dists[:, i].copy()
    #     # print("curr: ", curr)
    #     k_smallest = np.zeros(k, dtype=int)
    #     # print("initialize k smallest: ", k_smallest)
    #     labels = np.zeros(2, dtype=int)
    #     for j in range(k):  # k smallest
    #         smallest_index = np.argmin(curr)
    #         k_smallest[j] = smallest_index
    #         smallest_label = ytrain[smallest_index]
    #         labels[smallest_label] += 1
    #         curr[smallest_index] = np.inf
    #     # print("k_smallest: ", k_smallest)
    #     # print("labels: ", labels)
    #     if labels[0] > labels[1]:
    #         ypred[i] = 0
    #     elif labels[0] < labels[1]:
    #         ypred[i] = 1
    #     else:
    #         k_smallest.sort()
    #         # print("sorted: ", k_smallest)
    #         # print("else: ")
    #         # print(k_smallest[0])
    #         # print(ytrain[k_smallest[0]])
    #         ypred[i] = ytrain[k_smallest[0]]
    # return ypred


def compute_error_rate(y, ypred):
    """
	Compute the error rate of prediction based on the true labels.
	Inputs:
	- y: A numpy array with of shape (num_test,) where y[i] is the true label
	  of the ith test point.
	- ypred: A numpy array with of shape (num_test,) where ypred[i] is the
	  prediction of the ith test point.
	Returns:
	- err: The error rate of prediction (scalar).
	"""
    #####################################################
    #				 YOUR CODE HERE					    #
    #####################################################
    error_num = 0
    for i in range(y.shape[0]):
        if y[i] != ypred[i]:
            error_num += 1
    return error_num/y.shape[0]


def find_best_k(K, ytrain, dists, yval):
    """
	Find best k according to validation error rate.
	Inputs:
	- K: A list of ks.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the distance between the ith test point and the jth training
	  point.
	- yval: A numpy array with of shape (num_val,) where y[i] is the true label
	  of the ith validation point.
	Returns:
	- best_k: The k with the lowest error rate.
	- validation_error: A list of error rate of different ks in K.
	- best_err: The lowest error rate we get from all ks in K.
	"""
    #####################################################
    #				 YOUR CODE HERE					    #
    #####################################################
    all_err = []
    for single_k in K:
        ypred = predict_labels(single_k, ytrain, dists)
        err = compute_error_rate(yval, ypred)
        all_err.append(err)
    best_k_index = np.argmin(np.array(all_err))
    # return best_k, validation_error, best_err
    return K[best_k_index], all_err, min(all_err)


def main():
    input_file = 'disease.json'
    output_file = 'knn_output.txt'

    # ==================Problem Set 1.1=======================

    with open(input_file) as json_data:
        data = json.load(json_data)

    # Compute distance matrix
    Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

    dists = compute_l2_distances(Xtrain, Xval)

    # Compute validation accuracy when k=4
    k = 4
    ypred = predict_labels(k, ytrain, dists)
    err = compute_error_rate(yval, ypred)
    print("The validation error rate is", err, "in Problem Set 1.1")
    print()

    # ==================Problem Set 1.2=======================

    # Compute distance matrix
    Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=False,
                                                                                   do_normalization=True)

    dists = compute_l2_distances(Xtrain, Xval)

    # Compute validation accuracy when k=4
    k = 4
    ypred = predict_labels(k, ytrain, dists)
    err = compute_error_rate(yval, ypred)
    print("The validation error rate is", err, "in Problem Set 1.2 when using normalization")
    print()

    # Compute distance matrix
    Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=True,
                                                                                   do_normalization=False)

    dists = compute_l2_distances(Xtrain, Xval)

    # Compute validation accuracy when k=4
    k = 4
    ypred = predict_labels(k, ytrain, dists)
    err = compute_error_rate(yval, ypred)
    print("The validation error rate is", err, "in Problem Set 1.2 when using minmax_scaling")
    print()

    # ==================Problem Set 1.3=======================

    # Compute distance matrix
    Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)
    dists = compute_cosine_distances(Xtrain, Xval)
    # added

    # Compute validation accuracy when k=4
    k = 4
    ypred = predict_labels(k, ytrain, dists)
    err = compute_error_rate(yval, ypred)
    print("The validation error rate is", err, "in Problem Set 1.3, which use cosine distance")
    print()

    # ==================Problem Set 1.4=======================
    # Compute distance matrix
    Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

    # ======performance of different k in training set=====
    K = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    #####################################################
    #				 YOUR CODE HERE					    #
    #####################################################
    # print("compute best k with training set")
    dists = compute_l2_distances(Xtrain, Xtrain)
    best_k, all_err, best_err = find_best_k(K, ytrain, dists, ytrain)
    plt.figure()
    plt.plot(K, all_err, color='r', label="error rate of different k")
    plt.xlabel("k (num of neighbours)")
    plt.ylabel("error rate")
    plt.legend(loc="best")
    plt.title('error rate on different k on training set')
    plt.show()
    plt.savefig("result_plots/training_set_error_rate")

    # ==========select the best k by using validation set==============
    dists = compute_l2_distances(Xtrain, Xval)
    best_k, validation_error, best_err = find_best_k(K, ytrain, dists, yval)
    plt.figure()
    plt.plot(K, validation_error, color='r', label="error rate of different k")
    plt.xlabel("k (num of neighbours)")
    plt.ylabel("error rate")
    plt.legend(loc="best")
    plt.title('error rate on different k on validation set')
    plt.show()
    plt.savefig("result_plots/validation_set_error_rate")


    # ===============test the performance with your best k=============
    dists = compute_l2_distances(Xtrain, Xtest)
    ypred = predict_labels(best_k, ytrain, dists)
    test_err = compute_error_rate(ytest, ypred)
    print("In Problem Set 1.4, we use the best k = ", best_k, "with the best validation error rate", best_err)
    print("Using the best k, the final test error rate is", test_err)
    # ====================write your results to file===================
    f = open(output_file, 'w')
    for i in range(len(K)):
        f.write('%d %.3f' % (K[i], validation_error[i]) + '\n')
    f.write('%s %.3f' % ('test', test_err))
    f.close()


if __name__ == "__main__":
    main()
