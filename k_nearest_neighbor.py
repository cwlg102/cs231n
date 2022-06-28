from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        print(X.shape)
        print(self.X_train.shape)
        for i in range(num_test): #500
            for j in range(num_train): #5000
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                dists[i][j] = (np.sum((X[i] - self.X_train[j])**2))**0.5
                #단순 포문 두개
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            c_X_train = self.X_train.copy()
            c_X_train = c_X_train - X[i]
            c_X_train = c_X_train ** 2
            c_X_train = np.transpose(np.sqrt(np.sum(c_X_train, axis = 1, 
            keepdims = False)))
            dists[i] = c_X_train

            #train데이터를 건들수 없으므로, copy해와야
            #test 데이터인 X[i]한줄을 브로드 캐스팅하면 (a-b)
            #그것을 제곱하면 (a-b)^2
            #그거 제곱근하면 L2
            #차원 맞춰주기위해 쭉더하고 transpose
            #dists[i]에 저장.

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        #데이터 거리 구할땐 두 행렬에 대해 행마다 빼주고 제곱 후 제곱근인데,
        #행렬곱하면 각 원소는 ab인데, 어? 어디서 많이본거네? 
        #L2를 전개하면 a^2-2ab+b^2인데
        #a^2이랑 b^2만 해서 행렬곱한것에 더해주면 되겠네?

        result = np.sum(X**2, axis = 1, keepdims = True) \
         - 2 * X @ np.transpose(self.X_train) \
        + np.sum((np.transpose(self.X_train))**2, axis = 0, keepdims = True)
        dists = np.sqrt(result)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #y_train도 5000, dists도 열 5000

            #dists[i]에 맞는 label.
            tmp_idx = np.argsort(dists[i])
            for j in range(k):
                closest_y.append(self.y_train[tmp_idx[j]])
            #train 5000개, train label도 5000개
            #dists는 test데이터 -> train데이터 간 L2 distance정보가 있음 (500, 5000)
            #ex)첫번째 행은, test의 첫번째 데이터에 대해, 각열에 train데이터
            #와의 L2 distance를 기록을 해놓음.
            #그걸 오름 차순으로 정렬하면, 제일 거리가 짧은거(데이터 유사도가 높은거) 순서로
            #정렬하게됨. (길이는 5000개)
            #여담으로 argsort하므로 정렬하되 원래 배열의 인덱스를 가져옴
            #그러면 , 주어진 test 데이터와 train데이터 간 L2가 제일짧은 거리를 가진 
            #것을, 그 인덱스를 y_train것을 사용하면 가장 가까운 k개의 label을 가져올 수 있음.
            #(X_train의 인덱스와 1:1 대응되는 y_train에있는 label정보)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            #가까운거 k개 중 제일 label이 많이 나온것의 key를,
            #y_pred[i]에 초기화.
            
            find_most_common_label = {}
            for j in range(k):
                try:find_most_common_label[closest_y[j]] += 1
                except:find_most_common_label[closest_y[j]] = 1
            y_pred[i]=(max(find_most_common_label, key=find_most_common_label.get))


            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
