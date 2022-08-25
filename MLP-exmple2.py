from sklearn.neural_network import MLPClassifier

import gzip
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
with gzip.open('/Users/zhangxiaoheng/Downloads/mnist.pkl.gz') as f_gz:

    train_data,valid_data,test_data = pickle.load(f_gz,encoding='bytes')

X_training_data, y_training_data = train_data
X_valid_data, y_valid_data = valid_data
X_test_data, y_test_data = test_data


def show_data_struct():
    print(X_training_data.shape, y_training_data.shape)
    print(X_valid_data.shape, y_valid_data.shape)
    print(X_test_data.shape, y_test_data.shape)
    print(X_training_data[0])
    print(y_training_data[0])


#show_data_struct()
X_training = np.vstack((X_training_data, X_valid_data))
y_training = np.append(y_training_data, y_valid_data)


def show_image():
    plt.figure(1)
    for i in range(10):
        image = X_training[i]
        pixels = image.reshape((28, 28))
        plt.subplot(5, 2, i + 1)
        plt.imshow(pixels, cmap='gray')
        plt.title(y_training[i])
        plt.axis('off')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                        wspace=0.85)
    plt.show()


#show_image()

clf = MLPClassifier(solver='sgd',activation = 'identity',max_iter = 10,alpha = 1e-5,hidden_layer_sizes = (40,50,50),random_state = 1,verbose = True)

clf.fit(train_data[0][:10000],train_data[1][:10000])

print(clf.predict(test_data[0][:10]))

print(clf.score(test_data[0][:100],test_data[1][:100]))

#print(clf.predict_proba(test_data[0][:10]))
"""
mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(100,), (100, 30)],
                             "solver": ['adam', 'sgd', 'lbfgs'],
                             "max_iter": [20],
                             "verbose": [True]
                             }
mlp = MLPClassifier()
estimator = GridSearchCV(mlp, mlp_clf__tuned_parameters, n_jobs=6)
estimator.fit(X_training, y_training)
print(estimator.get_params().keys())
print(estimator.best_params_)
"""