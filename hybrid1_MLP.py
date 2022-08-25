from sklearn.neural_network import MLPClassifier

import numpy as np
x_train_data=np.load('/Users/zhangxiaoheng/Desktop/youtube_data/covarep_train.npy')
x_train_data=x_train_data.reshape(x_train_data.shape[0],x_train_data.shape[1]*x_train_data.shape[2])
y_train_data=np.load('/Users/zhangxiaoheng/Desktop/youtube_data/y_train.npy')
print(x_train_data.shape)
print(y_train_data.shape)

x_test_data=np.load('/Users/zhangxiaoheng/Desktop/youtube_data/covarep_test.npy')
x_test_data=x_test_data.reshape(x_test_data.shape[0],x_test_data.shape[1]*x_test_data.shape[2])
y_test_data=np.load('/Users/zhangxiaoheng/Desktop/youtube_data/y_test.npy')
clf = MLPClassifier(solver='adam',activation = 'relu',alpha = 1e-5,hidden_layer_sizes = (100,100,100),random_state = 42,verbose = True)

clf.fit(x_train_data,y_train_data)

print(clf.predict(x_test_data))

print(clf.score(x_test_data,y_test_data))