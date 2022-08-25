import numpy as np
labels_data=np.load('/Users/zhangxiaoheng/Desktop/youtube_data/y_valid.npy')
labels=[]
print(labels_data)
row=labels_data.shape[0]
#print(row)
#print(labels_data[:,1])
#print(labels_data[1,:])
for i in range(row): #0-40
    if (labels_data[i, :][0]==1.0):
        labels.append(0)
    elif (labels_data[i, :][1]==1.0):
        labels.append(1)
    elif (labels_data[i, :][2]==1.0):
        labels.append(2)
print(labels)
labels_data=np.column_stack((labels_data,labels))
print(labels_data)

