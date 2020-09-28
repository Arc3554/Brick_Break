# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:02:22 2019

@author: scream
"""
import pickle
with open("xxx.pickle", "rb") as f:
    data_list = pickle.load(f)

# save each information seperately
Frame=[]
Status=[]
Ballposition=[]
PlatformPosition=[]
Bricks=[]
for i in range(0,len(data_list)):
    if i>=1 and data_list[i].ball[1] < data_list[i-1].ball[1] and data_list[i].ball[1]<=200 and data_list[i].ball[1]>=190:
       i=i
    else:
       Frame.append(data_list[i].frame)
       Status.append(data_list[i].status)
       Ballposition.append(data_list[i].ball)
       PlatformPosition.append(data_list[i].platform)
       Bricks.append(data_list[i].bricks)

#%% calculate instruction of each frame using platformposition
import numpy as np
PlatX=np.array(PlatformPosition)[:,0][:, np.newaxis]
PlatX_next=PlatX[1:,:]
instruct=(PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5


# select some features to make x
Ballarray=np.array(Ballposition[:-1])
x=np.hstack((Ballarray, PlatX[0:-1,0][:,np.newaxis]))
# select intructions as y
y=instruct

#%% train your model here
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
neigh = KNeighborsClassifier(n_neighbors = 10)
neigh.fit(x_train,y_train)
y_knn = neigh.predict(x_test)
# check the acc to see how well you've trained the model
acc = accuracy_score(y_knn,y_test)


#%% save model
import pickle

filename="knn_吳政緯.sav"
pickle.dump(neigh, open(filename, 'wb'))

# load model
#l_model=pickle.load(open(filename,'rb'))
#yp_l=l_model.predict(x_test)
#print("acc load: %f " % accuracy_score(yp_l, y_test))