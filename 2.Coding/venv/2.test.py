import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pylab as plt

# 폰트 설정
mpl.rc('font', family='NanumGothic')
# 유니코드에서  음수 부호설정
mpl.rc('axes', unicode_minus=False)
#get current path
currentPath = os.getcwd()

#change path
os.chdir('D:/OH/Final_project/2.Coding')

test = pd.read_csv('HomeC.csv')

test.head()
test.tail()

test = test.drop(503910,0)
test0 = test[0:86399]
test1 = test0["use "]
test.tail()
plt.plot(test1)
plt.show()

test.loc[0:86399]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np