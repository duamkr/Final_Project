import numpy as np
import pandas as pd
from datetime import datetime
import os
import random
os.chdir("E:/4.프로젝트_빅데이터솔루션/Final_Project/2.Coding")
print(os.getcwd())
import sys
sys.stdout = open('2015_2018.txt','w')


# 월별 시간당 평균 kwh
a = [0.286805556,0.40162037,0.446412037,0.469791667,0.493287037,0.544907407,0.538194444,0.643865741,0.593287037]
# 표준편차
b = [0.015505647,0.022766637,0.02345431,0.024874298,0.027060391,0.02576232,0.026889197,0.036098766,0.102803084]
# 고객번호
home = ['H0012','H0024','H0033','H0042','H0055','H0062','H0074','H0083','H0092','H0102','H0111','H0122','H0134','H0142',
        'H0153','H0164','H0173','H0184','H0193','H0203','H0211','H0224','H0232','H0245','H0252','H0263','H0274','H0284',
        'H0292','H0303','H0319','H0321','H0334','H0343','H0353','H0363','H0374','H0383','H0392','H0402','H0414','H0421',
        'H0433','H0441','H0452','H0464','H0475','H0482','H0493','H0502','H0513','H0521','H0533','H0544','H0554','H0562',
        'H0572','H0588','H0594','H0605','H0614','H0621','H0631','H0643','H0652','H0662','H0674','H0687','H0692','H0704',
        'H0713','H0722','H0731','H0745','H0754','H0766','H0772','H0781','H0793','H0802','H0811','H0823','H0831','H0844',
        'H0852','H0863','H0874','H0881','H0894','H0901','H0915','H0924','H0932','H0941','H0953','H0964','H0973','H0982','H0993','H1002']
# 월별 가중치
mon= [1.11,1.06,1.02,0.98,0.97,0.94,0.95,1.02,1.06,0.96,0.95,1.00]
# 최소 - 최대값
min1 = [0.271299909,0.378853733,0.422957728,0.444917368,0.466226646,0.519145087,0.511305248,0.607766975,0.490483953]
max1 = [0.302311202,0.424387008,0.469866347,0.494665965,0.520347428,0.570669728,0.565083641,0.679964507,0.696090121]

# Gauss 정규분포
file = open('output.txt','a')
for i in pd.date_range("2015-1-1", "2019-12-31", freq = 'h') :
    for j in home :
        c = int(j[-1])
        d = int(i.month)
        print(j, i.year,i.month,i.day,i.hour, (random.gauss(a[c-1], b[c-1]) * mon[d-1]),sep = "/")

file.close()

# 최소-최대값
for i in pd.date_range("2018-1-1", "2018-12-31", freq = 'h') :
    for j in home :
        c = int(j[-1])
        d = int(i.month)
        print(j, i.year,i.month,i.day,i.hour, (random.uniform(min1[c-1], max2[c-1]) * mon[d-1]),sep = "/")


# 선택적 저장
file = open('output.txt','a')

for ....:
if .... :
file.write("조건에 맞았습니다. \n")

file.close()