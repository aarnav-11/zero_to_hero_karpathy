import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import random


data_path = "data/cluster_data.csv"
dataset = pd.read_csv(data_path)
dataset_og = pd.read_csv(data_path)




dataset = dataset.to_numpy()


def calc_distance(x1,y1,x2,y2):
    return pow(pow(x2-x1,2) + pow(y2-y1,2),1/2)


def training():
    minx, maxx, miny, maxy = np.min(dataset[:,0]), np.max(dataset[:,0]), np.min(dataset[:,1]), np.max(dataset[:,1])
# print(minx, miny, maxx, maxy)
    k = 2
    initx1, inity1 = random.uniform(minx, maxx), random.uniform(miny,maxy)
    initx2, inity2 = random.uniform(minx, maxx), random.uniform(miny,maxy)
    prev1 = [initx1, inity1]
    prev2 = [initx2, inity2]
    assignments1 = []
    assignments2 = []
    assignmentsprev = []
    j = 0
    while True:
        j += 1
        sumonex, sumtwox, sumoney, sumtwoy = 0, 0, 0, 0
        for x,y in dataset:
            one = calc_distance(x,y,initx1,inity1)
            two = calc_distance(x,y,initx2,inity2)
            if one < two:
                assignments1.append((x,y))
                sumonex += x
                sumoney += y
            else:
                assignments2.append((x,y))
                sumtwox +=  x
                sumtwoy += y
        
        if len(assignments1) > 0:
            initx1 = sumonex / len(assignments1)
            inity1 = sumoney / len(assignments1)

        if len(assignments2) > 0:
            initx2 = sumtwox / len(assignments2)
            inity2 = sumtwoy / len(assignments2)
        if (np.allclose(prev1, [initx1, inity1]) and np.allclose(prev2, [initx2, inity2])) or j == 15:
            break
        print(f"step: {j}, 1: ({initx1, inity1}), 2:({initx2, inity2})")
        assignments1, assignments2= [], []
        prev1 = [initx1, inity1]
        prev2 = [initx2, inity2]   
    return initx1, inity1, initx2, inity2

x1, x2 = 0,0
y1, y2 = 0,0

for i in range(100):
    a = training()
    x1 += a[0] 
    x2 += a[2]
    y1 += a[1]
    y2 += a[3]

final1 = (x1/100, y1/100)
final2 = (x2/100, y2/100)
print(final1, final2)

f1 = dataset_og["Feature 1"]
f2 = dataset_og["Feature 2"]
plt.scatter(*final1, color="red", s=200, marker="X", label="Centroid 1")
plt.scatter(*final2, color="green", s=200, marker="X", label="Centroid 2")
plt.scatter(f1, f2)
plt.show()
# def inference(x,y):
#     one = calc_distance(x,y,final1[0],final1[1])
#     two = calc_distance(x,y,final1[0],final2[1]) 
#     if one < two:
#         return 1
#     else:
#         return 2
    
# a = inference(5,5)
# print(a)


# print(dataset[:5])
