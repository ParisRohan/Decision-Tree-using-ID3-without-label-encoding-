
from sklearn.tree import DecisionTreeClassifier as DTC
import numpy as np
import pandas as pa

dataset = pa.read_csv("tree.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
#print(X,Y)

Dtree = DTC(criterion="entropy")
#print(tree)

model = Dtree.fit(X,Y)

y1 = [1,148,72,0,33.6,0.627,50]
print("Prediction of tuple :",y1)
print(model.predict([y1]))


import graphviz as gv
import sklearn.tree as tree

gv_comp_model = tree.export_graphviz(model,feature_names=["pregnant","glucose","BP","insuline","BMI","pedigree","age"],class_names=['yes','no'])
#print(gv_comp_model)
x = gv.Source(gv_comp_model)
#print(x)
x.render("tree")





