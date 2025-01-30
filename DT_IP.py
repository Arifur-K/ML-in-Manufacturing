import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#Step-1: Import data set
df=pd.read_csv("C:\\Users\\marks\\OneDrive\\Desktop\\Admission\\Website Projects\\ML\\DT_IP.csv")

print(df)

#Step-2: Convert catagorical data to numerical data 
d1={'Punch':1,'Laser':2}
df['Cutting']=df['Cutting'].map(d1)
d2={'Yes':1,'No':0}
df['Deburring']=df['Deburring'].map(d2)
d3={'Good':1,'Bad':0}
df['Quality']=df['Quality'].map(d3)
print(df)

#Step-3: Steperate feature columns and target column
features=['Thickness','Cutting','Deburring']
X=df[features]
y=df['Quality']
print(X)
print(y)
dtree=DecisionTreeClassifier()
dtree=dtree.fit(X,y)

plt.figure(figsize=(10, 9))
tree.plot_tree(dtree,feature_names=features)
plt.show()

print(dtree.predict([[5, 2, 0]]))

print("[1] means 'Good'")
print("[0] means 'Bad'")