import numpy as np #type:ignore
import pandas as pd #type:ignore
import matplotlib.pyplot as plt #type:ignore

def generate_data(num_points,dimension,num_classes,spread):
    X = np.random.rand(num_points,dimension) * spread
    y = np.random.randint(0,num_classes,size=num_points)
    return X,y

X,y = generate_data(200,2,2,100)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

data = np.hstack((X,y.reshape(-1,1)))

df = pd.DataFrame(data,columns=["X1","X2","Label"])
df.to_csv("dataset.csv",index=False)
print("Dataset saved to csv successfully")
