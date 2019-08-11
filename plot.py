import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel("preprocess_copy.xlsx")

rdx = np.array(data[r'circle area'])
rdy = np.array(data[r"elipse area"])

plt.plot(rdx[70:-1],rdy[70:-1],"bo",color = "green", label = "Normal")
plt.plot(rdx[:69],rdy[0:69],"bo",color = "red", label = "Vestibular disorder")

plt.legend(title= "Mean frequency x, y")
plt.show()