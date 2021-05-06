import matplotlib.pyplot as plt
import numpy as np

#Reading Testing Data Output
testDF = pd.read_csv('TestingResults.txt', header=None)
y_labels = testDF[24].tolist()
testDF = testDF.drop(24, axis=1)
x_data = testDF.values.tolist()

hours = np.arrange()