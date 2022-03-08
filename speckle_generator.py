import numpy as np
import matplotlib.pyplot as plt
import pyspeckle

y = pyspeckle.create_Exponential(201,2)
# pyspeckle.statistics_plot(y)
plt.imshow(y)

plt.show()


