# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:57:40 2020

@author: Hassan
"""

import scipy.stats as st
print(st.bernoulli.pmf(1, .5))
print(st.bernoulli.pmf(0, .5))


import numpy as np
params = np.linspace(0, 1, 100)


import matplotlib.pyplot as plt
import numpy as np
plt.xlabel('x: height [cm]')
plt.axis([140,200,-2,2])
x_1 = np.random.normal(165,5,20)
x_2 = np.random.normal(180,6,20)
plt.plot(x_1,np.zeros(len(x_1)),'rx')
plt.plot(x_2,np.zeros(len(x_2)),'ko')
plt.show()


x_1
np.random.normal(165,5,20)