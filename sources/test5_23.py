# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import *


from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt

x = Variable(np.array(3.0))
y = Variable(np.array(2.0))
