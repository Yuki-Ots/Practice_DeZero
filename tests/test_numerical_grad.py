import unittest
import numpy as np
import dezero as dz
import dezero.functions as F
from dezero import Variable

class TestNumericalGrad(unittest.TestCase):
    def test_numerical_grad1(self):
        a = Variable(np.array([-1, 0, 5]))
        grad = F.numerical_grad(F.exp, a)
        grad_prime = np.exp(np.array([-1, 0, 5]))
        self.assertTrue(np.allclose(grad, grad_prime))

if __name__ == "__main__":
    unittest.main()
