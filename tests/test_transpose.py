import unittest
import numpy as np
import dezero as dz
import dezero.functions as F
from dezero import Variable


class TestTranspose(unittest.TestCase):
    def test_transpose(self):
        a = dz.Variable(np.arange(10)).reshape(2, -1)
        a = a.transpose()
        a_prime = np.arange(10).reshape(2, -1).transpose()
        a_prime = Variable(a_prime)
        self.assertEqual(a, a_prime)

    def test_transpose2(self):
        a = dz.Variable(np.arange(24)).reshape(2, 3, 4)
        a = a.transpose(2, 0, 1)
        a_prime = np.arange(24).reshape(2, 3, 4).transpose(2, 0, 1)
        a_prime = Variable(a_prime)
        self.assertEqual(a, a_prime)

    def test_transpose3(self):
        a = dz.Variable(np.arange(10)).reshape(2, -1)
        a = F.transpose(a)
        a_prime = np.arange(10).reshape(2, -1).transpose()
        a_prime = Variable(a_prime)
        self.assertEqual(a, a_prime)


if __name__ == "__main__":
    unittest.main()
