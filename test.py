import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

z1 = 1+2j
z2 = 3+4j

def inner_product(c1, c2):
    return np.conj(c1) * c2

def born_rule_calculator(phi, psi):
    top = inner_product(phi, psi)
    print(top)
    bottom = np.sqrt(inner_product(phi, phi)) * np.sqrt(inner_product(psi, psi))

    return top / bottom

print(born_rule_calculator((1, 0), (1/np.sqrt(2), -1/np.sqrt(2))))
