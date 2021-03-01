
"""

Computing features about accelerometer orientations

Author: Binod Thapa Chhetry

Date: Jul 10, 2018
"""
import numpy as np
from numpy.linalg import norm
from SWaN-accel.utils import *

class EnergyFeature:
    def __init__(self, X, subwins=30):
        self._X = X
        self._subwins = subwins

    @staticmethod
    def energies(X):
        X = as_float64(X)
        if not has_enough_samples(X):
            print(
                '''One of sub windows do not have enough samples, will ignore in
                feature computation''')
            energies = np.array([np.nan])
        else:
            energies = np.array([np.sum(np.square(X))/(X.shape[0])])
        return vec2rowarr(energies)
    
    
    def get_energies(self):
        result = apply_over_subwins(
           self._X, EnergyFeature.energies, subwins=self._subwins)

        self._energies = np.concatenate(result, axis=0)
        return self

    def smv_energy_sum(self):
        smv_energy_sum = np.nansum(self._energies, axis=0)
        result = vec2rowarr(smv_energy_sum)
        result = add_name(result, self.smv_energy_sum.__name__)
        return result

    def smv_energy_var(self):
        smv_energy_var = np.nanvar(self._energies, axis=0)
        result = vec2rowarr(smv_energy_var)
        result = add_name(result, self.smv_energy_var.__name__)
        return result

    