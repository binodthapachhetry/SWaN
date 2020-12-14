"""

Computing features about accelerometer orientations

Author: Qu Tang

Date: Jul 10, 2018
"""
import numpy as np
from numpy.linalg import norm
from SWaN.utils import *
from math import *

class OrientationFeature:
    def __init__(self, X, subwins=4):
        self._X = X
        self._subwins = subwins

    @staticmethod
    def orientation_xyz(X, unit='rad'):
        X = as_float64(X)
        if not has_enough_samples(X):
            print(
                '''One of sub windows do not have enough samples, will ignore in
                feature computation''')
            orientation_xyz = np.array([np.nan, np.nan, np.nan])
        else:
#             gravity = np.array(np.mean(X, axis=0), dtype=np.float)
#             print(gravity.shape)
#             gravity_vm = norm(gravity, ord=2, axis=0)
#             orientation_xyz = np.arccos(
#                 gravity / gravity_vm) if gravity_vm != 0 else np.zeros_like(gravity)
#             if unit == 'deg':
#                 orientation_xyz = np.rad2deg(orientation_xyz)
            
            gravity = np.median(X, axis=0)
            theta = atan(gravity[0] / np.sqrt(np.sum(np.square(np.array(gravity[[1,2]])))) ) * (180 / pi)
            trident = atan(gravity[1] / np.sqrt(np.sum(np.square(np.array(gravity[[0,2]])))) ) * (180 / pi)
            phi= atan( np.sqrt(np.sum(np.square(np.array(gravity[[0,1]])))) / gravity[2] ) * (180 / pi) 
            orientation_xyz = np.array([theta,trident,phi])
            
#             theta = atan(gravity[0] / norm(gravity[1:2]) ) * (180 / pi)
#             trident = atan(gravity[1] / norm(gravity[[0,2]]) ) * (180 / pi)
#             phi= atan( norm(gravity[0:1])/ gravity[2] ) * (180 / pi) 
#             orientation_xyz = np.array([theta,trident,phi])

            orientation_xyz = np.array([theta,trident,phi])
            
           
        return vec2rowarr(orientation_xyz)

    def estimate_orientation(self, unit='rad'):
        result = apply_over_subwins(
            self._X, OrientationFeature.orientation_xyz, subwins=self._subwins, unit=unit)
        self._orientations = np.concatenate(result, axis=0)
        return self

    def median_angles(self):
        median_angles = np.nanmedian(self._orientations, axis=0)
        result = vec2rowarr(np.array(median_angles))
        result = add_name(result, self.median_angles.__name__)
        return result

    def median_x_angle(self):
        median_angles = np.nanmedian(self._orientations, axis=0)
        result = vec2rowarr(np.array([median_angles[0]]))
        result = add_name(result, self.median_x_angle.__name__)
        return result

    def ori_x_median(self):
        median_angles = np.nanmedian(self._orientations, axis=0)
        result = vec2rowarr(np.array([median_angles[0]]))
        result = add_name(result, self.ori_x_median.__name__)
        return result

    def ori_y_median(self):
        median_angles = np.nanmedian(self._orientations, axis=0)
        result = vec2rowarr(np.array([median_angles[1]]))
        result = add_name(result, self.ori_y_median.__name__)
        return result

    def ori_z_median(self):
        median_angles = np.nanmedian(self._orientations, axis=0)
        result = vec2rowarr(np.array([median_angles[2]]))
        result = add_name(result, self.ori_z_median.__name__)
        return result

    def median_y_angle(self):
        median_angles = np.nanmedian(self._orientations, axis=0)
        result = vec2rowarr(np.array([median_angles[1]]))
        result = add_name(result, self.median_y_angle.__name__)
        return result

    def median_z_angle(self):
        median_angles = np.nanmedian(self._orientations, axis=0)
        result = vec2rowarr(np.array([median_angles[2]]))
        result = add_name(result, self.median_z_angle.__name__)
        return result

    def range_angles(self):
        range_angles = abs(np.nanmax(
            self._orientations, axis=0) - np.nanmin(self._orientations, axis=0))
        result = vec2rowarr(np.array(range_angles))
        result = add_name(result, self.range_angles.__name__)
        return result

    def range_x_angle(self):
        range_angles = np.nanmax(
            self._orientations, axis=0) - np.nanmin(self._orientations, axis=0)
        result = vec2rowarr(np.array([range_angles[0]]))
        result = add_name(result, self.range_x_angle.__name__)
        return result

    def range_y_angle(self):
        range_angles = np.nanmax(
            self._orientations, axis=0) - np.nanmin(self._orientations, axis=0)
        result = vec2rowarr(np.array([range_angles[1]]))
        result = add_name(result, self.range_y_angle.__name__)
        return result

    def range_z_angle(self):
        range_angles = np.nanmax(
            self._orientations, axis=0) - np.nanmin(self._orientations, axis=0)
        result = vec2rowarr(np.array([range_angles[2]]))
        result = add_name(result, self.range_z_angle.__name__)
        return result

    def std_angles(self):
        std_angles = np.nanstd(self._orientations, axis=0)
        result = vec2rowarr(np.array(std_angles))
        result = add_name(result, self.std_angles.__name__)
        return result

    def std_x_angle(self):
        std_angles = np.nanstd(self._orientations, axis=0)
        result = vec2rowarr(np.array([std_angles[0]]))
        result = add_name(result, self.std_x_angle.__name__)
        return result

    def std_y_angle(self):
        std_angles = np.nanstd(self._orientations, axis=0)
        result = vec2rowarr(np.array([std_angles[1]]))
        result = add_name(result, self.std_y_angle.__name__)
        return result

    def std_z_angle(self):
        std_angles = np.nanstd(self._orientations, axis=0)
        result = vec2rowarr(np.array([std_angles[2]]))
        result = add_name(result, self.std_z_angle.__name__)
        return result

    def ori_var_sum(self):
        var_angles = np.nanvar(self._orientations, axis=0)
        result = vec2rowarr(np.array([np.sum(var_angles)]))
        result = add_name(result, self.ori_var_sum.__name__)
        return result
    
        
    def ori_range_max(self):
        range_angles = abs(np.nanmax(
            self._orientations, axis=0) - np.nanmin(self._orientations, axis=0))
        result = vec2rowarr(np.array([np.sum(range_angles)]))
        result = add_name(result, self.ori_range_max.__name__)
        return result