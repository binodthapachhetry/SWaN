import SWaN.spectrum as spectrum
import SWaN.orientation as orientation
import pandas as pd
import SWaN.config as config
import SWaN.utils as utils
import SWaN.energy as energy
from numpy.linalg import norm

def compute_extra_features(X,sampling):
   
    feature_list = []

    spectrum_feature_extractor = spectrum.FrequencyFeature(X, sr=sampling)
    spectrum_feature_extractor.fft()
    spectrum_feature_extractor.peaks()
    feature_list.append(spectrum_feature_extractor.dominant_frequency())
    feature_list.append(spectrum_feature_extractor.dominant_frequency_power())
    feature_list.append(spectrum_feature_extractor.total_power())


    ori_feature_extractor = orientation.OrientationFeature(X, subwins=config.winSize)
    ori_feature_extractor.estimate_orientation(unit='deg')
    feature_list.append(ori_feature_extractor.ori_x_median())
    feature_list.append(ori_feature_extractor.ori_y_median())
    feature_list.append(ori_feature_extractor.ori_z_median())
    feature_list.append(ori_feature_extractor.ori_var_sum())
    feature_list.append(ori_feature_extractor.ori_range_max())

    X_vm = utils.vec2colarr(norm(X, ord=2, axis=1))
    energy_feature_extractor = energy.EnergyFeature(X_vm, subwins=30)
    energy_feature_extractor.get_energies()
    feature_list.append(energy_feature_extractor.smv_energy_sum())
    feature_list.append(energy_feature_extractor.smv_energy_var())

    result = pd.concat(feature_list, axis=1)
    return result
