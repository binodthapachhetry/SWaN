import os, gzip, pickle, sys, datetime, struct
from glob import glob
import pandas as pd
import subprocess
import shutil
import numpy as np
from datetime import timedelta
from io import StringIO

from SWaN import config
from SWaN import utils
from SWaN import feature_set
pd.options.mode.chained_assignment = None  # default='warn'

# JAR = 'jar/readBinaryFile.jar'

# col = ["HEADER_TIME_STAMP","X","Y","Z"]

col = ["HEADER_TIME_STAMP","X_ACCELERATION_METERS_PER_SECOND_SQUARED",
       "Y_ACCELERATION_METERS_PER_SECOND_SQUARED","Z_ACCELERATION_METERS_PER_SECOND_SQUARED"]

MHEALTH_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

PROB_WEAR = 'PROB_WEAR'
PROB_SLEEP = 'PROB_SLEEP'
PROB_NWEAR = 'PROB_NWEAR'

ori_header = ['ORI_X_MEDIAN', 'ORI_Y_MEDIAN', 'ORI_Z_MEDIAN']


def mhealth_timestamp_parser(val):
    return datetime.datetime.strptime(val, MHEALTH_TIMESTAMP_FORMAT)

def contigous_regions_usingOri(condition):
    d = np.floor(np.absolute(np.diff(condition)))
    idx, = d.nonzero()
    idx += 1
    idx = np.r_[0, idx - 1]
    idx = np.r_[idx, condition.size - 1]

    bout_lis = []
    for i in range(len(idx) - 1):
        if i == 0:
            first = idx[i]
        else:
            first = idx[i] + 1
        second = idx[i + 1]
        bout_lis = bout_lis + [[first, second]]

    this_ar = np.asarray(bout_lis)

    return this_ar

def contigous_regions(condition):
    d = np.diff(condition)
    idx, = d.nonzero()
    idx += 1
    idx = np.r_[0, idx - 1]
    idx = np.r_[idx, condition.size - 1]

    bout_lis = []
    for i in range(len(idx) - 1):
        if i == 0:
            first = idx[i]
        else:
            first = idx[i] + 1
        second = idx[i + 1]
        bout_lis = bout_lis + [[first, second]]

    this_ar = np.asarray(bout_lis)

    return this_ar

def filterUsingZori(bout_array, fil_df, lab_str, ref_str, prob_wear, prob_sleep, prob_nwear):
    fdf = fil_df.copy()
    tmp_fdf = fil_df.copy()
    for n in range(len(bout_array)):
        ar_sub = fdf[bout_array[n][0]:bout_array[n][1] + 1]
        ar_sub_pred = ar_sub[lab_str].values[0]
        ar_sub_start = ar_sub.index[0]
        ar_sub_ori = ar_sub[ref_str].values
        bout_array_sub = contigous_regions_usingOri(ar_sub_ori)
        bout_array_sub_final = bout_array_sub + ar_sub_start
        for m in range(len(bout_array_sub_final)):
            start = bout_array_sub_final[m][0]
            end = bout_array_sub_final[m][1]
            if ar_sub_pred == 0:
                if start == end:
                    fdf.loc[start, 'PREDICTED_SMOOTH'] = 0
                    fdf.loc[start, 'PROB_WEAR_SMOOTH'] = tmp_fdf.loc[start][prob_wear]
                    fdf.loc[start, 'PROB_SLEEP_SMOOTH'] = tmp_fdf.loc[start][prob_sleep]
                    fdf.loc[start, 'PROB_NWEAR_SMOOTH'] = tmp_fdf.loc[start][prob_nwear]
                else:
                    fdf.loc[start:end, 'PREDICTED_SMOOTH'] = 1
                    fdf.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_fdf.loc[start:end][prob_sleep]
                    fdf.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_fdf.loc[start:end][prob_wear]
                    fdf.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_fdf.loc[start:end][prob_nwear]
            elif ar_sub_pred == 1:
                if start == end:
                    fdf.loc[start, 'PREDICTED_SMOOTH'] = 0
                    fdf.loc[start, 'PROB_WEAR_SMOOTH'] = tmp_fdf.loc[start][prob_sleep]
                    fdf.loc[start, 'PROB_SLEEP_SMOOTH'] = tmp_fdf.loc[start][prob_wear]
                    fdf.loc[start, 'PROB_NWEAR_SMOOTH'] = tmp_fdf.loc[start][prob_nwear]
                else:
                    fdf.loc[start:end, 'PREDICTED_SMOOTH'] = 1
                    fdf.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_fdf.loc[start:end][prob_wear]
                    fdf.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_fdf.loc[start:end][prob_sleep]
                    fdf.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_fdf.loc[start:end][prob_nwear]
            elif ar_sub_pred == 2:
                if start == end:
                    fdf.loc[start, 'PREDICTED_SMOOTH'] = 0
                    fdf.loc[start, 'PROB_WEAR_SMOOTH'] = tmp_fdf.loc[start][prob_sleep]
                    fdf.loc[start, 'PROB_SLEEP_SMOOTH'] = tmp_fdf.loc[start][prob_wear]
                    fdf.loc[start]['PROB_NWEAR_SMOOTH'] = tmp_fdf.loc[start][prob_nwear]
                else:
                    fdf.loc[start:end, 'PREDICTED_SMOOTH'] = 2
                    fdf.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_fdf.loc[start:end][prob_wear]
                    fdf.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_fdf.loc[start:end][prob_sleep]
                    fdf.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_fdf.loc[start:end][prob_nwear]
    return fdf

def lookBeforeAfter(lo_df):
    global new_lab
    df = lo_df.copy()
    tmp_df = lo_df.copy()
    tmp_ar = tmp_df['PREDICTED_SMOOTH'].values
    ff_obout_array = contigous_regions(tmp_ar)
    bout_df = pd.DataFrame(ff_obout_array, columns=['START_IND', 'STOP_IND'])
    bout_df['SIZE'] = bout_df['STOP_IND'] - bout_df['START_IND'] + 1

    start_ind = bout_df.iloc[0]['START_IND']
    stop_ind = bout_df.iloc[-1]['STOP_IND']
    size = len(bout_df.index)

    for bout_ind, bout_row in bout_df.iterrows():
        start, end, this_size = bout_row['START_IND'], bout_row['STOP_IND'], bout_row['SIZE']
        lab = tmp_df.loc[start]['PREDICTED_SMOOTH']
        bout_df.loc[bout_ind, 'LABEL'] = lab
        if lab == 1:
            if (bout_ind == len(bout_df.index) - 1) or (this_size >= 480):
                #             if(this_size >= 480):
                bout_df.loc[bout_ind, 'LABEL'] = 2
                df.loc[start:end, 'PREDICTED_SMOOTH'] = 2
                df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']

    # print('Done nonwear first')

    sleep_df = bout_df[bout_df.LABEL == 1]
    #     ref_df_short = sleep_df[(sleep_df.SIZE >= 30)]
    ref_df_short = sleep_df[(sleep_df.SIZE >= 20)]
    ref_ind_ar_short = ref_df_short.index

    # nonwear related
    nwear_df = bout_df[bout_df.LABEL == 2]
    nwear_ref_ind_ar_short = None
    if not nwear_df.empty:
        nwear_ref_ind_ar_short = nwear_df.index

    # also add nonwear vicinity
    for bout_ind, bout_row in bout_df.iterrows():
        start, end = bout_row['START_IND'], bout_row['STOP_IND']
        lab = bout_row['LABEL']
        size = bout_row['SIZE']
        if lab == 1:
            if (size < 480) and (size >= 60):
                #                 min_distance = 60

                min_distance = 20

                nwear_min_distance = 10

                up, down = ref_ind_ar_short[ref_ind_ar_short < bout_ind], ref_ind_ar_short[ref_ind_ar_short > bout_ind]
                up_dist = None
                down_dist = None

                if len(up) != 0:
                    up_ind = up[-1]
                    sub_bout_df = bout_df.loc[(bout_df.index > up_ind) & (bout_df.index < bout_ind)]
                    up_dist = sub_bout_df[sub_bout_df.LABEL == 0]['SIZE'].sum()

                if len(down) != 0:
                    down_ind = down[0]
                    sub_bout_df = bout_df.loc[(bout_df.index > bout_ind) & (bout_df.index < down_ind)]
                    down_dist = sub_bout_df[sub_bout_df.LABEL == 0]['SIZE'].sum()

                # nonwear related
                nwear_up_dist = None
                nwear_down_dist = None
                if not nwear_df.empty:
                    nwear_up = nwear_ref_ind_ar_short[nwear_ref_ind_ar_short < bout_ind]
                    nwear_down = nwear_ref_ind_ar_short[nwear_ref_ind_ar_short > bout_ind]

                    if len(nwear_up) != 0:
                        nwear_up_ind = nwear_up[-1]
                        sub_bout_df = bout_df.loc[(bout_df.index > nwear_up_ind) & (bout_df.index < bout_ind)]
                        nwear_up_dist = sub_bout_df[sub_bout_df.LABEL == 0]['SIZE'].sum()

                    if len(nwear_down) != 0:
                        nwear_down_ind = nwear_down[0]
                        sub_bout_df = bout_df.loc[(bout_df.index > bout_ind) & (bout_df.index < nwear_down_ind)]
                        nwear_down_dist = sub_bout_df[sub_bout_df.LABEL == 0]['SIZE'].sum()

                # nonwear vicinity related
                if nwear_down_dist:
                    if nwear_down_dist < nwear_min_distance:
                        # print('flip', start, end, nwear_up_dist, nwear_down_dist)
                        bout_df.loc[bout_ind, 'LABEL'] = 2
                        df.loc[start:end, 'PREDICTED_SMOOTH'] = 2
                        df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                        df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                        df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                        continue

                if nwear_up_dist:
                    if nwear_up_dist < nwear_min_distance:
                        # print('flip', start, end, nwear_up_dist, nwear_down_dist)
                        bout_df.loc[bout_ind, 'LABEL'] = 2
                        df.loc[start:end, 'PREDICTED_SMOOTH'] = 2
                        df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                        df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                        df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                        continue

                        # sleep vicinity related
                if (not up_dist) & (not down_dist):
                    # print('flip', start, end, up_dist, down_dist)
                    bout_df.loc[bout_ind, 'LABEL'] = 2
                    df.loc[start:end, 'PREDICTED_SMOOTH'] = 2
                    df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                    df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                    df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                    continue

                if up_dist and down_dist:
                    if (up_dist > min_distance) and (down_dist > min_distance):
                        # print('flip', start, end, up_dist, down_dist)
                        bout_df.loc[bout_ind, 'LABEL'] = 2
                        df.loc[start:end, 'PREDICTED_SMOOTH'] = 2
                        df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                        df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                        df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                        continue

                # print('untouched', start, end, up_dist, down_dist)

    sleep_df = bout_df[bout_df.LABEL == 1]
    ref_df_short = sleep_df[(sleep_df.SIZE >= 30)]
    ref_ind_ar_short = ref_df_short.index

    for bout_ind, bout_row in bout_df.iterrows():
        start, end = bout_row['START_IND'], bout_row['STOP_IND']
        lab = bout_row['LABEL']
        size = bout_row['SIZE']
        if lab == 1:
            if (size < 60) and (size > 30):
                min_distance = 30
                up, down = ref_ind_ar_short[ref_ind_ar_short < bout_ind], ref_ind_ar_short[ref_ind_ar_short > bout_ind]
                up_dist = None
                down_dist = None

                if len(up) != 0:
                    up_ind = up[-1]
                    sub_bout_df = bout_df.loc[(bout_df.index > up_ind) & (bout_df.index < bout_ind)]
                    up_dist = sub_bout_df[sub_bout_df.LABEL == 0]['SIZE'].sum()

                if len(down) != 0:
                    down_ind = down[0]
                    sub_bout_df = bout_df.loc[(bout_df.index > bout_ind) & (bout_df.index < down_ind)]
                    down_dist = sub_bout_df[sub_bout_df.LABEL == 0]['SIZE'].sum()

                if (not up_dist) & (not down_dist):
                    # print('flip', start, end, up_dist, down_dist)
                    bout_df.loc[bout_ind, 'LABEL'] = 0
                    df.loc[start:end, 'PREDICTED_SMOOTH'] = 0
                    df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                    df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                    df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                    continue
                if not up_dist:
                    if down_dist:
                        if down_dist > min_distance:
                            # print('flip', start, end, up_dist, down_dist)
                            bout_df.loc[bout_ind, 'LABEL'] = 0
                            df.loc[start:end, 'PREDICTED_SMOOTH'] = 0
                            df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                            df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                            df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                            continue

                if not down_dist:
                    if up_dist:
                        if up_dist > min_distance:
                            # print('flip', start, end, up_dist, down_dist)
                            bout_df.loc[bout_ind, 'LABEL'] = 0
                            df.loc[start:end, 'PREDICTED_SMOOTH'] = 0
                            df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                            df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                            df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                            continue

                if up_dist and down_dist:
                    if (up_dist > min_distance) and (down_dist > min_distance):
                        # print('flip', start, end, up_dist, down_dist)
                        bout_df.loc[bout_ind, 'LABEL'] = 0
                        df.loc[start:end, 'PREDICTED_SMOOTH'] = 0
                        df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                        df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                        df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                        continue

                # print('untouched', start, end, up_dist, down_dist)

    sleep_df = bout_df[bout_df.LABEL == 1]
    ref_df_short = sleep_df[(sleep_df.SIZE >= 30)]
    ref_ind_ar_short = ref_df_short.index

    for bout_ind, bout_row in bout_df.iterrows():
        start, end = bout_row['START_IND'], bout_row['STOP_IND']
        lab = bout_row['LABEL']
        size = bout_row['SIZE']
        if lab == 1:
            if size <= 30:
                min_distance = 30

                up, down = ref_ind_ar_short[ref_ind_ar_short < bout_ind], ref_ind_ar_short[ref_ind_ar_short > bout_ind]
                up_dist = None
                down_dist = None

                if len(up) != 0:
                    up_ind = up[-1]
                    sub_bout_df = bout_df.loc[(bout_df.index > up_ind) & (bout_df.index < bout_ind)]
                    up_dist = sub_bout_df[sub_bout_df.LABEL == 0]['SIZE'].sum()

                if len(down) != 0:
                    down_ind = down[0]
                    sub_bout_df = bout_df.loc[(bout_df.index > bout_ind) & (bout_df.index < down_ind)]
                    down_dist = sub_bout_df[sub_bout_df.LABEL == 0]['SIZE'].sum()

                if (not up_dist) & (not down_dist):
                    # print('flip', start, end, up_dist, down_dist)
                    bout_df.loc[bout_ind, 'LABEL'] = 0
                    df.loc[start:end, 'PREDICTED_SMOOTH'] = 0
                    df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                    df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                    df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                    continue
                if not up_dist:
                    if down_dist:
                        if down_dist > min_distance:
                            # print('flip', start, end, up_dist, down_dist)
                            bout_df.loc[bout_ind, 'LABEL'] = 0
                            df.loc[start:end, 'PREDICTED_SMOOTH'] = 0
                            df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                            df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                            df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                            continue

                if not down_dist:
                    if up_dist:
                        if up_dist > min_distance:
                            # print('flip', start, end, up_dist, down_dist)
                            bout_df.loc[bout_ind, 'LABEL'] = 0
                            df.loc[start:end, 'PREDICTED_SMOOTH'] = 0
                            df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                            df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                            df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                            continue

                if up_dist and down_dist:
                    if (up_dist > min_distance) or (down_dist > min_distance):
                        # print('flip', start, end, up_dist, down_dist)
                        bout_df.loc[bout_ind, 'LABEL'] = 0
                        df.loc[start:end, 'PREDICTED_SMOOTH'] = 0
                        df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                        df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                        df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                        continue

                # print('untouched', start, end, up_dist, down_dist)

    # smooth the wear between sleep period
    tmp_ar = df['PREDICTED_SMOOTH'].values
    ff_obout_array = contigous_regions(tmp_ar)
    bout_df = pd.DataFrame(ff_obout_array, columns=['START_IND', 'STOP_IND'])
    bout_df['SIZE'] = bout_df['STOP_IND'] - bout_df['START_IND'] + 1

    tmp_df = df.copy()
    for i in range(len(bout_df) - 1):
        # print(i)
        start, end, this_size = bout_df.loc[i, 'START_IND'], bout_df.loc[i, 'STOP_IND'], bout_df.loc[i, 'SIZE']
        lab = df.loc[start]['PREDICTED_SMOOTH']

        if this_size <= 20:

            prev_start = None
            next_start = None

            if i != 0:
                prev_start, prev_end, prev_size = bout_df.loc[i - 1, 'START_IND'], bout_df.loc[i - 1, 'STOP_IND'], \
                                                  bout_df.loc[i - 1, 'SIZE']
                prev_lab = df.loc[prev_start]['PREDICTED_SMOOTH']

            if i != len(bout_df):
                next_start, next_end, next_size = bout_df.loc[i + 1, 'START_IND'], bout_df.loc[i + 1, 'STOP_IND'], \
                                                  bout_df.loc[i + 1, 'SIZE']
                next_lab = df.loc[next_start]['PREDICTED_SMOOTH']

            if prev_start and next_start:
                if prev_size >= next_size:
                    new_lab = prev_lab
                else:
                    new_lab = next_lab

            elif prev_start:
                new_lab = prev_lab
            elif next_start:
                new_lab = next_lab

            if lab == 2:
                # print(start,end,lab,new_lab)
                if new_lab == 0:
                    df.loc[start:end, 'PREDICTED_SMOOTH'] = 0
                    df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                    df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                    df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                if new_lab == 1:
                    df.loc[start:end, 'PREDICTED_SMOOTH'] = 1
                    df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                    df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                    df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']

    # smooth the wear between sleep period
    tmp_ar = df['PREDICTED_SMOOTH'].values
    ff_obout_array = contigous_regions(tmp_ar)
    bout_df = pd.DataFrame(ff_obout_array, columns=['START_IND', 'STOP_IND'])
    bout_df['SIZE'] = bout_df['STOP_IND'] - bout_df['START_IND'] + 1

    tmp_df = df.copy()
    for i in range(len(bout_df) - 1):
        # print(i,len(bout_df))
        start, end, this_size = bout_df.loc[i, 'START_IND'], bout_df.loc[i, 'STOP_IND'], bout_df.loc[i, 'SIZE']
        lab = df.loc[start]['PREDICTED_SMOOTH']

        if this_size <= 20:

            prev_start = None
            next_start = None

            if i != 0:
                prev_start, prev_end, prev_size = bout_df.loc[i - 1, 'START_IND'], bout_df.loc[i - 1, 'STOP_IND'], \
                                                  bout_df.loc[i - 1, 'SIZE']
                prev_lab = df.loc[prev_start]['PREDICTED_SMOOTH']

            if i != len(bout_df):
                next_start, next_end, next_size = bout_df.loc[i + 1, 'START_IND'], bout_df.loc[i + 1, 'STOP_IND'], \
                                                  bout_df.loc[i + 1, 'SIZE']
                next_lab = df.loc[next_start]['PREDICTED_SMOOTH']

            if prev_start and next_start:
                if prev_size >= next_size:
                    new_lab = prev_lab
                else:
                    new_lab = next_lab

            elif prev_start:
                new_lab = prev_lab
            elif next_start:
                new_lab = next_lab

            if lab == 0:
                # print(start,end,lab,new_lab)
                if new_lab == 2:
                    df.loc[start:end, 'PREDICTED_SMOOTH'] = 2
                    df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                    df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                    df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                if new_lab == 1:
                    df.loc[start:end, 'PREDICTED_SMOOTH'] = 1
                    df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                    df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                    df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']

    # smooth the wear between sleep period
    tmp_ar = df['PREDICTED_SMOOTH'].values
    ff_obout_array = contigous_regions(tmp_ar)
    bout_df = pd.DataFrame(ff_obout_array, columns=['START_IND', 'STOP_IND'])
    bout_df['SIZE'] = bout_df['STOP_IND'] - bout_df['START_IND'] + 1

    tmp_df = df.copy()
    for i in range(len(bout_df) - 1):
        start, end, this_size = bout_df.loc[i, 'START_IND'], bout_df.loc[i, 'STOP_IND'], bout_df.loc[i, 'SIZE']
        lab = df.loc[start]['PREDICTED_SMOOTH']

        if this_size <= 20:

            prev_start = None
            next_start = None

            if i != 0:
                prev_start, prev_end, prev_size = bout_df.loc[i - 1, 'START_IND'], bout_df.loc[
                    i - 1, 'STOP_IND'], \
                                                  bout_df.loc[i - 1, 'SIZE']
                prev_lab = df.loc[prev_start]['PREDICTED_SMOOTH']

            if i != len(bout_df):
                next_start, next_end, next_size = bout_df.loc[i + 1, 'START_IND'], bout_df.loc[
                    i + 1, 'STOP_IND'], \
                                                  bout_df.loc[i + 1, 'SIZE']
                next_lab = df.loc[next_start]['PREDICTED_SMOOTH']

            if prev_start and next_start:
                if prev_size >= next_size:
                    new_lab = prev_lab
                else:
                    new_lab = next_lab

            elif prev_start:
                new_lab = prev_lab
            elif next_start:
                new_lab = next_lab

            if lab == 1:
                # print(start, end, lab, new_lab)
                if new_lab == 0:
                    df.loc[start:end, 'PREDICTED_SMOOTH'] = 0
                    df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']
                    df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                    df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                if new_lab == 2:
                    df.loc[start:end, 'PREDICTED_SMOOTH'] = 1
                    df.loc[start:end, 'PROB_WEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_WEAR_SMOOTH']
                    df.loc[start:end, 'PROB_SLEEP_SMOOTH'] = tmp_df.loc[start:end]['PROB_NWEAR_SMOOTH']
                    df.loc[start:end, 'PROB_NWEAR_SMOOTH'] = tmp_df.loc[start:end]['PROB_SLEEP_SMOOTH']

    return df

def daterange(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + timedelta(n)

def correctPredictionsSingleDate(folder, dStr, sampling_rate=80):
    dObj = datetime.datetime.strptime(dStr, "%Y-%m-%d")

    prev = dObj - datetime.timedelta(days=1)
    next = dObj + datetime.timedelta(days=1)

    prevStr = prev.strftime("%Y-%m-%d")
    nextStr = next.strftime("%Y-%m-%d")

    oriDF = pd.DataFrame(data=None)

    prevFolder = os.path.join(folder, 'data-watch', prevStr)
    if os.path.isdir(prevFolder):
        daily_feature_file = os.path.join(prevFolder,"SWaN_" + prevStr+"_dailyfeatures.csv")
        if(os.path.isfile(daily_feature_file)):
            odf = pd.read_csv(daily_feature_file, header=0, skiprows=0, sep=',', compression="infer", quotechar='"',
                                  parse_dates=['HEADER_TIME_STAMP','START_TIME','STOP_TIME'], date_parser=mhealth_timestamp_parser)
            oriDF = pd.concat([oriDF, odf], ignore_index=True)
        else:
            odf = get_daywise_prediction_df(prevFolder, sampling_rate)
            oriDF = pd.concat([oriDF, odf], ignore_index=True)

    thisFolder = os.path.join(folder, 'data-watch', dStr)
    if os.path.isdir(thisFolder):
        daily_feature_file = os.path.join(thisFolder, "SWaN_" + dStr + "_dailyfeatures.csv")
        if (os.path.isfile(daily_feature_file)):
            odf = pd.read_csv(daily_feature_file, header=0, skiprows=0, sep=',', compression="infer", quotechar='"',
                              parse_dates=['HEADER_TIME_STAMP','START_TIME','STOP_TIME'], date_parser=mhealth_timestamp_parser)
            oriDF = pd.concat([oriDF, odf], ignore_index=True)
        else:
            odf = get_daywise_prediction_df(thisFolder, sampling_rate)
            oriDF = pd.concat([oriDF, odf], ignore_index=True)

    nextFolder = os.path.join(folder, 'data-watch', nextStr)
    if os.path.isdir(nextFolder):
        daily_feature_file = os.path.join(nextFolder, "SWaN_" + nextStr + "_dailyfeatures.csv")
        if (os.path.isfile(daily_feature_file)):
            odf = pd.read_csv(daily_feature_file, header=0, skiprows=0, sep=',', compression="infer", quotechar='"',
                              parse_dates=['HEADER_TIME_STAMP','START_TIME','STOP_TIME'], date_parser=mhealth_timestamp_parser)
            oriDF = pd.concat([oriDF, odf], ignore_index=True)
        else:
            odf = get_daywise_prediction_df(nextFolder, sampling_rate)
            oriDF = pd.concat([oriDF, odf], ignore_index=True)

    if oriDF.empty:
        print("No data found for this day or previous and next day.")
        return

    oriDF.sort_values(by='HEADER_TIME_STAMP', inplace=True)

    if oriDF.dropna().empty:
        print('No prediction data in the folder: '+folder +' for data: ' + dStr)
        return

    outPath = os.path.join(folder, 'data-watch', dStr, 'SWaN_' + dStr + '_final.csv')

    oriDF.replace({'PREDICTED': {2: 1}}, inplace=True)
    oriDF['PREDICTED_SMOOTH'] = None
    oriDF['PROB_WEAR_SMOOTH'] = None
    oriDF['PROB_SLEEP_SMOOTH'] = None
    oriDF['PROB_NWEAR_SMOOTH'] = None
    tmp_ar = oriDF['PREDICTED'].values

    # compute contigous bouts based on window-level prediction
    obout_array = contigous_regions(tmp_ar)

    # in case only one type of bout present in the data
    if (obout_array.shape[0] == 1) & (oriDF.iloc[0]['PREDICTED'] == 1):
        oriDF['PREDICTED_SMOOTH'] = 2
        oriDF['PROB_WEAR_SMOOTH'] = oriDF[PROB_WEAR]
        oriDF['PROB_SLEEP_SMOOTH'] = oriDF[PROB_NWEAR]
        oriDF['PROB_NWEAR_SMOOTH'] = oriDF[PROB_SLEEP]
        # oriDF.to_csv(outPath, index=False, float_format='%.3f')
        # return

    elif (obout_array.shape[0] == 1) & (oriDF.iloc[0]['PREDICTED'] == 2):
        oriDF['PREDICTED_SMOOTH'] = 2
        oriDF['PROB_WEAR_SMOOTH'] = oriDF[PROB_WEAR]
        oriDF['PROB_SLEEP_SMOOTH'] = oriDF[PROB_SLEEP]
        oriDF['PROB_NWEAR_SMOOTH'] = oriDF[PROB_NWEAR]
        # oriDF.to_csv(outPath, index=False, float_format='%.3f')
        # return

    elif (obout_array.shape[0] == 1) & (oriDF.iloc[0]['PREDICTED'] == 0):
        oriDF['PREDICTED_SMOOTH'] = 0
        oriDF['PROB_WEAR_SMOOTH'] = oriDF[PROB_WEAR]
        oriDF['PROB_SLEEP_SMOOTH'] = oriDF[PROB_SLEEP]
        oriDF['PROB_NWEAR_SMOOTH'] = oriDF[PROB_NWEAR]
        # oriDF.to_csv(outPath, index=False, float_format='%.3f')
        # return

    else:
        # use z orientation to filter
        f_odf = filterUsingZori(obout_array, oriDF, 'PREDICTED', 'ORI_Z_MEDIAN', PROB_WEAR, PROB_SLEEP, PROB_NWEAR)
        oriDF = lookBeforeAfter(f_odf)

        # l_f_odf = lookBeforeAfter(f_odf)
        # l_f_odf.to_csv(outPath, index=False, float_format='%.3f')

    currDateObj = datetime.datetime.strptime(dStr, "%Y-%m-%d")
    nextDateObj = currDateObj + datetime.timedelta(days=1)

    mask = (oriDF['HEADER_TIME_STAMP'] > currDateObj) & (oriDF['HEADER_TIME_STAMP'] < nextDateObj)
    final_df = oriDF.loc[mask][
        ['HEADER_TIME_STAMP', 'PREDICTED_SMOOTH', 'PROB_WEAR_SMOOTH', 'PROB_SLEEP_SMOOTH', 'PROB_NWEAR_SMOOTH']]
    print(datetime.datetime.now().strftime("%H:%M:%S") + " Finished performing rule-based filtering.")

    final_df.to_csv(outPath, index=False, float_format='%.3f')

def correctPredictions(folder, startdStr, stopdStr, sampling_rate=80):
    startdObj = datetime.datetime.strptime(startdStr, "%Y-%m-%d")
    stopdObj = datetime.datetime.strptime(stopdStr, "%Y-%m-%d")

    # prev = startdObj - datetime.timedelta(days=1)
    # next = stopdObj + datetime.timedelta(days=1)

    prev = startdObj
    next = stopdObj

    pid = os.path.basename(folder)

    for dt in daterange(prev, next):
        dStr = dt.strftime("%Y-%m-%d")

        refPath = os.path.join(folder, 'data-watch', dStr, 'SWaN_' + dStr + '_final.csv')

        if not os.path.exists(refPath):
            print("Performing rule-based filtering for participant: " + pid + " for date: " + dStr)
            correctPredictionsSingleDate(folder, dStr, sampling_rate=sampling_rate)
            print("Done rule-based filtering for participant: " + pid + " for date: " + dStr)
        else:
            print("Final rule-based filtered file present for participant: " + pid + " for date " + dStr)

def readBinary(inFile):
    tz = os.path.basename(inFile).split('.')[2].split('-')[-1]

    hourdiff = int(tz[1:3])
    minutediff = int(tz[3:])

    if (tz[0] == 'M'):
        hourdiff = -int(tz[1:3])
        minutediff = -int(tz[3:])

    file = open(inFile, "rb")
    b = file.read(20)
    diction = {}
    i = 0
    while len(b) >= 20:
        t = int.from_bytes(b[0:8], byteorder='big')
        x = struct.unpack('>f', b[8:12])[0]
        y = struct.unpack('>f', b[12:16])[0]
        z = struct.unpack('>f', b[16:20])[0]
        diction[i] = {'time': t, 'x': x, 'y': y, 'z': z}
        i = i + 1

        b = file.read(20)

    df = pd.DataFrame.from_dict(diction, "index")
    df.columns = col
    df['HEADER_TIME_STAMP'] = pd.to_datetime(df['HEADER_TIME_STAMP'], unit='ms') + \
                              datetime.timedelta(hours=hourdiff) + datetime.timedelta(minutes=minutediff)
    return df


def get_daywise_prediction_df(inFolder, sampling_rate=80):
    try:
        import importlib.resources as pkg_resources
    except ImportError:
        # Try backported to PY<37 `importlib_resources`.
        import importlib_resources as pkg_resources

    # trainedModel = pickle.load(open(config.modelPath, "rb"))
    # standardScalar = pickle.load(open(config.scalePath, "rb"))

    trainedModel = pickle.load(pkg_resources.open_binary(__package__,config.modelPath))
    standardScalar = pickle.load(pkg_resources.open_binary(__package__,config.scalePath))


    final_day_df = pd.DataFrame()
    for file in sorted(
            glob(os.path.join(inFolder, '*/AndroidWearWatch-AccelerationCalibrated-NA.*.sensor.baf'))):
        outfilePath = os.path.join(os.path.dirname(file),
                                   ".".join(os.path.basename(file).split('.')[1:-2]) + ".prediction.csv")
        if os.path.exists(outfilePath):
            print(outfilePath + " present. Reading that file.")
            odf = pd.read_csv(outfilePath, header=0, skiprows=0, sep=',', compression="infer", quotechar='"',
                              parse_dates=[0], date_parser=mhealth_timestamp_parser)
            final_day_df = pd.concat([final_day_df, odf], ignore_index=True)
            continue

        print(datetime.datetime.now().strftime("%H:%M:%S") + ' Reading binary file :' + file)

        try:
            df = readBinary(file)
        except:
            print('Issue with converting baf file to a dataframe - ' + file)
            continue

        print(datetime.datetime.now().strftime("%H:%M:%S") + ' Computing feature set for :' + file)

        time_grouper = pd.Grouper(key='HEADER_TIME_STAMP', freq='30s')
        grouped_df = df.groupby(time_grouper)


        feature_df = pd.DataFrame()
        for name, group in grouped_df:
            if len(group) > sampling_rate * 15:
                op = get_feature_sleep(group, sampling_rate)
                op['HEADER_TIME_STAMP'] = name
                feature_df = pd.concat([feature_df, op], ignore_index=True)

        final_feature_df = feature_df.dropna(how='any', axis=0, inplace=False)
        if final_feature_df.empty:
            print("No feature row computed or remaining after dropping zero rows. So not moving to prediction.")
            continue

        final_feature_df.rename(columns={'HEADER_TIME_STAMP': 'START_TIME'}, inplace=True)
        final_feature_df['HEADER_TIME_STAMP'] = final_feature_df['START_TIME']
        final_feature_df['STOP_TIME'] = final_feature_df['START_TIME'] + pd.Timedelta(seconds=30)

        print(datetime.datetime.now().strftime("%H:%M:%S") + " Performing window-level classification for :" + file)
        final_feature_df = final_feature_df.dropna()
        subfdata = final_feature_df[config.feature_lis]
        sfdata = standardScalar.transform(subfdata)
        prediction_prob = trainedModel.predict_proba(sfdata)
        prediction = np.argmax(prediction_prob, axis=1)
        p = prediction.reshape((-1, 1))
        final_feature_df["PREDICTED"] = p
        final_feature_df['PROB_WEAR'] = prediction_prob[:, 0]
        final_feature_df['PROB_SLEEP'] = prediction_prob[:, 1]
        final_feature_df['PROB_NWEAR'] = prediction_prob[:, 2]

        final_day_df = pd.concat([final_day_df, final_feature_df], ignore_index=True)

    dateStr = os.path.basename(inFolder)
    outPath = os.path.join(inFolder, "SWaN_" + dateStr + "_dailyfeatures.csv")

    final_day_df.to_csv(outPath, index=False, float_format="%.3f")
    print("Created prediction file:" + outPath)
    return final_day_df

def get_feature_sleep(tdf, sampling):
    X_axes = utils.as_float64(tdf.values[:, 1:])
    result_axes = feature_set.compute_extra_features(X_axes, sampling)
    return result_axes


def main(sampling_rate=None,input_folder=None,file_path=None,startdateStr=None,stopdateStr=None):
    # len_args = len(sys.argv)
    # if len_args < 4:
    #     print("Syntax error. It should be one of these formats:\n"
    #           "python SWaNforTIME_final.py SAMPLING RATE INPUT_FOLDER PARTICIPATN_ID/FILE_PATH_WITH_PARTICIPANT_ID\n"
    #           "python SWaNforTIME_final.py SAMPLING RATE INPUT_FOLDER PARTICIPANT_ID/FILE_PATH_WITH_PARTICIPANT_ID YYYY_MM_DD\n "
    #           "python SWaNforTIME_final.py SAMPLING RATE INPUT_FOLDER PARTICIPANT_ID/FILE_PATH_WITH_PARTICIPANT_ID YYYY_MM_DD YYYY_MM_DD\n")
    #     return

    if (startdateStr is None) and (stopdateStr is None):
        print("doing for all dates")
        # sampling_rate = int(sys.argv[1])
        # input_folder = sys.argv[2]
        # file_path = sys.argv[3]
        if not (file_path.endswith('.txt')):
            pid = file_path + "@timestudy_com"
            sub_folder = os.path.join(input_folder, pid)
            final_input_folder = os.path.join(input_folder, pid)

            date_lis = [os.path.basename(x) for x in glob(os.path.join(final_input_folder, 'data-watch', '*'))]

            for dateStr in date_lis:
                final_input_folder = os.path.join(input_folder, pid, 'data-watch', dateStr)

                if not os.path.isdir(final_input_folder):
                    print("Missing folder: " + final_input_folder)
                    continue

                refPath = os.path.join(final_input_folder, 'SWaN_' + dateStr + '_final.csv')

                if not os.path.exists(refPath):
                    print("Performing rule-based filtering for participant: " + pid + " for date: " + dateStr)
                    correctPredictionsSingleDate(sub_folder, dateStr, sampling_rate=sampling_rate)
                    print("Done filtering predictions.")
                else:
                    print("Final rule-based filtered file present.")

            return

        if not (os.path.isfile(file_path)):
            print("File with participant ids does not exist")
            return

        with open(file_path) as f:
            content = f.readlines()
        pidLis = [x.strip() for x in content]

        for pid in pidLis:
            pid = pid + "@timestudy_com"

            sub_folder = os.path.join(input_folder, pid)
            final_input_folder = os.path.join(input_folder, pid)

            date_lis = [os.path.basename(x) for x in glob(os.path.join(final_input_folder, 'data-watch', '*'))]

            for dateStr in date_lis:
                final_input_folder = os.path.join(input_folder, pid, 'data-watch', dateStr)

                if not os.path.isdir(final_input_folder):
                    print("Missing folder: " + final_input_folder)
                    continue

                refPath = os.path.join(final_input_folder, 'SWaN_' + dateStr + '_final.csv')

                if not os.path.exists(refPath):
                    print("Performing rule-based filtering for participant: " + pid + " for date: " + dateStr)
                    correctPredictionsSingleDate(sub_folder, dateStr, sampling_rate=sampling_rate)
                    print("Done filtering predictions.")
                else:
                    print("Final rule-based filtered file present.")

        return

    if (startdateStr) and (stopdateStr is None):
        dateStr = startdateStr
        # print("doing for a specific date")
        # sampling_rate = int(sys.argv[1])
        # input_folder = sys.argv[2]
        # file_path = sys.argv[3]
        # dateStr = sys.argv[4]

        if not (file_path.endswith('.txt')):
            pid = file_path + "@timestudy_com"
            sub_folder = os.path.join(input_folder, pid)
            final_input_folder = os.path.join(input_folder, pid, 'data-watch', dateStr)

            if not os.path.isdir(final_input_folder):
                print("Missing folder: " + final_input_folder)
                return

            refPath = os.path.join(final_input_folder, 'SWaN_' + dateStr + '_final.csv')

            if not os.path.exists(refPath):
                print(datetime.datetime.now().strftime("%H:%M:%S") + " Performing rule-based filtering for participant: " + pid + " for date: " + dateStr)
                correctPredictionsSingleDate(sub_folder, dateStr, sampling_rate=sampling_rate)
                print("Done filtering predictions.")
            else:
                print("Final rule-based filtered file present " + refPath)

            return

        if not (os.path.isfile(file_path)):
            print("File with participant ids does not exist")
            return

        with open(file_path) as f:
            content = f.readlines()
        pidLis = [x.strip() for x in content]

        for pid in pidLis:
            pid = pid + "@timestudy_com"
            sub_folder = os.path.join(input_folder, pid)
            final_input_folder = os.path.join(input_folder, pid, 'data-watch', dateStr)

            if not os.path.isdir(final_input_folder):
                print("Missing folder: " + final_input_folder)
                continue

            refPath = os.path.join(final_input_folder, 'SWaN_' + dateStr + '_final.csv')

            if not os.path.exists(refPath):
                print("Performing rule-based filtering for participant: " + pid + " for date: " + dateStr)
                correctPredictionsSingleDate(sub_folder, dateStr, sampling_rate=sampling_rate)
                print("Done filtering predictions.")
            else:
                print("Final rule-based filtered file present.")

        return

    if (startdateStr and stopdateStr):
        print("doing for a date range")

        # sampling_rate = int(sys.argv[1])
        # input_folder = sys.argv[2]
        # file_path = sys.argv[3]
        # startdateStr = sys.argv[4]
        # stopdateStr = sys.argv[5]

        if not (file_path.endswith('.txt')):
            pid = file_path + "@timestudy_com"
            sub_folder = os.path.join(input_folder, pid)
            first_input_folder = os.path.join(input_folder, pid, 'data-watch', startdateStr)

            if not os.path.isdir(first_input_folder):
                print("Missing folder: " + first_input_folder)
                return

            last_input_folder = os.path.join(input_folder, pid, 'data-watch', stopdateStr)

            if not os.path.isdir(last_input_folder):
                print("Missing folder: " + last_input_folder)
                return

            print(
                "Performing rule-based filtering for participant: " + pid + " for date between: " + startdateStr + " and " + stopdateStr)
            correctPredictions(sub_folder, startdateStr, stopdateStr, sampling_rate=sampling_rate)
            print("Done filtering predictions.")

            return

        if not (os.path.isfile(file_path)):
            print("File with participant ids does not exist")
            return
        with open(file_path) as f:
            content = f.readlines()
        pidLis = [x.strip() for x in content]

        for pid in pidLis:
            pid = pid + "@timestudy_com"
            sub_folder = os.path.join(input_folder, pid)
            first_input_folder = os.path.join(input_folder, pid, 'data-watch', startdateStr)

            if not os.path.isdir(first_input_folder):
                print("Missing folder: " + first_input_folder)
                continue

            last_input_folder = os.path.join(input_folder, pid, 'data-watch', stopdateStr)

            if not os.path.isdir(last_input_folder):
                print("Missing folder: " + last_input_folder)
                continue

            print(
                "Performing rule-based filtering for participant: " + pid + " for date between: " + startdateStr + " and " + stopdateStr)
            correctPredictions(sub_folder, startdateStr, stopdateStr, sampling_rate=sampling_rate)
            print("Done filtering predictions.")

# if __name__ == "__main__":
#     main()
