import os, datetime
from glob import glob
import pandas as pd

import numpy as np
from datetime import timedelta

pd.options.mode.chained_assignment = None  # default='warn'

PROB_WEAR = 'PROB_WEAR'
PROB_SLEEP = 'PROB_SLEEP'
PROB_NWEAR = 'PROB_NWEAR'

MHEALTH_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

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
    o_fdf = fil_df.copy(deep=True)
    fdf = fil_df.copy(deep=True)
    tmp_fdf = fil_df.copy(deep=True)

    for n in range(len(bout_array)):
        ar_sub = o_fdf[bout_array[n][0]:bout_array[n][1] + 1]
        ar_sub_pred = ar_sub[lab_str].values[0]
        ar_sub_start = bout_array[n][0]
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
                    fdf.loc[start, 'PROB_WEAR_SMOOTH'] = tmp_fdf.loc[start][prob_nwear]
                    fdf.loc[start, 'PROB_SLEEP_SMOOTH'] = tmp_fdf.loc[start][prob_sleep]
                    fdf.loc[start]['PROB_NWEAR_SMOOTH'] = tmp_fdf.loc[start][prob_wear]
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

def correctPredictionsSingleDate(folder, dStr, mode):
    dObj = datetime.datetime.strptime(dStr, "%Y-%m-%d")

    prev = dObj - datetime.timedelta(days=1)
    next = dObj + datetime.timedelta(days=1)

    prevStr = prev.strftime("%Y-%m-%d")
    nextStr = next.strftime("%Y-%m-%d")

    oriDF = pd.DataFrame(data=None)

    prevFolder = os.path.join(folder, prevStr)
    for feature_file in sorted(glob(os.path.join(prevFolder, '*/AndroidWearWatch-AccelerationCalibrated-NA.*.feature.csv.gz'))):
        odf = pd.read_csv(feature_file, header=0, skiprows=0, sep=',', compression="infer", quotechar='"',
                          parse_dates=['HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'],
                          date_parser=mhealth_timestamp_parser)
        oriDF = pd.concat([oriDF, odf], ignore_index=True)


    thisFolder = os.path.join(folder, dStr)
    for feature_file in sorted(glob(os.path.join(thisFolder, '*/AndroidWearWatch-AccelerationCalibrated-NA.*.feature.csv.gz'))):
        odf = pd.read_csv(feature_file, header=0, skiprows=0, sep=',', compression="infer", quotechar='"',
                          parse_dates=['HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'],
                          date_parser=mhealth_timestamp_parser)
        oriDF = pd.concat([oriDF, odf], ignore_index=True)

    nextFolder = os.path.join(folder, nextStr)
    for feature_file in sorted(glob(os.path.join(nextFolder, '*/AndroidWearWatch-AccelerationCalibrated-NA.*.feature.csv.gz'))):
        odf = pd.read_csv(feature_file, header=0, skiprows=0, sep=',', compression="infer", quotechar='"',
                          parse_dates=['HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'],
                          date_parser=mhealth_timestamp_parser)
        oriDF = pd.concat([oriDF, odf], ignore_index=True)

    if oriDF.empty:
        print("No data found for this day or previous and the following day.")
        return

    oriDF.sort_values(by='HEADER_TIME_STAMP', inplace=True)
    oriDF.reset_index(drop=True,inplace=True)

    if oriDF.dropna().empty:
        print('No prediction data in the folder: '+folder +' for data: ' + dStr)
        return

    outPath = os.path.join(folder, dStr, 'SWaN_' + dStr + '_final.csv')

    if mode == 'Yes':
        outPath = os.path.join(folder, dStr, 'SWaN_' + dStr + '_debug.csv')
    else:
        outPath = os.path.join(folder, dStr, 'SWaN_' + dStr + '_final.csv')

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

    if mode == 'Yes':
        final_df = oriDF
    else:
        final_df = oriDF.loc[mask][
            ['START_TIME', 'STOP_TIME', 'PREDICTED_SMOOTH', 'PROB_WEAR_SMOOTH', 'PROB_SLEEP_SMOOTH', 'PROB_NWEAR_SMOOTH']]

    final_df.rename(columns={'PREDICTED_SMOOTH':'PREDICTION'}, inplace=True)
    final_df['PREDICTION'].replace({0:'Wear',1:'Sleep',2:'Nonwear'}, inplace=True)

    print(datetime.datetime.now().strftime("%H:%M:%S") + " Finished performing rule-based filtering.")

    final_df.to_csv(outPath, index=False, float_format='%.3f', compression='infer')

def main(day_folder=None, debug='No'):
    if (day_folder is None):
        print("Must enter day folder path.")
        return

    if(not os.path.isdir(day_folder)):
        print("Folder does not exists.")
        return

    tmp_tup = os.path.split(day_folder)
    inFold = tmp_tup[0]
    dateSt = tmp_tup[1]

    if debug == 'Yes':
        path = os.path.join(day_folder,'SWaN_' + dateSt + '_debug.csv')
    else:
        path = os.path.join(day_folder,'SWaN_' + dateSt + '_final.csv')

    if(os.path.exists(path)):
        print("Second pass output file aleady exists.")
        return

    correctPredictionsSingleDate(inFold,dateSt,debug)
