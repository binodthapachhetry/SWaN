### What is this repository for? ###

* Classify sleep-wear, wake-wear, and non-wear from accelerometer data from the smartwatch 
* Version 0.1

### How do I get set up? ###

* Dependencies : python packages, i.e., numpy, math, pandas, scikit-learn

### How do I run the software? ###
* [STEP 1] python SWaNforTIME.py [root folder with data for all participants] [participant ID] [sampling rate]

    e.g., python SWaNforTIME.py D:/TIME_sample_data_to_test_SWaN eskimovocalizeveggie@timestudy_com 50

    For all unprocessed hourly files of the participant, this command will compute features and perform window-level
    prediction, which is required for the rule-based correction in STEP 2.

* [STEP 2] python SWaNforTIME_correctPrediction.py [root folder with data for all participants] [participant ID] [start_date] [stop_date]

    e.g., python SWaNforTIME_correctPrediction.py D:/TIME_sample_data_to_test_SWaN eskimovocalizeveggie@timestudy_com 2020-02-13 2020-02-21

    For all the hourly files within 2020-02-13 of the participant, this command will perform the rule-based correction
    of the window-level predictions.

    The final output csv file is called 'SWaN_' + dateStr +'_final.csv', which is saved inside the dateStr folder within
    the data-watch folder of the participant.

    The header of the csv file includes:
    'HEADER_TIME_STAMP','PREDICTED_SMOOTH','PROB_WEAR_SMOOTH','PROB_SLEEP_SMOOTH','PROB_NWEAR_SMOOTH'

### Who do I talk to? ###

* Binod Thapa-Chhetry
