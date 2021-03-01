# SWaN_accel package

This is an algorithm to distinguish between sleep-wear, wake-wear, and non-wear in accelerometer dataset. 

To install the package, use the following pip command:
### pip install swan_accel

To import the two relevant methods from the package, type:
### from SWaN_accel import swan_first_pass, swan_second_pass

To run swan first pass algorithm:
### swan_first_pass.main(df=dataframe object, file_path=path for output file,sampling_rate=sampling rate of data)

To run swan second pass algorithm
### swan_second_pass.main(day_folder=path of the date folder)


