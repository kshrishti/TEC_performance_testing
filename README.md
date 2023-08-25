# TIF TEC performance testing data analysis

`tif_data_analysis.py` has all the code to run data analyses on the data collected at the TIF. Please read the documentation for each function! Especially important when running the analysis on annealing cycles because they haven't been studied too much. 

Also, most functions have commented blocks of code which plot out the results. Please plot the plots regularly - and at each step the first few times you are analysing a dataset - so that you can quickly manually verify if the algorithm is working as expected. Sometimes if the tests aren't smooth (or there are multiple things happening, like trips in the power supplies or adding/removing insulation - manual things that you know about but the algorithm can be confused by) you can have unexpected errors. 

The CSV files contain the data from the MTRS that is used in the data analysis algorithm. 
Previously, you couldn't save 8 channels' data in 1 CSV, so I ended up having multiple CSV files for each dataset (1 for each channel). However, this is now fixed, so I would recommend plotting the important channels (see below my recommendations) which will be useful for a full data analysis, and store them all in just one CSV file so it's easier to store data. Then, the CSV headers will remain the same over multiple datasets, and only the filename will change. 
You will also need an elog CSV file from that day's data collection, which contains the TEC current and voltage at each step of TEC power. Make sure the elog has headers 'Time' (I used HH:MM:SS format to extract the timestamps), 'Voltage' and 'Current'. 


Important channels: their channel mapping ('their header in the  MTRS-generated CSV file')
-------------------------------------
1. Module [I used J20 primarily]: 18-0-2 ('cms_mtdtf_dcs_1:MTRS/MTD_TIF_18/Chip_0/Channel_2.actual.value')
2. Cu housing: 33-0-2 ('cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_0/Channel_2.actual.value')
3. Cold plate: 34-1-0 ('cms_mtdtf_dcs_1:MTRS/MTD_TIF_34/Chip_1/Channel_0.actual.value')
4. LYSO: 33-1-3 ('cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_1/Channel_3.actual.value')
5. Ambient temperature sensor: 33-2-0 ('cms_mtdtf_dcs_1:MTRS/MTD_TIF_33/Chip_2/Channel_0.actual.value')
