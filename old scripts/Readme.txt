Script execution order

1) in psg_edf_2_hdf.py call psg_to_hdf()  and neuroon_to_hdf which will save the hdf database in parsed_data/hdf/ folder.

	than any channel can be loaded using psg_edf_2hdf.load_psg(channel_name) or load_neuroon() very fast

2) in explore_eeg.py call parallell_parse to save all parsed data to csv

TODO: TRY CROSS CORRELATION AS AUTOCORRELATION