This repo contains process_utill.py to preprocess data. For example test.tsv.
test.py provide test that checks logic on synthetic data.
demo.ipynb show example how to use process_utill script

In general process_utill script work as follows. 
Process data by chunks. Chunk size can be specified.
And store this chunks in sqlite db.
When we are processing n's chunk. After find all statistics 
we need to update our previous n-1 chunks with this new data
(That's why I use sqlite db for buffer purpose. I didn't find a way to 
efficiently implement this with additional csv file. Cannot update specific part in csv file) 
After we process all data sqlite db converted to tsv file
