This repo contains process_utill.py to preprocess data. For example test.tsv
test.py provide test that checks logic on syntetic data.
demo.ipynb show example how to use process_utill script

In general process_utill script work as follow. 
Process data by chunks that can be specified.
And store this chunks in sqlite db.
When we process n chunk. Find all statistics 
we need to update our previous n-1 chunks with this new data
(That's why I use sqlite db for buffer purpose. I didn't find a way to 
efficiently implement this only with csv file) 
After we process all data sqlite db converted to tsv file