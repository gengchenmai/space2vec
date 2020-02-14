YFCC100M_GEO100 dataset from "Improving Image Classification with Location Context" - ICCV 2015.

The original dataset can be downloaded from [here](https://sites.google.com/site/locationcontext/).  
Note that the dataset does not have a defined train, val, and test split. We create one by running `make_YFCC_dataset.py`.    

Inception V3 - pre-trained imagenet, 299x299  
After 11 epochs, test set results are: Top 1 50.146, Top 5	82.453.  
