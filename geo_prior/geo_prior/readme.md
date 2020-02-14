# Presence-Only Geographical Priors for Fine-Grained Image Classification - Train and Evaluation

`train_geo_net.py` trains our spatio-temporal prior on different datasets.  
`run_evaluation.py` evaluates different priors for the task of image classification.  


## Getting Started
1) Download the required datasets and metadata from our project [website](http://www.vision.caltech.edu/~macaodha/projects/geopriors/index.html) and put them in the `../data/` directory. This contains features and predictions extracted from trained image classifiers along with the location metadata. You will also need to download the training and validation annotation files for each of the [iNat datasets](https://github.com/visipedia/inat_comp). If you want to evaluate the existing trained models make sure you put them in `../models/`.  
2) Update the paths in `paths.py` so they point to the correct locations on your system.  
3) Make sure you have the package versions specified in `requirements.txt`. Model training and evaluation was performed with Python 3.7.   
