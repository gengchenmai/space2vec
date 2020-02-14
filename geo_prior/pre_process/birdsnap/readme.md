Had to re-download the BirdSnap dataset as the data was not available.

Original BirdSnap has 49,829 images we got 46,687. Note there is at least one duplicate image in the dataset - 8516871691.jpg, 8755483178.jpg is also a duplicate but we were not able to download it. The original BirdSnap has some very visually similar images, not sure how they are divided into the train and test splits.

We made a validation set by selecting 3 images per class from the train set (this was the train set we were able to download). Only about 35-45% of images have GT location. Images are not just from North America, but come from all over the world.

num train ims                 42926  
num missing train ims         2960  

num test ims                  2262  
num missing test ims          181  

num val ims                   1500  
num missing val ims           0  

Resized and converted (to jpg) all the raw birdsnap images with (run from inside the directory where the data is):  
`mogrify -geometry "800x800>" -format jpg -path ../images_sm *`


