Contains code for extracting features from pre-trained iNat2018 network.  

Code is from here:  
https://github.com/macaodha/inat_comp_2018  

Tested on PyTorch 0.3.0  


Standard model  
InceptionV3 trained for 75 epochs at 299x299.
Val Acc@1 60.20 acc@3 77.9  


High res model  
Start with previous model trained at 299x299 for 75 epochs and then finetuned until 97 epochs at 520x520.  
Val Acc@1 66.18 acc@3 83.32  

Run `extract_inat_feats.py` to extract the features before the last fully connected layer for a trained model.   
Change the image resolution in `inat2018_loader.py` so that it is the correct size e.g. 299 or 520.  
