## Multi-Modal Framework for prediction of time-to-MRI

This provides a framework for integration of multimodal imaging (DWI + FLAIR) with tabular (clinical) data
for classification and regression tasks.

This framework was used for the study 'A Deep Learning Analysis of Stroke Onset Time Prediction and Comparison to 
DWI-FLAIR Mismatch'. 

## Data:
Retrospective data of patients with acute ischemic stroke from 1000Plus (Hotter et al., 2009).

* clinical data:
  * tabular data with clinical features - clinical features were not used for this study.
  * time-to-MRI targets

* imaging data:
  * DWI and FLAIR coregistered volumes and resized to the same size of 192 x 192 x 50 voxels.
  * the volumes were merged to one volume channel-wise to provide an input for the model, this resulted in 
  volumes of size 192 x 192 x 50 x 2 voxels.

Exemplary files containing random data are stored in folder './exemplary_data' to provide the user of this framework 
an example of the data structure and file naming needed for the framework. The paths to data and file naming 
can be changed in the MAIN scripts.

Files of all three modalities (clinical, DWI and FLAIR) have to be provided to the framework,
even if not used for the training. The framework loads all of them and then matches them patients-wise to create
the data generators. Then only those modalities provided in the 'mode' variable are used for the training and evaluation
of a model.

'mode' can be one of: 'clin', 'dwi', 'flair' or combinations of those, e.g. 'clin_dwi_flair'.

## Run commands:

* create cross-validation splits:
  > python splits.py

* train and evaluate models:
  * run baseline CNN:
    > python MAIN.py -m dwi -a convnet -s dicho_tti

  * run baseline G-CNN:
    > python MAIN.py -m dwi -a groupnet -s dicho_tti

  * run G-CNN with AE regularization:
    > python MAIN.py -m dwi -a groupnet_AE -s dicho_tti

  * run vanilla AE for pretraining the model on image reconstruction task:
    > python MAIN_AE_vanilla.py -m dwi -a groupnet_vanilla_AE

  * run pretrained G-CNN:
    > python MAIN.py -m dwi -a groupnet_pretrained -s dicho_tti

  * run pretrained G-CNN with AE regularization:
    > python MAIN.py -m dwi -a groupnet_AE_pretrained -s dicho_tti