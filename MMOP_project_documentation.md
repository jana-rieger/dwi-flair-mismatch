# Multi-Modal Outcome Prediction (MMOP)

This project provides a framework for integration of multimodal imaging (DWI + FLAIR) with tabular (clinical) data for outcome prediction in acute stroke patients.

## Database

* DB: 1000plus
* Inclusion criteria:
* Exclusin criteria:
* [Patients Table](add link here)

## Data pre-processing

#### Inclusion/ Exclusion criteria
Inclusion criteria: Supratentorial stroke 
Exclusion criteria: *Inconsistent imaging dimensions or poor imaging quality

##### Summary of excluded patients:
* No. of patients excluded for inadequate image size, corrupted file or missing data: //add here on final DB//

#### FLAIR reslicing
FLAIR dimensions were defined according the the most common lower dimensionality of included FLAIR images: (232,256,25)

* The existing FLAIR images with the following dimensions were resliced to the above dimensions: (232, 256, 26) , (464, 512, 25), (464, 512, 26), (256, 256, 31)
* The existing FLAIR images with the following dimensions were excluded: (464,512,19), (464,512,20), (464, 512, 22) --> 3 pts. in total

The scipt for images copy and FLAIR reslicing can be found [here](https://git2.brainsurety.com/machine-learning/mmop/blob/master/Data_Preprocessing/Image_pre_processing.ipynb).

#### Centralization of numerical data
Numerical data (i.e. not categorical) is centralized in the Pipeline ADT using the preprocess function. 
In the current version it is applied in the generate_DUFs step, i.e. by default on the raw tabular data (before splits to training/ validation / test sets)


## Abstract Data Types (ADT)
Two ADTs were defined for the scope of the project:
* __*DataGenerator*__ - defines a referential storage of data allows to generate aggregated data batches in a scalable manner
* __*Model*__ - defines a ML model object
*  __*Pipeline*__ - consists the pipeline for creating DUFs, loading architecture, validation and training of the model

Full documentation of the ADT and how to use them can be found at [ADT](https://git2.brainsurety.com/machine-learning/mmop/tree/master/ADT)


## Models architecture

#### Literature on relevant architectures
* Co-Attentive Cross-Modal Deep Learning for Medical Evidence Synthesis and Decision Making: https://arxiv.org/pdf/1909.06442.pdf


## Results

