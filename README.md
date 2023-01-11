# AMLS_assignment22_23-

## Initialising the conda environment

The conda environment is created using the following command:

```
conda env create -f environment.yml
```

When the process is finished activate the environment with

```
conda activate ml-final
```

Finally, the library **dlib** needs to be installed manually using

```
conda install -c conda-forge dlib

```

You're all set!

## Folder structure

This repository has the following folder structure:

```
|__ A1                  ## Code for Task A1
|__ A2                  ## Code for Task A2
|__ B1                  ## Code for Task B1
|__ B2                  ## Code for Task B2
|__ Datasets            ## Emply folder with the datasets
|__ .gitignore          
|__ environment.yml     ## Conda environment file
|__ main.py             ## Main file running every task
|__ README.md
```

### A1

- [gender_detection.py](./A1/gender_detection.py): This file runs calls the feature extraction
functions for this task and trains the model. It prints the prediction and the accuracy of the model

- [lab2_landmarks.py](./A1/lab2_landmarks.py): This file peforms the data preprocessing and extract faces
from images. It is based on this [paper](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)

### A2

- [smiling_detection.py](./A2/smiling_detection.py): This file runs calls the feature extraction
functions for this task and trains the model. It prints the prediction and the accuracy of the model

- [lab2_landmarks.py](./A2/lab2_landmarks.py): This file peforms the data preprocessing and extract faces
from images.

### B1

- [face_shape_detection.py](./B1/face_shape_detection.py): This file runs calls the feature extraction
functions and trains the model. It also perfoms the PCA algorithm on each image. It prints the prediction and the accuracy of the model

- [features_extraction.py](./B1/features_extraction.py): This file peforms the data preprocessing and extract faces
from images.

### B2

- [eye_color_detection.py](./B2/eye_color_detection.py): his file runs calls the feature extraction
functions and trains the model. It prints the prediction and the accuracy of the model.

- [features_extraction.py](./B2/feature_extraction.py): This file peforms the data preprocessing and extract faces
from images. It find the eye and extracts it from the image