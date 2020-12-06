# MALIS project: Monument classification

This is the repository of the MALIS project of Hugo DANET, Victor MASIAK and Julien THOMAS (EURECOM, 2020).

## Run the code

* Clone this repository locally: `git clone git@github.com:vim951/MALIS-project.git`
* Navigate to the repository folder: `cd MALIS-project`
* Download required files: `python3 init_directory.py csv pdb`

## Preprocessing

## Neural Network

### Architecture

Conv2D, MaxPooling2D, Dropout, Flatten, Dense

### Adjust weights because of imbalanced data

Scikit learning

### Validation with training set

10% of the training set used to validate the model

## Visualization

## Sources

* https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d
* https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275
* https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
