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
