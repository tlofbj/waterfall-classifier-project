# Waterfall Classifier Project

This project demonstrates binary classification of standard SATNOGS waterfall images based on the presence of signal in the images using a Support Vector Machine (SVM) trained over a sample from NOAA-19 at 137.1 MHz. More data may be added later to improve the model.

## Versions

* **1.0.0** - 87.5% Accuracy. Last edited Feb 23, 2025.

## Dataset

* Training Set: 27 images with signal and 17 without
* Testing Set: 12 images with signal and 12 without

## Usage

* Raw waterfall images downloaded from https://network.satnogs.org/observations/ are stored in `data/train/raw` and `data/test/raw` respectively. Those stored in the former will be used only for training while the latter only for testing.
* Run `src/utils/preprocess.py` to modify the raw images stored in `data/.../raw` directories. Processed images are removed of margin, resized to 256x256, and converted to grayscale. They are stored in separate directories `data/train/processed` and `data/test/processed`. All code should be run from the project folder to ensure valid relative paths.
* Run `src/train.py` to train the model. Training uses data from `data/train/processed`. The model will be saved in the `models` directory. `-h` for more options.
* Run `src/eval.py` to evaluate the model. Evaluation uses data from `data/test/processed'. `-h` for more options.
* Run `src/predict.py <image_path>` to predict the presence of signal in a single raw waterfall image. Output 1 is "signal" and output 0 is "no_signal". Any valid image path may be used but it has to be an unedited waterfall download. `-h` for more options.
* Adjust paths and hyperparameters as needed.

## Dependencies

* `scikit-learn` - SVC class, evaluation tools, etc.
* `opencv-python` - Image preprocessing
* `joblib` - Saving and loading models
* `numpy` - Numpy arrays
* `argparse` - Command line argument retrieving
* `termcolor` - Command line colors

## Contributors

This project was created by Tata Li and mentored by Mitch McLean, February, 2025.

