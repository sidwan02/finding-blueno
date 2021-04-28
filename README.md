# finding-blueno

## Goal

The goal of this project is to reliably generate target masks of Blueno from images. The model generated, which follows the
[UNET Architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/), detects Blueno pasted on random background images.

The code has been made generic such that one should be able to use any choice of target object and retrain the model.

This project was initially created as a Starter Project for [Brown Visual Computing](https://visual.cs.brown.edu/).

## Training

First, you must download the [Stanford Background Dataset](https://www.kaggle.com/balraj98/stanford-background-dataset) at the root of the cloned repository.

Next, please run the following commands in succession at the root of the cloned repository terminal:

1. `python .\generate_dataset.py`
2. `python .\train.py`

You may find it useful to change the `batch_size` hyperparameter in the [train file](train.py)

## Prediction

If you wish to predict Blueno for your own images, please create the following directory at the root of the cloned repository: `test_model`

Within test_model, create another folder and place your test images within it: `test_images`

Then, run the following command: `python .\predict_masks.py`

You should be able to see the predicted masks within a new folder in `test_model`: `generated_masks`

## Different Target Object

If you wish to use your own target object other than Blueno, please delete the contents of this folder: `target_original`

Make sure to paste in your own object within this folder.

Then, follow the steps earlier to train the model and give your own test images.

## References

References are contained within the code where applicable.
