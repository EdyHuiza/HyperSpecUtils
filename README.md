# HyperSpecUtils
Java class library with utilities for processing hyperspectral images from PSI hyperspectral camera and PhenoMobileLite and training and using machine-learning pixel classification models.

Author: Alexander Ivakov

This project is not being developed. No commits please, fork it if you want to develop it.

This is a class library that contains utility functions for processing of hyperspectral images. 

To use: build a JAR and import into your project. Some example programs for typical use cases are below.

This code requires the Java port of OpenCV version 2.4.13 to run.

Example driver programs:

HyperSpecJava: Example program to open a BIL file, normalise using a white reference calibration image, train a two-class KNN classifier on the spectra using the mouse to select spectra, then save the model to an XML file.

HyperImageClassifier: Example program to open and normalise a BIL file, load a pre-trained spectral KNN classifier from an XML file, predict class for all pixels, then export all spectra from pixels falling into class 1 (this may be plant pixels) for each of 20 image regions corresponding to pots in a tray, as CSV files.

HyperImagePCA: Example program to read and normalise a BIL file, do PCA on the image (with pixels as observations and wavelengths as varaibles), plot the results as images with PCA score as image intensity, and save these images for a defined number of principal components. Optionally first apply a KNN classifier to the image and then do PCA only on the pixels falling in class 1. 

HyperImageVisualiseLDA: Example program to visualise LDA models obtained using multivariate spectral mapping as images (LDA scores as image intensity).

HyperImageClustering: Example program to apply a pre-trained KNN classifier to a BIL image, identify pixels of class 1, then apply K-means clustering to the spectra of those pixels, in each of 20 image regions corresponding to pots in a tray. 
