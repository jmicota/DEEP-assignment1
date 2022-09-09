Deep Learning Assignment 1 - September 2022
Reykjavik University

Predicting stock prices based on several .txt files (in data folder) containing arrays of some indicators per days of years 2017-2020.

Predictions are made for 3 companies (!) at the same time, network output is an integer of range 0-7 being later on converted to boolean 3-tuple,
where every n-th boolean means 'the price of company with id n will increase the next day'.

- Deep_Assignment1_py.py - copy of google colab notebook for local implementation
- model.py - contains Model class with neural network model implementation
- predictor.py - contains Predictor class
- simulator.py - class provided by professor, with simulate() method for network testing
- stock_preprocessing.py - custom data preprocessing with pandas, tensors, DataLoaders, tensor Datasets
