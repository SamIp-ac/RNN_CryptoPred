# RNN_CrptoPred
Using SimpleRNN, GRU, LSTM to do the 'Forecasting on Cryptocurrency Price' with small dataset and its modification


Introduction

In recent years, Cryptocurrencies have been a hot topic. It is a form of digital product with a decentralized structure. The purpose of Cryptocurrencies is to build up a market without control of governments and third parties.

This project is going to find out the predicting power of Recurrent Neural Network (RNN) in a small dataset. For the RNN part, this report will use python. The modification part will also be included as well. 

In this project, 10 different cryptocurrency stocks will be used for experiments’ generalization. A small dataset will be used in this project. Since there is much research on RNN models proving that the LSTM model will always perform the best in similar tesk, the worst is SimpleRNN. However, they mostly undergo an ideal situation. It is valuable to compare the performance when a small dataset is used. Since it is not always perfect in the real world, the data may not be good enough. The cost of collecting data may be so high in special situations. Therefore, deep learning with a small dataset is a good research idea.

In this project, we want to find out:

Which RNN model is the best ?
How to modify the RNN models ?
What is their performance under different settings?


Dataset

The data download from Yahoo Finance. Each .csv file includes 1 year + 7 days data, it is a small dataset. The first 360 days will be used for training and the last 7 days will be used for testing. Since cryptocurrency stock is not stopped by holidays. It means that we do not need to worry about the data preprocessing will be too much trouble.


RNN model

Since the time series data is autocorrelated, i.e. the day i data may related to day i - 1, the day i - 1 data may related to day i - 2. The RNN model can simulate this property by adding the previous data with weights and bias to the new data in the same layer. The relation between these data is not only perpendicular, but also parallel. That is why RNN works for time series time like stock or NLP. Three kinds of RNN will be used. They are Simple RNN (SimpleRNN), Long Short-term Memory (LSTM) and Gate Recurrent Unit (GRU). We set 5 as the timestep.


Mod1: Using different data sizes and timestep for training
Mod3: Normalization and use sigmoid function
Mod4: Model selection method and hyperparameters tuning
Mod5: Using different architecture of model, 
1:Conv1d as data preprocessing with a dense layer. 
2: Using Bidirectional model
Mod6: Mix different methods---method 1, 4

