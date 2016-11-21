# Linear Classification and Subspaces

-Language of coding: Python
-Each cell of Jupyter Notebook is one of PCA, LDA and Naive Baye's.
-First cell is header cell, followed by data segmentation. Here we have segmented the data to classes, and computed their means.

-PCA: For analysis on MNIST Data, change the variable 'mnist' to 1. This will pass the required MNIST datasets to the cell. For wine data, the variable mnist must be set to 0, and the number of training samples must be set using the variable 'trainsize' 
-We are doing PCA using SVD. First we are changing the data to mean-centered. Thereafter, we are using these eigen vectors to project the data onto a lower dimentional subspace.

-LDA: LDA is used to make the data cleaner and easier to classify. It essentially reparates the distribution of two classes such that we can project these distributions onto a lower dimensional subspace and use a threshold point to classify it. So, we have taken the means of projections of two classes on the principal component, and set the threshold to the mid point of them. Any new test vector can be projected onto the same subspace, and compared with the threshold

-Naive Baye's: Here, we are fitting the given data to a Baye's distribution, and considering that it is gaussian in nature Thus among al the components, we just need mean and variances of the random variables, and we can classify the incomming data.
