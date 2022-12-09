## Clustering ESN Dynamics Using Conceptors

### Setup

To get the code to run, make sure to:
1. Download the TIMIT dataset (e.g., from https://catalog.ldc.upenn.edu/LDC93S1, or Kaggle) and place it into the
root directory. The data folder should be named TIMIT.
2. Install python 3.10 or higher
3. Install Anaconda
4. Run `conda env create -f environment.yml` from the root directory to load the dependencies into a new environment
5. Now you should be able to use the jupyter notebooks

### Abstract

For data analysis, explainability, and more, it can be of interest to identify groups and hierarchies
within the activity dynamics of recurrent neural networks (RNNs). Using conceptors to
represent RNN dynamics, this study aims to cluster the dynamics of Echo State
Networks (ESNs), a variant of RNNs. Conceptors were computed from the ESN-response to phoneme recordings of
the TIMIT dataset. These conceptors were then used to perform phoneme classification with evidences (supervised),
clustering with an adaptation of K-means (unsupervised), and hierarchical clustering with average linkage (partly
supervised). Conceptor-based phoneme classification reached a reasonable accuracy and clustering produced groups
and hierarchies that resemble existing phonetical taxonomies. I conclude that conceptors
are well-suited to classifying and clustering ESN-dynamics and the time-series that induced these dynamics.