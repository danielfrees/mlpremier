# CNN FPL Modeling

`cnn_data.py` takes the preprocessed data and prepares it for training via the CNN. Splits
out target (the next weeks FPL score), drops collinear/ unlikely to be useful features.

TODO: Augment with another input into the fully-connected layer containing info 
about the upcoming gameweek match difficulty. 

`model.py` sets up the 1D regularized CNN, prepared for training. Model training
function also defined.

`analysis.py` plots the results of a set of CNN hyperparameter experiments