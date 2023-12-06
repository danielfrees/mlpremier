# CNN FPL Modeling

`preprocess.py` takes the general cleaned data and prepares it for training via the CNN. Splits
out time windows & target (the next weeks FPL score), drops collinear/ unlikely to be useful features. Also handles train/test/val splits with stratification for player skill, 


`model.py` sets up the 1D regularized CNN, prepared for training. Model training
function also defined.

`analysis.py` plots the results of a set of CNN hyperparameter experiments

[Data scraped from `vaastav`'s FPL Data Repository](https://github.com/vaastav/Fantasy-Premier-League)
