# CNN FPL Modeling

`preprocess.py` takes the general cleaned data and prepares it for training via the CNN. Splits
out time windows & target (the next weeks FPL score), drops collinear/ unlikely to be useful features. Also handles train/test/val splits with stratification for player skill, 

`model.py` sets up the 1D regularized CNN, and sets up training regime.

`evaluate.py` used for result evaluation, plotting, gridsearch analysis + plots.

`experiment.py` defines the gridsearch experiments. `> python experiment.py` will run the current hyperparameter grid defined in the experiment setup. 

`config.py` contains global hyperparameters.

`cnn.ipynb` is where most of my dynamic analysis using the CNN scripts lies.

