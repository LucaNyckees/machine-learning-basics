# Machine Learning (CS-433)

## The Higgs Boson Project

### Team

Raphaël Mirallié, Luca Nyckees, Clarisse Schumer

---

## Virtual environment

---

Use the following command lines to create and use venv python package:

```
python3.10 -m venv venv
```

Then use the following to activate the environment:

```
source venv/bin/activate
```

You can now use pip to install any packages you need for the project and run python scripts, usually through a `requirements.txt`:

```
python -m pip install -r requirements.txt
```

When you are finished, you can stop the environment by running:

```
deactivate
```

---

## Project Organization

---

    ├── README.md          -- Top-level README.
    │
    ├── notebooks          -- Jupyter notebooks.
    │
    ├── files              -- Notes and report (Latex, pdf).
    │
    ├── data               -- data sets.
    │
    ├── requirements.txt   -- Requirements file for reproducibility.
    │
    └── src                -- Source code for use in this project.
        │
        ├── ...

### `src`

`src` contains `.py` files, including the following.

- `helper_implementations.py`, defining the basic tools used in `implementations.py`,
- `implementations.py`, where the main machine learning algorithms are implemented,
- `cleaning_data.py`, where do a bit of initial data analysis (splitting methods, standardization, treating outliers),
- `cross_validation.py`, where we set some important cross-validation methods to optimize the choice of hyperparameters.
- Some notebooks found at the extension `ML_Project1/Projet/scripts/.ipynb_checkpoints`, where all implemented methods are put to use, the training of tha data and the producing of the prediction files take place.

### `submission`

`submission` contains a list of `.csv`submission files, consisting of predictions on the given dataset.
