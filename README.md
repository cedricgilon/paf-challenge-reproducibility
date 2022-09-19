# PAF Prediction Challenge Reproducibility

[_Reproducibility of Machine Learning Models for Paroxysmal Atrial Fibrillation Onset Prediction_
](https://difusion.ulb.ac.be/vufind/Record/ULB-DIPOT:oai:dipot.ulb.ac.be:2013/348586/Holdings)

Cédric Gilon<sup>1</sup>, Jean-Marie Grégoire<sup>1,2</sup>, Jérome Helinckx<sup>1</sup>, Stéphane Carlier <sup>2</sup>, Hugues Bersini<sup>1</sup>
1. IRIDIA, Université Libre de Bruxelles, Belgium
2. Département de Cardiologie, Université de Mons, Belgium

Computers in Cardiology 2022, Tampere Finland


## Dependancies

The source code is written in Python3 and is using [Poetry](https://python-poetry.org/) as virtual environment and package manager.
The package is divided between the `data` and the `src` folders.
The `src` folder contains the `features` scripts, the `models` and finally some `util`and `visualization` tools.

To create the virtual environment and install the dependancies, you can use:
```
poetry install
```

## Run models

To run the scripts, the `src` folder should be in the `PYTHONPATH`. The run each model you can use
```
python src/models/20xx_model_name/main_model_name.py
```

The results are stored in csv files and the figures are saved as png files.

## Data

The data are already in the repositroy and were downloaded from the PAF Prediction Challenge Database, which is available on the [Physionet website](https://physionet.org/content/afpdb/1.0.0/). 
Labels are also still available on the [results page](https://physionet.org/content/challenge-2001/1.0.0/).

## Licence

GNU General Public License v3.0

## References

(1) https://physionet.org/content/afpdb/1.0.0/

(2) https://physionet.org/content/challenge-2001/1.0.0/
