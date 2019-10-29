# Data Scientist Nanodegree - Capstone project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Needed package (development version):
- Python versions 3.*.
- Jupyter
- Spark

No installation needed, just clone the GitHub.
To be ale to run the jupyter, you will need to unzip the data file. As it was bigger than the max file size allowed on github.

## Project Motivation<a name="motivation"></a>

I am currently embarked on an Udacity Nanodegree. Which will be concluded, successfully I hope, by this repository and the work related to it.

For this project, I will be using Sparkify data, to create a Spark prediction model.
The aim is to train a model to determined if users will churn, upgrade or downgrade from service.

## File Descriptions <a name="files"></a>

The jupyter notebook in the jupyter directory is allowing once the data source as been unzipped to 
run the training and prediction on the mini dataset.
You have also a html file in the azure_run_logs directory that shows you the processing that has been done on the big dataset.
Using Azure infrastructure. Unfortunately the results block are harder to export than it looks.

```bash
- azure_run_logs
|- Full_sparkify_upgrade.html       # big dataset processing

- data
|- mini_sparkify_event_data.zip     # Zipped json file

- jupyter
|- sparkify.ipynb                   # mini dataset processing
|- sparkify.html                    # static html version
|- workspace_utils.py               # code not to timeout

- models
|- churn_model.save                 # trained model for churn
|- upgrade_model.save               # trained model for upgrade
|- downgrade_model.save             # trained model for downgrade

- README.md
- requirements.txt
```

## Results<a name="results"></a>

Using a serie of transformers (even a custom one), I am using gradientBoostedTree Classifier.
* F1 Score on the rebalanced train dataset is 0.98
* F1 Score on the training dataset is 0.14
* F1 Score on the test dataset (30%) is 0.097

This is the [link](https://medium.com/@pierre.bedlow/sparkify-predicting-churn-in-a-music-streaming-service-using-logs-17564357f9ad) to the medium post providing explanation to this work.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to [Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025) and Spakify for the data.