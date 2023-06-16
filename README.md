# Classification performance thresholds for BERT-based models on COVID-19 Twitter misinformation

## Paper Link:
// 

## File and Directory Descriptions

* **model_trainer_subsets** : The central script used to train BERT models given a data directory and fraction value of training data to be used.
* **run_models.bat** : Contains the Windows Powershell commands for running the primary python script 'model_trainer_subsets.py'
* **data-tokenizer** : Data formatting notebook.
* **subset_linear_test.ipynb** : notebook used to train and test linear model comparisons using Word2Vec and logistic classifier.

## How to run

* Hydrate Tweets by id under the *data/* directory.
* Clean and format data using *data-tokenizer.ipynb*
* Run *model_trainer_subsets.py* by desired increments of training data by modifying *run_models.bat* or some similar shell script.
* Load results in *Results_Visualizations.ipynb* to produce training curves.
* Optionally run *subset_linear_test.ipynb* for linear model comparison to BERT models.