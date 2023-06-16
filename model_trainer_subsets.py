# Inputs
import argparse
import os

# Data Processing
import pandas as pd
from datasets.load import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import set_seed

# Modeling
import torch
import torch.cuda
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

# Scoring
import numpy as np
from datasets.load import load_metric
from sklearn.metrics import f1_score

# Silence logging
import logging
import re
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])

# Save Outputs Using Pickle
import pickle

# Define Metrics to Compute
def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    f1w = f1_score(labels, predictions, average='weighted')

    ret_dict = {"accuracy": accuracy, "f1-weighed": f1w, "f1": f1}

    return ret_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='covid')
    parser.add_argument('--tokenizer', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--epochs', type=int, default=21)
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--data_path', type=str, default='None')
    parser.add_argument('--num_rows', type=str, default = '')
    parser.add_argument('--frac_train', type=float, default = '')

    args = parser.parse_args()

    # Make Sure WANDB is Disabled
    os.environ["WANDB_DISABLED"] = "true"

    # Setup GPU
    if (args.gpu) and (torch.cuda.is_available()): 
        dev = "cuda" 
    else: 
        dev = "cpu" 
    
    print("DEV: ", dev)

    device = torch.device(dev) 

    # set random seed
    set_seed(412)

    # Number of iterations per run
    iterations = 3

    accuracies = []
    f1w = []
    f1 = []

    for i in range(0,iterations):

        #----------------------------------------------------------------
        # Load and Prepare Data
        #----------------------------------------------------------------
        
        dataset = load_dataset('csv', data_files = 'data/' + args.topic + '-labeled-binary-all.csv',split='train')
                                    

        # Set the random split
        dataset_train_eval_splits = dataset.train_test_split(test_size=0.1)

        train_dataset = dataset_train_eval_splits['train']
        eval_dataset = dataset_train_eval_splits['test']

        # Subset by fraction of training set parameter
        frac = args.frac_train
        nrow = train_dataset.shape[0]
        samp_size = np.round(nrow * frac).astype(int)
        samples = np.random.choice(np.arange(0, nrow), size=samp_size, replace=False)
        train_dataset = train_dataset.select(samples)

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        # Prepare the text inputs for the model
        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)

        tokenized_train = train_dataset.map(preprocess_function, batched=True)
        tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

        # Use data_collector to convert our samples to PyTorch tensors and concatenate them with the correct amount of padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        #----------------------------------------------------------------
        # Fine Tune the Model
        #----------------------------------------------------------------

        # Load binary BERT classifier
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

        # Hand off computations to GPU
        model.to(device)

        training_args = TrainingArguments(
            output_dir="subset-logs/",
            evaluation_strategy='steps',
            num_train_epochs=args.epochs,
            # Saving models
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            #----------------------------
            log_level='debug',
            run_name= args.topic + '-' + args.model + '--' + str(args.frac_train),
            # Cuda
            no_cuda = False
            )
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )


        # Train the model
        train_result = trainer.train()

        metrics = trainer.evaluate()

        accuracies.append(metrics['eval_accuracy'])
        f1.append(metrics['eval_f1'])
        f1w.append(metrics['eval_f1-weighed'])

        # Keep track of Progress
        print('------------------------------------------------------------')
        print(args.model, args.frac_train, i)
        print('------------------------------------------------------------')

    # Load, record, and save dictionary
    if args.frac_train == 0.001:
        evalDict = {}
    else:
        evalDict = pickle.load(open('subset-logs/eval_metrics_fifteen-2.pkl', 'rb'))

    evalDict[(args.model, args.frac_train)] = [accuracies, f1w, f1]

    with open('subset-logs/eval_metrics_fifteen-2.pkl', 'wb') as openfile:
        pickle.dump(evalDict , openfile)

   