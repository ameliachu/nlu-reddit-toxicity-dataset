#Import Dependencies
from detoxify import Detoxify
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#Read in data
data_path = 'path_to_data'
results_path = 'path_to_store_results'

df = pd.read_csv(data_path+'labelled_master_data_2022-04-14.csv')

#Initialize model
model = Detoxify('unbiased')

#Get model predictions

length = df.shape[0]

#Doing it in batches of 10 to not overload ram
inds = np.linspace(0,length, 81)
inds = [int(num) for num in inds]
df_list = []
for i in range(len(inds)-1):
  results = model.predict(list(df['comment_for_evaluation'][inds[i]:inds[i+1]].values))
  df_ind = list(df['example_id'][inds[i]:inds[i+1]].values)
  results_df = pd.DataFrame(results, index=df_ind).round(5)
  df_list.append(results_df)

#Concat preditions into one dataframe
out = pd.concat(df_list)

#Combine obscene and sexual explicit into profanity
out['profanity'] = np.maximum(out['obscene'],out['sexual_explicit'])

#Model outputs probability.  Our decision threshold is .5
out = out.round(0)

#Get separate scores for each column
col_names = list(df.columns[4:].values)
scores = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

#Loop through columns to get scores
for name in col_names:
  labels, preds = df[name], out[name]
  accuracy = accuracy_score(labels, preds)
  precision, recall, f1_score, _ = precision_recall_fscore_support(labels, preds, labels=[0.0,1.0], beta=1, zero_division=0)

  scores['accuracy'].append(accuracy)
  #Only interested in the positive class
  scores['f1'].append(f1_score[1])
  scores['precision'].append(precision[1])
  scores['recall'].append(recall[1])

#Save results in csv
scores_df = pd.DataFrame(scores, index=col_names)
scores_df.reset_index(inplace=True)
scores_df.rename(columns={'index':'category'}, inplace=True)
scores_df.to_csv(results_path + 'detoxify_baseline_results.csv')