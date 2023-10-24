from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import plotly.express as px

nb_model = CategoricalNB()


def rename_labels(df_labeled, relabel_col):

    # rename lablels: rank of mean of US_equity_MA65 in each kmeans cluster(label) 
    df_summary = df_labeled.groupby("label")[relabel_col].describe()
    #df_summary['relabel'] = df_summary['mean'].rank(ascending= False).astype(int)
    df_summary['relabel'] = df_summary['mean'].rank().astype(int)

    # add column 'relabel' to df_labeled, keep the index of df_labeled
    df_relabeled = df_labeled.join(df_summary['relabel'], on = 'label')

    return df_relabeled

def build_dataset(relabels, window_period):

    Transition_table = pd.DataFrame(relabels, columns=['T']) # create a column T as Today's label
    Transition_table['Last_median_03'] = Transition_table['T'].shift(1).rolling(window=3).median() # compute the median of the previous 3 days
    Transition_table['Last_median_10'] = Transition_table['T'].shift(1).rolling(window=10).median() # compute the median of the previous 10 days
    Transition_table['Last_median_'+str(window_period)] = Transition_table['T'].shift(1).rolling(window=window_period).median() # compute the median of the previous window period
    # Transition_table['Last_mean_'+str(window_period)] = Transition_table['T'].shift(1).rolling(window=window_period).mean().round(0) # compute the mean of the previous window period
    Transition_table['Last_max_'+str(window_period)] = Transition_table['T'].shift(1).rolling(window=window_period).max() # compute the max of the previous window period
    Transition_table['Last_min_'+str(window_period)] = Transition_table['T'].shift(1).rolling(window=window_period).min() # compute the min of the previous window period
    dataset = Transition_table.iloc[window_period+1:,:] # remove NA rows

    return dataset


def naive_bayes(dataset, nb_model, if_eval):
    
    X = dataset.drop(['T'], axis=1)
    y = dataset['T']

    Last_X = X.iloc[-1] # get today's value

    if if_eval:
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

        nb_model.fit(X_train, y_train)
        y_pred = nb_model.predict(X_test)
        # predit the last day #predict_proba return: array-like of shape (n_samples, n_classes)
        last_pred  = nb_model.predict_proba([Last_X]).round(2)[0] # make prediction
        accuracy = accuracy_score(y_test, y_pred)
    
    else:
        nb_model.fit(X, y)
        y_pred = nb_model.predict(X)
        last_pred  = nb_model.predict_proba([Last_X]).round(2)[0] # make prediction
        accuracy = None

    return last_pred, y_pred, accuracy


def display(df_labeled, relabel_col, monitor_period):

    df_relabeled = rename_labels(df_labeled, relabel_col)
    #mapping of labels and relabels
    mapping = dict(zip(df_relabeled["label"], df_relabeled["relabel"]))

    fig1 = px.area(
        df_relabeled[-monitor_period:],
        x=df_relabeled[-monitor_period:].index,
        y=df_relabeled[-monitor_period:]["relabel"],
        labels=dict(x="Date", y="Cycle Stage"),
        title=f"Historical Trajectory of {relabel_col}",
        height=350, width=500,
        )
    

    relabels = list(df_relabeled["relabel"])

    # Bayes model probability
    window_list = [i for i in range(20, 500) if i % 10 == 0]
    probability = np.array([0])

    for window_period in window_list:

        dataset = build_dataset(relabels, window_period)
        result  = naive_bayes(dataset, nb_model, if_eval = False) # the list of probability for each culster
        pro_i = result[0]
        #acc_i = result[2]
        probability = np.add(pro_i, probability)
        #accurancy = accurancy+acc_i

    n = len(window_list)
    probability = ( probability/n).round(3)  # avarage probability for each culster
    probability = pd.DataFrame(probability)
    probability.index = probability.index+1

    fig2 = px.bar(
        probability,
        height=350, width=500,
        labels=dict(index="Cycle", value="Probability"),
        title="Transition Probability"
        )
    fig2.update(layout_showlegend=False)

    # statistics of each cluster
    summary_table = df_relabeled.groupby("relabel")[relabel_col].describe()

    # summary
    summary_table['Chance'] = probability
    summary_table['Equity_rally_odds'] = summary_table['mean'] * summary_table['Chance']
    summary_table['Equity_rally_STD'] = summary_table['std'] * summary_table['Chance']
    Mean = summary_table.Equity_rally_odds.sum().round(3)
    Std = summary_table.Equity_rally_STD.sum().round(3)

    return fig1, fig2, summary_table, mapping, Mean, Std
