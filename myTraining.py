import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == '__main__':
    #read the data
    df = pd.read_csv('heartdata.csv')
    train, test = data_split(df,0.2)
    X_train = train[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].to_numpy()
    X_test = test[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].to_numpy()
    
    Y_train = train[['target']].to_numpy().reshape(243,)
    Y_test = test[['target']].to_numpy().reshape(60,)
    
    clf = LogisticRegression()
    clf= LogisticRegression(max_iter=10000)
    clf.fit(X_train, Y_train)

  

    inputFeatures = [21,1,1,125,196,0,1,120,0,0,0,0,1]
    infProb = clf.predict_proba([inputFeatures])[0][1]

    filename = 'finalized_model.sav'
    joblib.dump(clf,filename)

    print('Accuracy is',infProb)

    