import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.simplefilter(action='ignore')

def FindBestModel(df,target, scalers = None, encoders = None, models = None):
    '''
    For each model, we find a combination of k-value, scaling, encoding, score, and parameters.
    print best score and best combination for each model.

    :param df: The Wisconsin Cancer dataset preprocessed
    :param target: target feature's name
    :param scalers: list of scalers
            None: [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
            if you want to scale other ways, then put the sclaer in list

    :param encoders: list of encoders
        None: [OrdinalEncoder(), LabelEncoder() OneHotEncoder()]
        if you want to use only one, put a encoder in list

    :param models: list of models
        None: [DecisionTreeClassifier(criterion='gini'), DecisionTreeClassifier(criterion='entropy'),
        LogisticRegression(), SVC()]
        if you want to fit other ways, then put the model in the list
    '''

    # Separate the target and all features other than the target
    X = df.drop(target, axis=1)
    y = df[target]

    # Dividing the numerical and object features of the X dataset for encoding and scaling.
    X_category = X.select_dtypes(include='object')
    cate_empty = X_category.empty
    X_numerical = X.select_dtypes(exclude='object')

    if encoders == None:
        encode = [OrdinalEncoder(),LabelEncoder(),OneHotEncoder()]
    else: encode = encoders

    if scalers == None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    else:scale = scalers

    if models == None:
        model = ['DecisionTreeClassifier(criterion=gini)', 'DecisionTreeClassifier(criterion=entropy)', 'LogisticRegression()', 'SVC()']
    else: model = models

    # Storing the best score, best params, k value of k-folds(2-10) and scaling and encoding functions used for each model.
    # DecisionTreeClassifier(criterion ='gini')
    best_score_gini = 0
    best_cv_gini = 0
    best_gini_param = []
    best_gini_scale_encode = []

    # DecisionTreeClassifier(criterion ='entropy')
    best_score_entropy = 0
    best_cv_entropy = 0
    best_entropy_param = []
    best_entropy_scale_encode = []

    # LogisticRegression ()
    best_score_logistic = 0
    best_cv_logistic = 0
    best_logistic_param = []
    best_logistic_scale_encode = []

    # SVC()
    best_score_svm = 0
    best_cv_svm = 0
    best_svm_param = []
    best_svm_scale_encode = []

    folds = range(2,11)
    # Making the best combination by pulling one by one from the lists where k-value, scaling, encoding, and models are stored.
    for k in folds:
        for i in scale:
            for j in encode:
                scaling= i
                scaling = pd.DataFrame(scaling.fit_transform(X_numerical))

                # In the data set used, there is no need to actually encode it, so it is only implemented.
                if j == OrdinalEncoder() and cate_empty is False:
                    enc = j
                    enc = enc.fit_transform(X_category)
                    new_df = pd.concat([scaling, enc], axis=1)
                elif j == LabelEncoder() and cate_empty is False:
                    enc = j
                    enc = enc.fit_transform(X_category)
                    new_df = pd.concat([scaling, enc], axis=1)
                elif j == OneHotEncoder() and cate_empty is False:
                    dum = pd.DataFrame(pd.get_dummies(X_category))
                    new_df = pd.concat([scaling, dum], axis=1)
                else:
                    new_df = scaling

                for m in model:
                    # Separate the data set into train and test sets.
                    X_train, X_test, y_train, y_test = train_test_split(new_df, y, test_size=0.3)

                    '''
                       Since the parameters used for each model are different, declare them in params in advance.
                       Using the declared params, the best parameters and scores are extracted through gridSearchCV.
                       At this time, the scaling and encoding used are also extracted.
                    '''

                    if m == 'DecisionTreeClassifier(criterion=gini)':
                        dec1_params = {'max_depth':[3,5,7,9,11],
                                    'min_samples_split':[3,5,7,9],
                                    'min_samples_leaf':[3,5,7],
                                    'max_features' : [3,7,9],
                                    'max_leaf_nodes' : [20, 30, 50 ,100]}
                        dec1 = DecisionTreeClassifier(criterion='gini')
                        dec1_turned = GridSearchCV(dec1, dec1_params, cv=k)
                        dec1_turned.fit(X_train, y_train)
                        score = dec1_turned.best_score_
                        if score > best_score_gini:
                            best_score_gini = score
                            best_cv_gini = k
                            best_gini_param = dec1_turned.best_params_
                            best_gini_scale_encode = [i,j]

                    elif m == 'DecisionTreeClassifier(criterion=entropy)':
                        dec2_params = {'max_depth':[3,5,7,9,11],
                                    'min_samples_split':[3,5,7,9],
                                    'min_samples_leaf':[3,5,7],
                                    'max_features' : [3,7,9],
                                    'max_leaf_nodes' : [20, 30, 50 ,100]}
                        dec2 = DecisionTreeClassifier(criterion='entropy')
                        dec2_turned = GridSearchCV(dec2, dec2_params, cv=k)
                        dec2_turned.fit(X_train, y_train)
                        score = dec2_turned.best_score_
                        if score > best_score_entropy:
                            best_score_entropy = score
                            best_cv_entropy = k
                            best_entropy_param = dec2_turned.best_params_
                            best_entropy_scale_encode = [i, j]

                    elif m == 'LogisticRegression()':
                        log_params = {'solver': ['lbfgs', 'liblinear'],
                                      'penalty': [None,'l2'],
                                      'max_iter' : [1, 10, 50, 100],
                                      'C': [0.1,0.5,1,5,10,20]}
                        log = LogisticRegression()
                        log_turned = GridSearchCV(log, log_params, cv=k)
                        log_turned.fit(X_train, y_train)
                        score = log_turned.best_score_
                        if score > best_score_logistic:
                            best_score_logistic= score
                            best_cv_logistic = k
                            best_logistic_param = log_turned.best_params_
                            best_logistic_scale_encode = [i, j]

                    else:
                        svm_params = {'kernel': ['linear', 'rbf'],
                                      'gamma': [0.01,0.03,0.1,0.3,1,3],
                                        'C': [1,2,3,5,10,15]}
                        svm = SVC()
                        svm_turned = GridSearchCV(svm,svm_params, cv=k)
                        svm_turned.fit(X_train, y_train)
                        score = svm_turned.best_score_
                        if score > best_score_svm:
                            best_score_svm = score
                            best_cv_svm = k
                            best_svm_param = svm_turned.best_params_
                            best_svm_scale_encode = [i, j]

    # For each model, we print the combination with the highest score.
    print("Best score for DecisionTree(gini): ", best_score_gini)
    print("DecisionTree(gini) best parameter: ", best_gini_param)
    print("Best k value of K-folds(2-10): ", best_cv_gini)
    print("Scaling and encoding: ", best_gini_scale_encode, end='')
    print('\n')

    print("Best score for DecisionTree(entropy): ", best_score_entropy)
    print("DecisionTree(entropy) best parameter: ", best_entropy_param)
    print("Best k value of K-folds(2-10): ", best_cv_entropy)
    print("Scaling and encoding: ", best_entropy_scale_encode, end='')
    print('\n')

    print("Best score for LogisticRegression: ", best_score_logistic)
    print("LogisticRegression best parameter: ", best_logistic_param)
    print("Best k value of K-folds(2-10): ", best_cv_logistic)
    print("Scaling and encoding: ", best_logistic_scale_encode, end='')
    print('\n')

    print("Best score for SVM: ", best_score_svm)
    print("SVM best parameter: ", best_svm_param)
    print("Best k value of K-folds(2-10): ", best_cv_svm)
    print("Scaling and encoding: ", best_svm_scale_encode)


df = pd.read_csv('breast-cancer-wisconsin.data')
# It is specified because there is no column name in the data set.
df.columns = ['id','thickness','size_uniformity','shape_uniformity','adhesion','epithelial_size',
              'bare_nucleoli','bland_chromatin','normal_nucleoli','mitoses','class']

# df.info()
# Preprocessing the dataset
df = df.drop(['id'], axis=1)
df = df.replace('?', np.nan)
df = df.fillna(axis=0, method='ffill')
df['bare_nucleoli'] = df['bare_nucleoli'].astype('category')

# Create a heatmap that shows the correlation between features
list_corr = ['thickness','size_uniformity','shape_uniformity','adhesion','epithelial_size',
              'bare_nucleoli','bland_chromatin','normal_nucleoli','mitoses','class']
sns.heatmap(df[list_corr].corr(), annot=True, linecolor="black")
plt.show()

# Execute a function that prints the best combination and score for each model
FindBestModel(df, 'class')