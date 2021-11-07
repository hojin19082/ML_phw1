Programming Homework 1
Auto ML for classificaion

FindBestModel(df,target, scalers = None, encoders = None, models = None):

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
