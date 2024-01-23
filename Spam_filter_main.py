from imports import *


def train_model(opti,ratio,df,par_dict):
    list_of_models={"MLPClassifier":("mlpc",MLPClassifier()),"KNeighborsClassifier":("knc",KNeighborsClassifier()),"RandomForestClassifier":("rfc",RandomForestClassifier()),"SVC":("svc",SVC())}
    list_of_aliases=[e[0] for e in list(list_of_models.values())]

    X = df['msg']
    y = df['cat']
    y[y=='spam']=1
    y[y=='ham']=0
    y=y.astype('int')

    preproc=Pipeline(steps=[("tfidf",TfidfVectorizer()),("tsvd",TruncatedSVD())])
    smote=SVMSMOTE()
    model=list_of_models[par_dict["model_name"]][1]
    model_alias=list_of_models[par_dict["model_name"]][0]

    keys=[key for key in par_dict.keys()]
    for e in keys:
        if (not model_alias in e) and e.split("__")[0] in list_of_aliases:
            par_dict.pop(e)

    model_name=par_dict.pop("model_name")
    no_smote=par_dict.pop("svmsmote__off")

    par_dict1,par_dict2,par_dict3={},{},{}
    for key in par_dict.keys():
        match key.split("__")[0]:
            case "tfidf":
                par_dict1[key]=par_dict[key]
            case "tsvd":
                par_dict1[key]=par_dict[key]
            case "svmsmote":
                par_dict2[key.split("__")[1]]=par_dict[key]
            case model_alias:
                par_dict3[key.split("__")[1]]=par_dict[key]

    preproc.set_params(**par_dict1)
    smote.set_params(**par_dict2)
    model.set_params(**par_dict3)

    if opti: #hyperparameters optimsation
        pl=Pipeline(steps=[("tfidf",TfidfVectorizer()),("tsvd",TruncatedSVD()),(model_alias,model)])
        shuffle_split = StratifiedShuffleSplit(n_splits=5, test_size=1-ratio, random_state=2)
        X_part=X.sample(frac=0.5)
        param_grid = {"MLPClassifier":{'mlpc__hidden_layer_sizes': [(int(1.5*n),n) for n in range(50,110,10)],'mlpc__alpha': [x/1000000 for x in range(1,105,5)]},
        "KNeighborsClassifier":{'knc__p':[1,2],'knc__n_neighbors':range(1,60,5)},
        "RandomForestClassifier":{'rfc__n_estimators':range(100,200,5)},
        "SVC":{'svc__C':range(1,11),'svc__degree':range(2,5)}}[model_name]
        search=RandomizedSearchCV(pl,param_grid,verbose=10,cv=shuffle_split,n_iter=30,random_state=2,scoring='f1_macro').fit(X_part,y[X_part.index])
        best_par=search.best_params_
        best_par['model_name']=model_name
        return best_par
    elif ratio != 1:
        X_preproc=preproc.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_preproc, y, test_size=1-ratio, random_state=2, stratify=y)
        if no_smote : X_res,y_res=X_train,y_train
        else : X_res,y_res=smote.fit_resample(X_train,y_train)
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)
        cm=confusion_matrix(y_test, y_pred, labels=[0,1])
        return preproc,model,accuracy_score(y_test, y_pred),cm
    else :
        X_preproc=preproc.fit_transform(X)
        if no_smote : X_res,y_res=X_train,y_train
        else : X_res,y_res=smote.fit_resample(X_preproc,y)
        model.fit(X_res, y_res)
        y_pred = model.predict(X_preproc)
        cm=confusion_matrix(y, y_pred, labels=[0,1])
        return preproc,model,accuracy_score(y, y_pred),cm

def test_msg(preproc,model,msg) : return model.predict(preproc.transform([msg]))[0]