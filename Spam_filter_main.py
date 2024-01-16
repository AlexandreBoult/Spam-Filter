import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,ShuffleSplit


df=pd.read_table("SMSSpamCollection")

#old code
"""
vectorizer = CountVectorizer() 
wm = vectorizer.fit_transform(df['msg'].tolist())
tokens=vectorizer.get_feature_names_out()
df_vect=pd.DataFrame(data=wm.toarray(),columns=tokens)

pca=PCA(n_components=200)

reduced_df=pd.DataFrame(pca.fit_transform(df_vect))

X = df['msg']
y = df['cat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MLPClassifier(random_state=1, max_iter=200)
model.fit(X_train, y_train)
"""

vectorize_pl = make_pipeline(CountVectorizer(),FunctionTransformer(lambda x: x.toarray(), accept_sparse=True))

pl=make_pipeline(vectorize_pl,PCA(n_components=200),MLPClassifier(random_state=1,max_iter=600,early_stopping=False,learning_rate='adaptive',solver='adam',warm_start=False))

X = df['msg']
y = df['cat']

opti=0
if opti: #hyperparameters optimsation
    shuffle_split = ShuffleSplit(n_splits=5,test_size=0.3)
    X_part=X.sample(frac=0.5)
    param_grid = {"mlpclassifier__solver":['lbfgs','sgd','adam'],"mlpclassifier__warm_start":[True,False],"mlpclassifier__learning_rate":['constant','invscaling','adaptive'],"mlpclassifier__early_stopping":[True,False]}
    search=GridSearchCV(pl,param_grid,verbose=10,cv=shuffle_split).fit(X_part,y[X_part.index])
    print(search.best_params_)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pl.fit(X_train, y_train)
    y_pred = pl.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("score :", round(score, 5))