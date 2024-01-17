import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,ShuffleSplit,RandomizedSearchCV


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

pl=make_pipeline(vectorize_pl,PCA(n_components=200),MLPClassifier(random_state=1,solver='sgd',max_iter=400,hidden_layer_sizes=(120, 80, 40),alpha=0.05))

X = df['msg']
y = df['cat']

opti=0
if opti: #hyperparameters optimsation
    shuffle_split = ShuffleSplit(n_splits=5,test_size=0.3)
    X_part=X.sample(frac=0.5)
    param_grid = {
    'mlpclassifier__hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30), (100,)],
    'mlpclassifier__max_iter': [50, 100, 150, 200, 300, 400],
    'mlpclassifier__activation': ['tanh', 'relu'],
    'mlpclassifier__solver': ['sgd', 'adam'],
    'mlpclassifier__alpha': [0.0001, 0.05],
    'mlpclassifier__learning_rate': ['constant','adaptive'],
    }
    search=RandomizedSearchCV(pl,param_grid,verbose=10,cv=shuffle_split,n_iter=30,random_state=1).fit(X_part,y[X_part.index])
    print(search.best_params_)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
    pl.fit(X_train, y_train)
    y_pred = pl.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("score :", round(score, 5))
    print(pd.concat([pd.Series(y_test,index=X_test.index),pd.Series(y_pred,index=X_test.index),X_test],axis=1).set_axis(["reference","prediction","message"],axis=1).head(20))