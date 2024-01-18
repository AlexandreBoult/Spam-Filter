import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV,ShuffleSplit,RandomizedSearchCV

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

#vectorize_pl = make_pipeline(CountVectorizer(),FunctionTransformer(lambda x: x.toarray(), accept_sparse=True))



def train_model(opti,ratio,df,par_dict):

    X = df['msg']
    y = df['cat']

    pl=make_pipeline(TfidfVectorizer(),TruncatedSVD(n_components=par_dict["n_components"],n_iter=par_dict["n_iter"],random_state=par_dict["random_state"]),MLPClassifier(random_state=par_dict["random_state"],max_iter=par_dict["max_iter"],hidden_layer_sizes=par_dict["hidden_layer_sizes"],alpha=par_dict["alpha"]))

    if opti: #hyperparameters optimsation
        shuffle_split = ShuffleSplit(n_splits=5,test_size=0.3)
        X_part=X.sample(frac=0.5)
        param_grid = {
        'mlpclassifier__hidden_layer_sizes': [(int(1.5*n),n) for n in range(50,110,10)],
        'mlpclassifier__alpha': [x/10000 for x in range(1,105,5)],
        }
        search=RandomizedSearchCV(pl,param_grid,verbose=10,cv=shuffle_split,n_iter=30,random_state=1).fit(X_part,y[X_part.index])
        return search.best_params_
    elif ratio != 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-ratio, random_state=5)
        pl.fit(X_train, y_train)
        y_pred = pl.predict(X_test)
        print(classification_report(y_test,y_pred))
        return pl,accuracy_score(y_test, y_pred)
    else :
        pl.fit(X, y)
        y_pred = pl.predict(X)
        return pl,accuracy_score(y, y_pred)

def test_msg(model,msg) : return model.predict([msg])[0]