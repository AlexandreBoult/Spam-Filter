import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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



def train_model(opti,df):
    global vectorize_pl
    global pl

    X = df['msg']
    y = df['cat']

    pl=make_pipeline(CountVectorizer(),TruncatedSVD(n_components=200,n_iter=5,random_state=1),MLPClassifier(random_state=1,max_iter=400,hidden_layer_sizes=(90, 60),alpha=0.0056))

    if opti: #hyperparameters optimsation
        shuffle_split = ShuffleSplit(n_splits=5,test_size=0.3)
        X_part=X.sample(frac=0.5)
        param_grid = {
        'mlpclassifier__hidden_layer_sizes': [(int(1.5*n),n) for n in range(50,110,10)],
        'mlpclassifier__alpha': [x/10000 for x in range(1,105,5)],
        }
        search=RandomizedSearchCV(pl,param_grid,verbose=10,cv=shuffle_split,n_iter=30,random_state=1).fit(X_part,y[X_part.index])
        return search.best_params_
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
        pl.fit(X_train, y_train)
        y_pred = pl.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        #print("score :", round(score, 5))
        #print(pd.concat([pd.Series(y_test,index=X_test.index),pd.Series(y_pred,index=X_test.index),X_test],axis=1).set_axis(["reference","prediction","message"],axis=1).head(20))
        return pl,score

def test_msg(model,msg) : return model.predict([msg])[0]

train_model(0,pd.read_table("SMSSpamCollection"))