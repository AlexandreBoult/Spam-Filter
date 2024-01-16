import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split



df=pd.read_table("SMSSpamCollection")


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

pl=make_pipeline(vectorize_pl,PCA(n_components=200),MLPClassifier(random_state=1, max_iter=200))

X = df['msg']
y = df['cat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pl.fit(X_train, y_train)
y_pred = pl.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print("score :", round(score, 5))