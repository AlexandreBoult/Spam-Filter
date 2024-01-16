import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
df=pd.read_table("SMSSpamCollection")
print(df)

vectorizer = CountVectorizer() 
wm = vectorizer.fit_transform(df['msg'].tolist())
tokens=vectorizer.get_feature_names_out()
df_vect=pd.DataFrame(data=wm.toarray(),columns=tokens)

print(df_vect)
from sklearn.decomposition import PCA
pca=PCA(n_components=200)
reduced_df=pd.DataFrame(pca.fit_transform(df_vect))
print(reduced_df)

from sklearn.model_selection import train_test_split
X = reduced_df
"""
transfo_target = Pipeline(steps=[
    ('Label_enco', LabelEncoder())
])
"""

y = df['cat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
"""
model = Pipeline(steps=[
                        ('model',KNeighborsClassifier(3,p=1))])
"""
#model = GaussianNB()
model = MLPClassifier(random_state=1, max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
knn_score = accuracy_score(y_test, y_pred)
print("score :", round(knn_score, 5))