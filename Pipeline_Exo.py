# data
import pandas as pd

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler

# Pipeline and model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

# Score of models
from sklearn.metrics import accuracy_score

# Load data
url = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/spotify.zip"
df = pd.read_csv(url)

# Selection of a small fraction of data to speed up the ML process (training time ~ 1 hour)
df_music = df.sample(frac=0.2)

y = df_music['genre']

X = df_music.drop(columns='genre')

column_cat = X.select_dtypes(include=['object']).columns.drop(['artist_name', 'track_name', 'track_id'])

transfo_cat = Pipeline(steps=[
    ('imputation', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))
])

column_num = X.select_dtypes(exclude=['object']).columns

transfo_num = Pipeline(steps=[
    ('imputation', SimpleImputer(strategy='median')),
    ('scaling', MinMaxScaler())
])

transfo_text = Pipeline(steps=[
    ('bow', CountVectorizer())
])

preparation = ColumnTransformer(
    transformers=[
        ('data_cat', transfo_cat , column_cat),
        ('data_num', transfo_num , column_num),
        ('data_track', transfo_text , 'track_name')
    ])

#model = LogisticRegression(penalty='elasticnet', l1_ratio=0.2, multi_class='ovr', solver='saga') #GaussianNB() #GradientBoostingClassifier() #CategoricalNB() #SVC() #CategoricalNB()#
model = LogisticRegression(penalty='l1', max_iter=5000, solver='saga')


pipe_model = Pipeline(steps=[('preparation', preparation),('model',model)])

from sklearn import set_config
set_config(display='diagram')
pipe_model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)
pipe_model.fit(X_train, y_train)
y_pred = pipe_model.predict(X_test)

# Score of models
score = accuracy_score(y_test, y_pred)
print("score :", round(score, 5))
