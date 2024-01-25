from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline

from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import string
import numpy as np
from sklearn.model_selection import GridSearchCV


# function utilities
 
def punctuation_ratio(text_series):
    punctuations = string.punctuation
    return text_series.apply(lambda text: sum(1 for char in text if char in punctuations) / len(text) if len(text) > 0 else 0)

def count_digits(text_series):
    return text_series.apply(lambda text: sum(char.isdigit() for char in text))

def ratio_uppercase(text_series):
    return text_series.apply(lambda text: sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0)

def ratio_digits(text_series):
    return text_series.apply(lambda text: sum(char.isdigit() for char in text) / len(text) if len(text) > 0 else 0)

def len_message(text_series):
    return text_series.apply(len)


def create_column_df(X):

    df=pd.DataFrame(data=X,columns=["msg"])
    df['punctuation_ratio'] = punctuation_ratio(df['msg'])
    #df['digit_count'] = count_digits(df['msg'])
    df['uppercase_ratio'] = ratio_uppercase(df['msg'])
    df['ratio_digits'] =  ratio_digits(df['msg'])
    #df['msg_length'] = len_message(df['msg'])
    return df

text_vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)

ajout_column_prepoccessing = FunctionTransformer(create_column_df, validate=False)

#scaler = MinMaxScaler()
# vectorise uniquement la colonne 'msg'
vectorizer = ColumnTransformer(
    transformers=[
        ('text', text_vectorizer, 'msg'),
        #('scaler', scaler, 'digit_count')
    ]
)

# preprocessing
preprocessor=Pipeline(steps=[
    ('ajout', ajout_column_prepoccessing),
    ('vectorizer', vectorizer), 

])

# Pipeline complète
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('dim_reduction', TruncatedSVD(n_components=150, random_state=42)),
    ('classifier', svm.SVC())
])




# telechargement de la data
df = pd.read_csv("SMSSpamCollection", header=None, sep='\t', names=['cat', 'msg'])

X = df['msg']
y = df['cat'].apply(lambda x: 1 if x == 'spam' else 0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# paramétre gridsearch 

param_grid = {
    'classifier__C': [0.1, 1, 10, 15, 20, 25],
    'classifier__kernel': ['rbf'],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__class_weight': ['balanced'],  
    'classifier__break_ties' :[True, False]
}

grid_search = GridSearchCV(pipeline, param_grid, scoring='f1', cv = 5, n_jobs =-1, verbose = 1)



grid_search.fit(X_train, y_train)




y_pred = grid_search.predict(X_test)


best_model = grid_search.best_estimator_
print("Meilleur modèle:", best_model)
y_pred = best_model.predict(X_test)
#y_pred = pipeline.predict(X_test)
#pipeline.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Rapport de classification :")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
plt.show()