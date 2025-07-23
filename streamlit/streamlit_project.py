import streamlit as st
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from lightgbm import LGBMRegressor


base_dir = r'C:\FBr\Weiterbildung\Project'
ft_param_path = os.path.join(base_dir, 'Scripts', 'ft_files', '03_feat_parameters.ft')
#ft_param_path = r'C:\FBr\Weiterbildung\Project\Scripts\ft_files\03_feat_parameters.ft'
df_params = pd.read_feather (ft_param_path)

pkl_lgbm_long_freq_path = os.path.join(base_dir, 'Scripts', 'model_dumps', '4_model_lgbm_tuned_long_frequencies.pkl')
lgbm_long_freq = pickle.load(open(pkl_lgbm_long_freq_path, 'rb'))

st.title ('Ultrasonic welding')
st.sidebar.title ('Table of contents')
pages = ['Exploration', 'Data visualization', 'Modelling']
page = st.sidebar.radio('Go to', pages)

if page == pages[0]:
    st.write('### Presentation of data')
    st.dataframe(df_params.head(5))
    st.write(df_params.shape)
    st.dataframe(df_params.describe())

#    if st.checkbox("Show NA") :
#        st.dataframe(df_params.isna().sum())

#if page == pages[1]:
#    st.write('### Data visualization')
#    fig = plt.figure()
#    sns.countplot(x='Survived', data=df)
#    st.pyplot(fig)


if page == pages[2] : 
    st.write('### Modelling')
    
    
    
"""    
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    y = df['Survived']
    X_cat = df[['Pclass', 'Sex',  'Embarked']]
    X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = pd.concat([X_cat_scaled, X_num], axis = 1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'SVC':
            clf = SVC()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf

    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))
        
    choice = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)

    clf = prediction(option)
    display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))
"""

