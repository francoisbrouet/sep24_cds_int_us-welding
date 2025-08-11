import streamlit as st
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from lightgbm import LGBMRegressor


# Load dataframes
base_dir = r'C:\FBr\Weiterbildung\Project\GitHub\sep24_cds_int_us-welding'

ft_param_path = os.path.join(base_dir, 'ft_files', '03_feat_parameters.ft')
ft_modes_path = os.path.join(base_dir, 'ft_files', '03_coll_modes.ft')
ft_nodes_path = os.path.join(base_dir, 'ft_files', '03_coll_nodes.ft')
ft_defs_path = os.path.join(base_dir, 'ft_files', '03_coll_defs.ft')

df_params = pd.read_feather (ft_param_path)
df_modes = pd.read_feather (ft_modes_path)
df_nodes = pd.read_feather (ft_nodes_path)
df_defs = pd.read_feather (ft_defs_path)

# Select the relevant columns of df_params only
lst_param = df_params.columns.tolist()
lst_param_geom = lst_param[1:24]
lst_param_sel = [lst_param[0]] + lst_param_geom + lst_param[24:26] + lst_param[29:32] + lst_param[57:61] + lst_param[137:139]
df_params = df_params[lst_param_sel]

# Load models
pkl_lgbm_long_freq_path = os.path.join(base_dir, 'model_dumps', '4_model_lgbm_tuned_long_frequencies.pkl')
lgbm_long_freq = pickle.load(open(pkl_lgbm_long_freq_path, 'rb'))


st.title ('Ultrasonic welding')
st.sidebar.title ('Table of contents')
pages = ['Exploration and visualization', 'Modelling of the frequencies', 'Modelling of the displacements']
page = st.sidebar.radio('Go to', pages)

if page == pages[0]:
    
    st.write('### Exploration and visualization of the data')
    
    st.write ('DataFrame of parameters')
    st.dataframe(df_params.head(100))
    st.write('Size:', df_params.shape)

    st.write ('DataFrame of modes')
    st.dataframe(df_modes.head(152))
    st.write('Size:', df_modes.shape)
    
    st.write ('DataFrame of nodes')
    st.dataframe(df_nodes.head(495))
    st.write('Size:', df_nodes.shape)
    
    st.write ('DataFrame of displacements')
    st.dataframe(df_defs.head(495))
    st.write('Size:', df_defs.shape)

# Example of the design point 1001
# modes
# long. mode
# displacements



    st.write('### Data visualization')
#    fig = plt.figure()
#    sns.countplot(x='Survived', data=df)
#    st.pyplot(fig)

    st.write('Distribution of the geometric variables')
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.boxplot(df_params[lst_param_geom], labels=lst_param_geom)
    plt.xticks(rotation=90)
    plt.grid(True)
    st.pyplot(fig)

#    st.write ('Correlation heatmap')
#    cm = df_params[lst_param].corr()
#    fig, ax = plt.subplots(figsize=(8, 7))
#    sns.heatmap(cm, annot=False, ax=ax, cmap='coolwarm')
#    st.pyplot(fig)



if page == pages[1] : 
    st.write('### Modelling the frequency of the longitudinal mode')
    st.write ('LightGBM model:')
    choice = ['base model', 'base model + tuned model', 'base model + tuned model + model with filtered data']
    option = st.selectbox('Choice of the model', choice)    
    
# Long frequencies:

# (selectbox)
# LGBM base model:                                  y_true y_pred       residuals
# LGBM base + tuned model                           y_true  y_pred      residuals
# LGBM base + tuned model + filtered model           y_true  y_pred      residuals




# Frequency residuals = f(MAC value)

# Points Best & worst predictions

    display = st.radio('Which prediction do we want to show ?', ('Best prediction', 'Worst prediction'))
    if display == 'Best prediction':
        st.write('Best prediction')
    if display == 'Worst prediction':
        st.write('Worst prediction')

# (radio) best prediction, worst prediction
# Frequency, MAC values, displacements, frequencies neighbor modes

if page == pages[2] : 
    st.write('### Modelling the displacement uniformity of the longitudinal mode')
    st.write ('Neural network model:')


# Examples

# (checkbox) yes/no
#    if st.checkbox("Show NA") :
#        st.dataframe(df_params.isna().sum())
    

# df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# y = df['Survived']
# X_cat = df[['Pclass', 'Sex',  'Embarked']]
# X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

# for col in X_cat.columns:
    # X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
# for col in X_num.columns:
    # X_num[col] = X_num[col].fillna(X_num[col].median())
# X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
# X = pd.concat([X_cat_scaled, X_num], axis = 1)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
# X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix

# def prediction(classifier):
    # if classifier == 'Random Forest':
        # clf = RandomForestClassifier()
    # elif classifier == 'SVC':
        # clf = SVC()
    # elif classifier == 'Logistic Regression':
        # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    # return clf

# def scores(clf, choice):
    # if choice == 'Accuracy':
        # return clf.score(X_test, y_test)
    # elif choice == 'Confusion matrix':
        # return confusion_matrix(y_test, clf.predict(X_test))
    
# choice = ['Random Forest', 'SVC', 'Logistic Regression']
# option = st.selectbox('Choice of the model', choice)
# st.write('The chosen model is :', option)

# clf = prediction(option)
# display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
# if display == 'Accuracy':
    # st.write(scores(clf, display))
# elif display == 'Confusion matrix':
    # st.dataframe(scores(clf, display))


