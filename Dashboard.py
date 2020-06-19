# import streamlit as st

# import numpy as np
# import pandas as pd

# st.title('Prêt à dépenser')
# st.write('100001, 100005, 100013, 100028')

# test_corrs_removed = pd.read_csv('test_bureau_corrs_removed.csv')

# user_input = st.text_input("Entrer le numéro du client")

# # st.dataframe(test_corrs_removed[test_corrs_removed['SK_ID_CURR']==user_input])

# #st.write(test_corrs_removed.head())

# if st.button('Entrer'):
#     st.write(test_corrs_removed[test_corrs_removed['SK_ID_CURR']==user_input])
#     st.text(user_input)



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
from sklearn.neighbors import NearestNeighbors

import shap

import streamlit as st
from joblib import dump, load
import lightgbm as lgb


def main():
    st.header("Prêt à dépenser")
    st.write('Client solvable : 100001')
    st.write('Client non solvable : 100005')

    #Load data
    df = load_data()
    #Preprocess with dummies and drop IDs
    test_features = preprocess_data(df)
    
    clf = model_choice()

    #Create an input
    user_input = st.text_input("Entrer le numéro du client")


    if st.button('Rechercher'):
        with st.spinner("Calcul en cours"):
            st.write(df[df['SK_ID_CURR']==int(user_input)])
            defaut_precentage = clf.predict_proba(test_features.iloc[df.index[df['SK_ID_CURR']==int(user_input)]])[0][1]
            st.write('Probabilité de défaut de paiment : ', np.around(defaut_precentage*100, decimals=2),'%')
            test = int(user_input)
            #explainer = shap.TreeExplainer(clf)
            #shap_values = explainer.shap_values(test_features.iloc[0,:])


            explain_score(test_features, df, test, clf)

           
            



#@st.cache
def load_data():
    df = pd.read_csv('test_bureau_corrs_removed.csv')
    return df

#@st.cache
def load_models():
    clf_p = load('lgb_pos_weights.joblib')
    clf_n = load('lgb_neg_weights.joblib')
    return clf_p, clf_n

#@st.cache
def preprocess_data(df):
    test_features = df
    test_features.fillna(test_features.median(), inplace=True)
    test_ids = test_features['SK_ID_CURR']
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    test_features = pd.get_dummies(test_features)
    return test_features


def model_choice():
    #Load Models
    clf_p, clf_n = load_models()
    #Choix du model
    choose_model = st.radio(
                "Selection du modèle :",
                ('Privilégier clients solvables', 'Détecter clients non solvables'))

    if choose_model == 'Privilégier clients solvables':
        clf = clf_n
    elif choose_model == 'Détecter clients non solvables':
        clf = clf_p
    return clf


#@st.cache
def explain_score(df_test, df, id, clf):
     lime1 = LimeTabularExplainer(df_test,
                                 feature_names=df_test.columns,
                                 discretize_continuous=False)
     test = df.index[df['SK_ID_CURR']==id]
     exp = lime1.explain_instance(df_test.iloc[test].to_numpy().ravel(), clf.predict_proba, num_features=10)
            
     st.write(exp.as_pyplot_figure(), plt.tight_layout())
     


if __name__ == "__main__":
    main()