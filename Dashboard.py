
# st.title('Prêt à dépenser')
# st.write('100001, 100005, 100013, 100028')




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
            client_id = int(user_input)
            #explainer = shap.TreeExplainer(clf)
            #shap_values = explainer.shap_values(test_features.iloc[0,:])

        with st.spinner("Calcul de l'interprétation"):
            exp_keys = explain_score(test_features, df, client_id, clf)

        with st.spinner("Calcul des similarités"):
            df_similaire = client_similarities(test_features, df, client_id, exp_keys)
            st.write('Clients similaires :')            
            st.write(df_similaire)

           
            



def load_data():
    df = pd.read_csv('test_bureau_corrs_removed.csv')
    return df

def load_models():
    clf_p = load('lgb_pos_weights.joblib')
    clf_n = load('lgb_neg_weights.joblib')
    return clf_p, clf_n

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


def explain_score(df_test, df, id, clf):
     lime1 = LimeTabularExplainer(df_test,
                                 feature_names=df_test.columns,
                                 discretize_continuous=False)
     test = df.index[df['SK_ID_CURR']==id]
     exp = lime1.explain_instance(df_test.iloc[test].to_numpy().ravel(), 
                                  clf.predict_proba, 
                                  num_features=10,
                                  #class_names=['client solvable, client non solvable']
                                  )

     exp_list = exp.as_list()       
     exp_keys = []
     exp_values = []
     exp_positives = []
     for i in range(len(exp_list)):
         exp_keys.append(exp_list[i][0])
         exp_values.append(exp_list[i][1])
         exp_positives.append(exp_values[i] > 0)
    
     x = exp_keys
     y = exp_values

     fig, ax = plt.subplots(figsize=(15,8))    
     width = 0.75 # the width of the bars 
     ind = np.arange(len(y))  # the x locations for the groups
     c = ax.barh(x, y, width, color=pd.Series(exp_positives).map({True: 'r', False: 'g'}))
     ax.set_yticks(ind+width/2)
     ax.set_yticklabels(x, minor=False)
     plt.gca().invert_yaxis()

     for index, value in enumerate(y):
         plt.text(value, index, str(np.round(value, decimals=3)))
     
     plt.title('Explication des résulats')
     plt.xlabel('Poids en pourcentage')   
     plt.tight_layout()
     st.pyplot()
     #st.write(exp.as_pyplot_figure(), plt.tight_layout())
     #st.write(exp.show_in_notebook(show_all=False))
     return exp_keys

def client_similarities(df_test, df, id, exp_keys):
    
    #Train the algo
    neigh = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(df_test)
    
    index = df[df["SK_ID_CURR"] == id].index.values

    data_client = df_test.iloc[index[0]]
    data_client = data_client.values.reshape(1, -1)

    #Find the 5 nearest neighborhoods
    indices = neigh.kneighbors(data_client, return_distance=False)
    
    clients_similaires = pd.DataFrame()
    #Extract informations of the 5 nearest
    for i in indices:
        clients_similaires = clients_similaires.append(df.iloc[i])

    #clients_similaires = clients_similaires[exp_keys]
    
    return clients_similaires


if __name__ == "__main__":
    main()