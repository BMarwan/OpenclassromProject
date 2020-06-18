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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
import streamlit as st

def main():
    st.header("Prêt à dépenser")
    df = load_data()
    user_input = st.text_input("Entrer le numéro du client")

    if st.button('Rechercher'):
        with st.spinner("Training ongoing"):
            st.write(df[df['SK_ID_CURR']==int(user_input)])
    

@st.cache
def load_data():
    df = pd.read_csv('test_bureau_corrs_removed.csv')

    return df


if __name__ == "__main__":
    main()