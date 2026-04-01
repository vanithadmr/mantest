import pandas as pd 
import numpy as np 
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
import pickle as pkl 
import re 



nltk.download("stopwords")
st.set_page_config(
    page_title = "Comment toxicity Detection",
    layout = 'wide'
) 

with open("C:/guvi/project_5_testing/model_lstm.pkl", 'rb' ) as f:
    model_lstm = pkl.load(f)
#st.write(model_lstm) 

with open("C:/guvi/project_5_testing/tokenizer.pkl", 'rb') as f:
    tokenizer = pkl.load(f)
#st.write(tokenizer)

c1,c2 = st.columns([1,1])
with c2:
    st.image("C:/guvi/project_5/comment.png") 

with c1:
    st.markdown(
    """
    <h1 style='text-align: center;'>
        Comment toxicity Detection 
    </h1> 
   
    """,
    unsafe_allow_html=True) 

st.markdown("""
        <style>
        [data-testid=stSidebar] {
        background-color: #B8E7AA


        } 
         div.row-widget.stRadio > div{
        flex-direction:column;
        }
        /* Target the text labels next to the radio buttons */
        div.stRadio label p {
        font-size: 20px !important;
        font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
max_len = 200
with st.sidebar: 
    
    options = ['Model_Deployment', 'Data insights','Data']
    option = st.radio("Pages :", options)  

if option == 'Model_Deployment':

    
    def comments_cleaning(comment):
        comments = comment.lower()
        comments = re.sub(r'[^a-zA-z]', " ", comment)
        words = comments.split()
        stop_words = stopwords.words('english')
        comments = [word for word in words if word not in stop_words]
        return " ".join(comments)
    
    st.subheader('User Inputs') 
    options = ['Single', 'Bulky']
    options = st.selectbox("", options)  

    if options == "Single" :
        user_text = st.text_area("", height=80) 
        #st.write(type(user_text))
    
        user_text = comments_cleaning(user_text)
        user_text = tokenizer.texts_to_sequences([user_text])
        user_text = pad_sequences(user_text, maxlen = max_len, padding = 'post', truncating = 'post')
        predictions_ = model_lstm.predict(user_text)
        #predictions_ 
        if st.button('Predict'):  
            st.info("Sucessfully detected!!!!")

            st.markdown("""
            <style>
            div.stButton > button:first-child {
            background-color: #B8E7AA; /* Green background */
            color: black;              /* White text */
            font-size: 20px;
            height: 3em;
            width: 10em;
            border-radius: 10px;       /* Rounded corners */
            }
            div.stButton > button:first-child:hover {
            background-color: #009900; /* Darker green on hover */
            }
            </style>
            """, unsafe_allow_html=True) 


            st.title("6 Target Labels Detection")
            labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            predictions_ = np.where(predictions_ > 0.5, 1, 0)
            df = pd.DataFrame(predictions_, columns=labels)
            
            
            
            if predictions_[0][0] == 1:
                st.button('Toxic') 
            if predictions_[0][1] == 1:
                st.button('Severe_Toxic')
            if predictions_[0][2] == 1:
                st.button('Obscene')
            if predictions_[0][3] == 1:
                st.button('Threat') 
            if predictions_[0][4] == 1:
                st.button('Insult') 
            if predictions_[0][5] == 1:
                st.button('Identity_hate')
            st.write("\n\n",df) 
            
    elif options == 'Bulky':
        user_text = st.text_area("", height=80) 
        if st.button('click me!'):  
            X_test = pd.read_csv(user_text) 
            X_test['comment_text'] = X_test['comment_text'].apply(comments_cleaning)
            X_test['comment_text'] = X_test['comment_text'].str.lower()
            sequences_test = tokenizer.texts_to_sequences(X_test['comment_text'])
            padded_seq_test = pad_sequences(sequences_test, maxlen = max_len, padding = 'post', truncating = 'post')
            y_pred = model_lstm.predict(padded_seq_test)
            
            labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            y_pred = (y_pred > 0.5).astype(int) 
            df = pd.DataFrame(y_pred, columns=labels)
            st.write(df)

    
    
        
    
elif option == 'Data insights':
    st.image("C:/guvi/project_5_testing/target_distribution.png")
    st.image("C:/guvi/project_5_testing/heatmap.png")
    st.image("C:/guvi/project_5_testing/roc_curvelstm.png") 
    st.image("C:/guvi/project_5_testing/toxic.png") 
    st.image("C:/guvi/project_5_testing/severe_toxic.png") 
    st.image("C:/guvi/project_5_testing/obscene.png") 
    st.image("C:/guvi/project_5_testing/threat.png") 
    st.image("C:/guvi/project_5_testing/insult.png") 
    st.image("C:/guvi/project_5_testing/identity_hate.png") 




elif option == 'Data':

    st.markdown(
        '<h1 <p style="font-size:30px; color:black;"> Before Cleaning Dataframe </h2></p>',
        unsafe_allow_html=True
        )
    st.write('Train data\n')
    train = pd.read_csv("C:/guvi/project_5_testing/train.csv") 
    test = pd.read_csv("C:/guvi/project_5_testing/test.csv") 
    st.dataframe(train) 
    st.write('Test data\n')
    st.dataframe(test) 

    st.markdown(
        '<h1 <p style="font-size:30px; color:black;"> After Cleaning Dataframe </h2></p>',
        unsafe_allow_html=True
        ) 
    st.write('Train data\n')
    train1 = pd.read_csv("C:/guvi/project_5_testing/cleaned_train.csv") 
    test1 = pd.read_csv("C:/guvi/project_5_testing/cleaned_test.csv") 
    st.dataframe(train1)
    st.write('Test data\n')
    st.dataframe(test1) 
