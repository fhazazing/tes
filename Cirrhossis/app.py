import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model 
svm = pickle.load(open('SVC.pkl','rb'))

#load dataset
data = pd.read_csv('Cirhossis Dataset.csv')
data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Cirrhosis')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Cirrhosis Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['SVM','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Sirosis adalah kondisi ketika organ hati telah dipenuhi dengan jaringan parut dan tidak bisa berfungsi dengan normal. Jaringan parut ini terbentuk akibat penyakit liver yang berkepanjangan, misalnya karena infeksi virus hepatitis atau kecanduan alkohol.</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
x = data.drop('Stage',axis=1)
y = data['Stage']
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('x_train')
    st.write(x_train.head())
    st.write(x_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('x_test')
    st.write(x_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    id = st.sidebar.slider('ID', 0, 350, 50)
    n_days = st.sidebar.slider('N_Days', 0, 5000, 200)
    status = st.sidebar.selectbox('Status', ['C', 'D', 'CL'])
    drug = st.sidebar.selectbox('Drug', ['D-penicillamine', 'Placebo'])
    age = st.sidebar.slider('Age', 0, 100, 30)
    sex = st.sidebar.selectbox('Sex', ['F', 'M'])
    ascites = st.sidebar.selectbox('Ascites', ['Y', 'N'])
    hepatomegaly = st.sidebar.selectbox('Hepatomegaly', ['Y', 'N'])
    spiders = st.sidebar.selectbox('Spiders', ['Y', 'N'])
    edema = st.sidebar.selectbox('Edema', ['Y', 'N'])
    bilirubin = st.sidebar.slider('Bilirubin', 0.0, 10.0, 1.0)
    cholesterol = st.sidebar.slider('Cholesterol', 0.0, 500.0, 200.0)
    albumin = st.sidebar.slider('Albumin', 0.0, 5.0, 3.0)
    copper = st.sidebar.slider('Copper', 0.0, 500.0, 100.0)
    alk_phos = st.sidebar.slider('Alk_Phos', 0.0, 1000.0, 500.0)
    sgot = st.sidebar.slider('SGOT', 0.0, 500.0, 100.0)
    triglycerides = st.sidebar.slider('Tryglicerides', 0.0, 500.0, 200.0)
    platelets = st.sidebar.slider('Platelets', 0.0, 500.0, 250.0)
    prothrombin = st.sidebar.slider('Prothrombin', 0.0, 20.0, 10.0)

    user_report_data = {
        'ID': id,
        'N_Days': n_days,
        'Status': status,
        'Drug': drug,
        'Age': age,
        'Sex': sex,
        'Ascites': ascites,
        'Hepatomegaly': hepatomegaly,
        'Spiders': spiders,
        'Edema': edema,
        'Bilirubin': bilirubin,
        'Cholesterol': cholesterol,
        'Albumin': albumin,
        'Copper': copper,
        'Alk_Phos': alk_phos,
        'SGOT': sgot,
        'Tryglicerides': triglycerides,
        'Platelets': platelets,
        'Prothrombin': prothrombin
    }

    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)
user_features = user_data[x_train.columns]
user_result = svm.predict(user_features)
svc_score = accuracy_score(y_test, svm.predict(x_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena Cirrhosis'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(svc_score*100)+'%')