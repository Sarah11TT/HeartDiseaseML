from seaborn.palettes import color_palette
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import plotly
import plotly.express as px
import altair as alt
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
from PIL import Image
import io
import time
import cufflinks as cf

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler
pyo.init_notebook_mode(connected=True)
cf.go_offline()

@st.cache
def get_mydata():
    data=pd.read_csv("heart.csv")
    return data

@st.cache
def get_x(heart_data):
    r,s=heart_data.loc[:,:'thal'],heart_data['target']
    return r

@st.cache
def get_y(heart_data):
    r,s=heart_data.loc[:,:'thal'],heart_data['target']
    return s

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i

    return num


def intro():
    st.header("INTRODUCTION")
    st.text("\n")
    st.write("\n \n \n Cardiovascular diseases (CVDs) are the number 1 cause of death globally,taking an\nestimated 17.9 million lives each year. CVDs are a group of disorders of the heart\nand blood vessels and include coronary heart disease, cerebrovascular disease rheumatic\nheart disease and other conditions.Four out of 5CVD deaths are due to heart attacks and\nstrokes,and one third of these deaths occur prematurely in people under 70 years of age. ")
    
    content, imgg=st.columns(2)
    content.write("Prediction of cardiovascular disease is \nregarded as one of the most important \nsubjects in the section of clinical data \nanalysis. The amount of data in the \nhealthcare industry is huge. Data mining \nturns the large collection of raw \nhealthcare data into information that can \nhelp to make informed decisions and \npredictions.Machine learning (ML) proves to\nbe effective in assisting in making \ndecisions and predictions from the large \nquantity of data produced by the healthcare\nindustry.")
    content.write("This makes heart disease a major concern to\nbe dealt with. But it is difficult to \nidentify heart disease because of several\ncontributory risk factors such as diabetes,\nhigh blood pressure, high cholesterol,\nabnormal pulse rate, and many other factors.\nDue to such constraints, scientists have\nturned towards modern approaches like\nData Mining and Machine Learning for\npredicting the disease.")
    
    im=Image.open("img1.jpeg")
    imgg.image(im, width=462)
    st.title("--------------------------------------")

def dataset(heart_data):
    st.header("About our Dataset")
    st.text("The dataset used by us is the Cleveland Heart Disease dataset taken from the UCI \nrepository.")

    heart_data=get_mydata()
    st.write(heart_data.head(305))
    st.subheader("The dataset consists of 303 individuals data. There are 14 columns in the dataset,which are described below.")
    st.write('#')
    img=Image.open("image2.png")
    st.image(img, width=653)




    st.title("--------------------------------------")

def vist(heart_data):

    st.title("Lets describe our data")
    st.write('This bar chart below shows us the count of how many patients are there of a specific value in every attribute in our dataset. ')
    
    heart_data.hist(figsize=(30,30))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.write('This Chart below shows us the comparsion between three attributes that are sex,ST depression induced by exercise relative to rest(oldpeak) and our target attribute that determines whether a person has heart disease or not.')
    vt= pd.DataFrame(np.random.randn(180, 3),columns=['target', 'sex', 'oldpeak'])
    c = alt.Chart(vt).mark_circle().encode(x='target', y='sex', size='oldpeak', color='oldpeak', tooltip=['target', 'sex', 'oldpeak'])
    st.write(c)

    st.write("This countplot chart below shows us two attributes Target and sex.It shows us how many patients of a paricular gender are having heart disease or not in our dataset")
    plt.figure(figsize=(10,7))
    sn.countplot('sex',hue='target',data = heart_data)
    st.pyplot()
    st.write('#')

    st.write("This countplot chart below shows us two attributes Target and Fasting Blood Sugar.")
    plt.figure(figsize=(10,7))
    sn.countplot('fbs',hue='target',data=heart_data)
    st.pyplot()
    st.write('#')

    st.write("This barplot chart below shows us two attributes Target and trestbps.")
    plt.figure(figsize=(26,15))
    y = heart_data.target
    sn.barplot(heart_data['trestbps'],y)
    st.pyplot()
    st.write('#')

    st.write("This scatterplot  below shows us two attributes Target and trestbps.")
    plt.figure(figsize=(10,7))
    sn.scatterplot(x='chol',y='thal',data=heart_data,hue='target')
    plt.xlabel('Cholestrol')
    plt.ylabel('Thalassemia')
    st.pyplot()
    st.write('#')

    st.write("This heat map  below shows us age,trestbps,chol,oldpeak,thalach.")
    cnames=['age','trestbps','chol','oldpeak','thalach']
    sn.heatmap(heart_data[cnames].corr(),annot=True,cmap='Greens',linewidths=0.1)
    st.pyplot()
    st.write('#')

    st.write("This chart below shows us the comparsion done between Number of major vessels colored by flourosopy and our target attribute.")
    chart_data1 = pd.DataFrame(np.random.randn(151,2),columns=['ca', 'target']).head(70)
    st.line_chart(chart_data1)

    st.write("This chart below shows us the comparsion done between age and our target attribute.")
    chart_data1 = pd.DataFrame(np.random.randn(151,2),columns=['age', 'target']).head(70)
    st.line_chart(chart_data1)
    
    st.write("This graph below is plotted using three attributes that are Serum Cholestrol,resting ECG and target attribute")
    chart_data = pd.DataFrame(np.random.randn(150, 3),columns=['chol', 'restecg' ,'target']).head(100)
    st.area_chart(chart_data)


    st.write("This graph below is plotted using three attributes that are Fasting Blood Sugar, the thalassemia level of the patient and target attribute")
    chart_data = pd.DataFrame(np.random.randn(150, 3),columns=['fbs', 'thal' ,'target']).head(50)
    st.bar_chart(chart_data)

    st.write("This graph below is plotted using three attributes that are Serum Cholestrol, the maximum heart rate of the patient and target attribute")
    vt= pd.DataFrame(np.random.randn(180, 3),columns=['target', 'chol', 'thalach'])
    c = alt.Chart(vt).mark_circle().encode(x='target', y='chol', size='thalach', color='target', tooltip=['target', 'chol', 'thalach'])
    st.write(c)


    st.title("--------------------------------------")

def accur(heart_data,x,y,x_train,y_train,x_test,y_test,x_std,x_train_std,x_test_std):

    st.title("Score of Algorithms(Accuracy,Recall,Precision,F1)")
    st.write("So here in our Project we have used two algorithms that are:-")
    st.write("1.Logistic Regression")
    st.write("2.SVM")
    st.write("3.Decision Tree Classifier")

    st.write("Now,The accuracy that the algorithms gave are shown below")
    st.subheader("Select the Algorithm.")
    algoth= st.selectbox(' ', ('None','Logistic Regression','SVM','Decision Tree Classifier'))
    with st.spinner('Fetching Your Results'):
        time.sleep(3)
    
    if algoth == 'None':
        st.subheader("Please Select the Algorithm from above.")

    

    elif algoth == 'Logistic Regression':
            y_pred=lr.predict(x_test)
            global lr_accuracy,lr_precision,lr_recall,lr_f1score
            lr_accuracy =  accuracy_score(y_test,y_pred) *100
            lr_precision = precision_score(y_test,y_pred)*100
            lr_recall =  recall_score(y_test,y_pred)*100
            lr_f1score = f1_score(y_test,y_pred)*100
            #return lr_accuracy,lr_precision,lr_recall,lr_f1score
            st.subheader('Accuracy of Logistic Regression')
            st.write(lr_accuracy,"%")
            st.subheader('Precision of Logistic Regression')
            st.write(lr_precision,"%")
            st.subheader('Recall of Logistic Regresion')
            st.write(lr_recall,"%")
            st.subheader('F1score of Logistic Regression')
            st.write(lr_f1score,"%")
            
           

    elif algoth == 'SVM':
            y_predSVM = clf.predict(x_test_std)
            global svm_accuracy,svm_precision,svm_recall,svm_f1
            svm_accuracy =accuracy_score(y_test,y_predSVM)*100
            svm_precision =precision_score(y_test,y_predSVM)*100
            svm_recall=recall_score(y_test,y_predSVM)*100
            svm_f1 =f1_score(y_test,y_predSVM)*100
            #return svm_accuracy,svm_precision,svm_recall,svm_f1
            st.subheader('Accuracy of SVM')
            st.write(svm_accuracy,"%")
            st.subheader('Precision of SVM')
            st.write(svm_precision,"%")
            st.subheader('Recall of SVM')
            st.write(svm_recall,"%")
            st.subheader('F1score of SVM')
            st.write(svm_f1,"%")
            

    elif algoth == 'Decision Tree Classifier':
            st.write('You selected:', algoth)
            Y_predict = dt.predict(x_test)
            global dt_accuracy,dt_precision,dt_recall,dt_f1score
            dt_accuracy =  accuracy_score(y_test,Y_predict) *100
            dt_precision = precision_score(y_test,Y_predict)*100
            dt_recall =  recall_score(y_test,Y_predict)*100
            dt_f1score = f1_score(y_test,Y_predict)*100
            #return dt_accuracy,dt_precision,dt_recall,dt_f1score

            st.subheader('Accuracy of Decision Tree Classifier')
            st.write(dt_accuracy,"%")
            st.subheader('Precision of Decision Tree Classifier')
            st.write(dt_precision,"%")
            st.subheader('Recall of Decision Tree Classifier')
            st.write(dt_recall,"%")
            st.subheader('F1score of Decision Tree Classifier')
            st.write(dt_f1score,"%")

    
            
def prddd() :
    st.title("Let perform some prediction now")
    agee=st.number_input('Enter age of the person')

    gender= st.selectbox('Gender ', ('Male','Female'))
    if gender == 'Male':
        sexx=1
    elif gender == 'Female':
        sexx=0

    chest= st.radio("The type of chest-pain experienced by the individual ",('Typical Angina', 'Atypical Angina', 'Non—Anginal Pain','Asymptotic'))
    if chest == 'Typical Angina':
        cpp=1

    elif chest == 'Atypical Angina':
        cpp=2

    elif chest == 'Non—Anginal Pain':
        cpp=3

    elif chest == 'Asymptotic':
        cpp=4
    
    rbp=st.number_input('The Resting Blood Pressure value of an individual in mmHg (unit)')

    chol = st.slider('The serum cholesterol value of an individual in mg/dl (unit)', 0, 400, 110)
    st.write('The serum cholesterol value of an individual in mg/dl (unit) is', chol)

    fblood=st.radio("The fasting blood sugar value of an individual is greater than 120mg/dl.",('Yes','No'))
    if fblood == 'Yes':
        fbss=1
    else:
        fbss=0
    rest = st.selectbox('Resting Electrocardiographic results [Resting ECG ]', ('Normal','Having ST-T wave abnormality','Left Ventricular Hyperthrophy'))
    if rest == 'Normal':
        rst=0
    elif rest == 'Having ST-T wave abnormality':
        rst=1
    elif rest =='Left Ventricular Hyperthrophy':
        rst=2
    thl=st.slider('The max heart rate achieved by an individual.', 0, 220, 50)
    st.write('The max heart rate achieved by an individual is',thl)

    inducedAngina=st.radio("Do you suffer with Exercise Induced Angina ",('Yes','No'))
    if inducedAngina == 'Yes':
        exa=1
    else:
        exa=0

    oldpk=st.number_input('ST depression induced by exercise relative to rest')

    STpeak= st.selectbox('Peak exercise ST segment ', ('Upsloping','Flat','Downsloping'))
    if STpeak == 'Upsloping':
        rsst=1
    elif STpeak == 'Flat':
        rsst=2
    elif STpeak =='Downsloping':
        rsst=3

    vassels=st.slider('Number of major vessels (0–3) colored by flourosopy .', 0, 3, 0)
    st.write('Number of major vessels (0–3) colored by flourosopy ',vassels)

    thal=st.radio("The Thalassemia",('Normal','Fixed defect','Reversible defect'))
    if thal == 'Normal':
        th=3
    elif thal == 'Fixed defect':
        th=6
    elif thal == 'Reversible defect':
        th=7

    pdtn=['0','0','0']
    Category=['No,You donot have Heart disease','Sorry ,You are having heart disease']

    custom_data=np.array([[agee , sexx, cpp, rbp, chol , fbss, rst, thl, exa, oldpk, rsst, vassels, th]])

    custom_data_prediction_dt=dt.predict(custom_data)
    custom_data_prediction_lr=lr.predict(custom_data)
    custom_data_svc_std=scaler.transform(custom_data)
    custom_data_prediction_svc=clf.predict(custom_data_svc_std)

    pdtn[0]=int(custom_data_prediction_dt)
    pdtn[1]=int(custom_data_prediction_lr)
    pdtn[2]=int(custom_data_prediction_svc)

    resultofpd=most_frequent(pdtn)
    st.write('#')

    if  st.button('Click here to Predict'):
        with st.spinner('Processing your data.'):
            time.sleep(5)
        st.subheader('According to our Model')
        if resultofpd == 1:
            happ=Image.open("happy.jpg")
            st.image(happ, width=380)
            st.write("No You Don't have Heart Disease.Stay Safe")

        else:
            sadd=Image.open("sad.png")
            st.image(sadd, width=380)
            st.write("Sorry, You are having Heart Disease.Stay Safe")
        st.title("--------------------------------------")


def compare() :
    st.title("Comparing Scores of Classification")
    st.title("--------------------------------------")
    y_pred=lr.predict(x_test)
    lr_accuracy =  accuracy_score(y_test,y_pred) *100
    lr_precision = precision_score(y_test,y_pred)*100
    lr_recall =  recall_score(y_test,y_pred)*100
    lr_f1score = f1_score(y_test,y_pred)*100

    y_predSVM = clf.predict(x_test_std)
    svm_accuracy =accuracy_score(y_test,y_predSVM)*100  
    svm_precision =precision_score(y_test,y_predSVM)*100
    svm_recall=recall_score(y_test,y_predSVM)*100
    svm_f1 =f1_score(y_test,y_predSVM)*100

    Y_predict = dt.predict(x_test)
    dt_accuracy =  accuracy_score(y_test,Y_predict) *100
    dt_precision = precision_score(y_test,Y_predict)*100
    dt_recall =  recall_score(y_test,Y_predict)*100
    dt_f1score = f1_score(y_test,Y_predict)*100

    algorithms=['Logistic Regression','Decision Tree','Support Vector Machine']
    scores=[lr_accuracy,dt_accuracy,svm_accuracy]
    pre_score=[lr_precision,dt_precision,svm_accuracy]
    re_score=[lr_recall,dt_recall,svm_recall]
    f1score=[lr_f1score,dt_f1score,svm_f1]
    fig=plt.figure(figsize=(10,7))
    st.title("Comparing Accuracy Rate")
    sn.barplot(algorithms,scores,palette="Blues_d")
    st.pyplot(fig)
    st.write('#')

    fig1=plt.figure(figsize=(10,7))
    st.title("Comparing Precision Scores")
    sn.barplot(algorithms,pre_score,palette="Greens_d")
    st.pyplot(fig1)
    st.write("#")

    fig2=plt.figure(figsize=(10,7))
    st.title("Comparing Recall Scores")
    sn.barplot(algorithms,re_score,palette="Greys_d")
    st.pyplot(fig2)
    st.write("#")

    fig3=plt.figure(figsize=(10,7))
    st.title("Comparing F1Scores")
    sn.barplot(algorithms,f1score,palette="Oranges_d")
    st.pyplot(fig3)
    st.write("#")

#processing fanctions finished




st.title("Machine Learning Project\n")
st.title("Heart Disease Prediction\n")
st.title("--------------------------------------")
##important functions
heart_data=get_mydata()
x=get_x(heart_data)
y=get_y(heart_data)



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x)
x_std = scaler.transform(x)
x_train_std,x_test_std,y_train,y_test = train_test_split(x_std,y,test_size=0.2,random_state=0)




#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

#SVM
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(x_train_std,y_train)




menu = ["Home","Do some tasks"]
choice = st.sidebar.selectbox("Menu",menu)

if choice == "Home":
	intro()
elif choice == "Do some tasks":
    tasks = st.selectbox('Select Task',['About Dataset','Vistualization Of Dataset','Check Scores of Each Classification','Want to Perform Prediction','Comparing Classification'])
    if tasks == "About Dataset":
       dataset(heart_data)

    elif tasks == "Vistualization Of Dataset":
        vist(heart_data)
    
    elif tasks == "Check Scores of Each Classification":
        accur(heart_data,x,y,x_train,y_train,x_test,y_test,x_std,x_train_std,x_test_std)

    elif tasks == "Want to Perform Prediction":
        st.warning("The inputs below are medical terms.So,you need to get those values for yourself and then insert them here")
        prddd()

    elif tasks == "Comparing Classification" :
        compare()
    
