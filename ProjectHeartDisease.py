#!/usr/bin/env python
# coding: utf-8


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import plotly
import plotly.express as px

import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
import cufflinks as cf
pyo.init_notebook_mode(connected=True)
cf.go_offline()


# In[2]:


#Data Collection And Processing
#load data to dataframe
heart_data = pd.read_csv('/Users/thetmyatnoe/Downloads/heart.csv')


# In[3]:


heart_data


# In[4]:


heart_data.info()


# In[5]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]

for i in range(len(info)):
    print(heart_data.columns[i]+":\t\t\t"+info[i])


# In[6]:


heart_data.isnull().sum()


# In[7]:


heart_data.describe()


# In[8]:


heart_data['target']


# In[9]:


heart_data.groupby('target').size()


# In[10]:


#Data Vitsualization
sn.set()


# In[11]:


heart_data.hist(figsize=(15,15))
plt.show()


# In[12]:


sn.countplot('target',data = heart_data)
heart_data.target.value_counts()


# In[13]:


sn.countplot('sex',hue='target',data=heart_data)
heart_data.sex.value_counts()


# In[14]:


sn.countplot('fbs',hue='target',data=heart_data)
heart_data.fbs.value_counts()


# In[15]:


pd.crosstab(heart_data.age,heart_data.target).plot(kind="bar",figsize=(24,8))
plt.title('Heart Disease Frequency For Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend(['Female','Male'])
plt.savefig('heartDiseaseAndAges.png')
plt.show()


# In[16]:


heart_data.head()


# In[17]:


pd.crosstab(heart_data.fbs,heart_data.target).plot(kind='bar',figsize=(20,10),color=['green','yellow'])
plt.title("Heart Disease results base on FBS(Fasting Blood Sugar)")
plt.xlabel('FBS >120mg/dl (1=true,0=false)')
plt.ylabel('Disease or not')
plt.legend(["Don't have Disease ","Have Disease"])
plt.show()


# In[18]:


plt.figure(figsize=(26,15))
y = heart_data.target
sn.barplot(heart_data['trestbps'],y)


# In[19]:


plt.figure(figsize=(26,15))
sn.barplot(heart_data['restecg'],y)


# In[20]:


plt.figure(figsize=(10, 10))
sn.barplot(heart_data["exang"],y)


# In[21]:


#plt.figure(figsize=(25, 10))
sn.barplot(heart_data["slope"],y)


# In[22]:


sn.countplot(heart_data['ca'])


# In[23]:


sn.barplot(heart_data['thal'],y)


# In[24]:


sn.scatterplot(x='chol',y='thal',data=heart_data,hue='target')
plt.xlabel('Cholestrol')
plt.ylabel('Thalassemia')
plt.show()


# In[25]:


sn.distplot(heart_data['thal'])


# In[26]:


sn.distplot(heart_data["chol"])
plt.show()


# In[27]:


#store numeric values in cnames
cnames=['age','trestbps','chol','oldpeak','thalach']


# In[28]:


sn.heatmap(heart_data[cnames].corr(),annot=True,cmap='Greens',linewidths=0.1)


# In[29]:


#data preprocessing
x = heart_data.iloc[:,:-1]


# In[30]:


x


# In[31]:


y = heart_data['target']
y


# In[32]:


heart_data.shape


# In[33]:


x.shape


# In[34]:


y.shape


# In[35]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[36]:


x_train


# In[37]:


y_train


# In[38]:


x_test


# In[39]:


y_test


# In[40]:


y_test.shape


# In[41]:


#using LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[42]:


y_pred=lr.predict(x_test)
y_pred


# In[43]:


lr_accuracy =  accuracy_score(y_test,y_pred) *100
lr_precision = precision_score(y_test,y_pred)*100
lr_recall =  recall_score(y_test,y_pred)*100
lr_f1score = f1_score(y_test,y_pred)*100

print('Accurary Rate', lr_accuracy,'%')
print('Precision Score: ', lr_precision,'%')
print('Recall Score:',lr_recall,'%')
print('f1 Score',lr_f1score,'%')


# In[44]:


#describing target_names
Category = ["No,you don\'t have Heart disease","Sorry,you are having Heart Disease"]


# In[45]:


Category


# In[46]:


n_data = np.array([[57,0,1,130,236,0,0,174,0,0.0,1,1,2]])
n_data_prediction_lr = lr.predict(n_data)


# In[47]:


n_data_prediction_lr


# In[48]:


print(Category[int(n_data_prediction_lr)])


# In[ ]:





# In[ ]:





# In[49]:


#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[50]:


x_test


# In[51]:


Y_predict = dt.predict(x_test)


# In[52]:


Y_predict


# In[53]:


dt_accuracy =  accuracy_score(y_test,Y_predict) *100
dt_precision = precision_score(y_test,Y_predict)*100
dt_recall =  recall_score(y_test,Y_predict)*100
dt_f1score = f1_score(y_test,Y_predict)*100

print('Accurary Rate', dt_accuracy,'%')
print('Precision Score: ', dt_precision,'%')
print('Recall Score:',dt_recall,'%')
print('f1 Score',dt_f1score,'%')


# In[54]:


dt.feature_importances_  #youtube mhr pyn kyi yan 


# In[55]:


def plot_feature_importance(model) :
    n_features = 13
    plt.barh(range(n_features), model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),x)
    plt.xlabel('Feature Importance')
    plt.ylabel("Feature")
    plt.ylim(-1,n_features)


# In[56]:


plot_feature_importance(dt)


# In[57]:


#describing target_names
Category = ["No,you don\'t have Heart disease","Sorry,you are having Heart Disease"]


# In[58]:


Category


# In[59]:


new_data = np.array([[57,0,1,130,236,0,0,174,0,0.0,1,1,2]])
new_data_prediction_dt = dt.predict(new_data)


# In[60]:


new_data_prediction_dt


# In[61]:


print(Category[int(new_data_prediction_dt)])


# In[ ]:





# In[ ]:





# In[82]:


#SVM

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x)
x_std = scaler.transform(x)


# In[83]:


x_train_std,x_test_std,y_train,y_test = train_test_split(x_std,y,test_size=0.25,random_state=0)


# In[84]:


x_train_std


# In[85]:


y_train


# In[77]:


#Training the Model
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(x_train_std,y_train)


# In[78]:


y_predSVM = clf.predict(x_test_std)
y_predSVM


# In[79]:


y_test


# In[95]:


svm_accuracy =accuracy_score(y_test,y_predSVM)*100
svm_precision =precision_score(y_test,y_predSVM)*100
svm_recall=recall_score(y_test,y_predSVM)*100
svm_f1 =f1_score(y_test,y_predSVM)*100

print('Accuracy',svm_accuracy,'%')
print('Precision',svm_precision,'%')
print('Recall',svm_recall,'%')
print('F1_score',svm_f1,'%')


# In[87]:


custom_data_svc=np.array([[63 ,1, 3,145,233,1,0,150,0,2.3,0,0,1]])
custom_data_svc_std=scaler.transform(custom_data_svc)
custom_data_prediction_svc=clf.predict(custom_data_svc_std)


# In[88]:


custom_data_prediction_svc


# In[89]:


print(Category[int(custom_data_prediction_svc)])


# In[98]:


algorithms=['Logistic Regression','Decision Tree','Support Vector Machine']
scores=[lr_accuracy,dt_accuracy,svm_accuracy]


# In[99]:


sn.barplot(algorithms,scores,palette="Blues_d")
plt.show()


# In[100]:


pre_scores = [lr_precision,dt_precision,svm_precision]


# In[102]:


plt.figure(figsize=(20,15))
sn.barplot(algorithms,pre_scores,palette='Oranges_d')
plt.show()


# In[104]:


recall_scores = [lr_recall,dt_recall,svm_recall]


# In[105]:


sn.barplot(algorithms,recall_scores,palette="Greens_d")
plt.show()


# In[106]:


f1_scores = [lr_f1score,dt_f1score,svm_f1]


# In[108]:


sn.barplot(algorithms,f1_scores,palette="Greys_d")
plt.show()


# In[ ]:




