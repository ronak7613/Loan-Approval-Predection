#!/usr/bin/env python
# coding: utf-8

# # DMBI OEP : Loan Approval Prection

# #### Group Members
# #### 170420107547	kirtan savani
# #### 170420107551	milan sheta
# #### 170420107556	neel tejani 
# #### 170420107557	pathik thakor
# #### 170420107559	uttam thummer
# #### 170424107001	priyank viradia

# #### Reading Dataset

# In[267]:


import pandas as pd
dataset = pd.read_csv('train.csv')


# #### Data Separation

# In[268]:


inputs = dataset.drop(['Loan_ID','Loan_Status'],axis='columns')


# In[269]:


target = pd.DataFrame(dataset['Loan_Status'])


# #### Data Preprocessing

# In[270]:


inputs.rename(columns={'ApplicantIncome': 'Applicant_Income', 'CoapplicantIncome': 'Coapplicant_Income', 'LoanAmount':'Loan_Amount'},inplace=True)


# In[271]:


#import math
#nanColumns = []
#for i in inputs:
#    for j in inputs[i]:
#        if(pd.isna(j)):
#            nanColumns.append(i)
#            break


# In[272]:


#print(nanColumns)


# In[273]:


#for i in  nanColumns:
 #   inputs[i].fillna(inputs[i].mode()[0],inplace=True)
inputs['Gender'].fillna(inputs['Gender'].mode()[0],inplace=True)
inputs['Married'].fillna(inputs['Married'].mode()[0],inplace=True)
inputs['Dependents'].fillna(inputs['Dependents'].mode()[0],inplace=True)
inputs['Self_Employed'].fillna(inputs['Self_Employed'].mode()[0],inplace=True)
inputs['Loan_Amount'].fillna(inputs['Loan_Amount'].mean(),inplace=True)
inputs['Loan_Amount_Term'].fillna(inputs['Loan_Amount_Term'].mode()[0],inplace=True)
inputs['Applicant_Income'].fillna(inputs['Applicant_Income'].mode()[0],inplace=True)
inputs['Coapplicant_Income'].fillna(inputs['Coapplicant_Income'].mode()[0],inplace=True) 
inputs['Credit_History'].fillna(inputs['Credit_History'].mode()[0],inplace=True)


# In[274]:


inputs['Dependents'].replace('3+','3',inplace=True)


# In[275]:


inputs


# In[ ]:





# #### Label Encoding

# In[276]:


from sklearn.preprocessing import LabelEncoder


# In[277]:


le_gender = LabelEncoder()
le_married = LabelEncoder()
le_education = LabelEncoder()
le_self_employed = LabelEncoder()
le_property_area = LabelEncoder()


# In[278]:


inputs['Gender_n'] = le_gender.fit_transform(inputs['Gender'])
inputs['Married_n'] = le_married.fit_transform(inputs['Married'])
inputs['Education_n'] = le_education.fit_transform(inputs['Education'])
inputs['Self_Employed_n'] = le_self_employed.fit_transform(inputs['Self_Employed'])
inputs['Property_Area_n'] = le_property_area.fit_transform(inputs['Property_Area'])


# In[279]:


inputs.drop(['Gender','Married','Education','Self_Employed','Property_Area'],axis="columns",inplace=True)


# In[280]:


le_target = LabelEncoder()
target['Loan_Status_n'] = le_target.fit_transform(target['Loan_Status'])
target.drop(['Loan_Status'],axis="columns",inplace=True)


# ### Train Test Split

# In[281]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[282]:


x_train, x_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)


# #### Data processing done

# In[283]:


from sklearn import tree


# In[284]:


model = tree.DecisionTreeClassifier()


# In[285]:


model.fit(x_train,y_train)


# In[286]:


##model.score(inputs,target)


# In[287]:


res = model.predict(x_test)


# In[288]:


a = list(res)


# In[289]:


k = 0
count = 0 
print("Predictions")
print("Actual\tPredicted")
for i in y_test['Loan_Status_n']:
    print(a[k],'\t',i)
    if(a[k] != i):
        count += 1
    k += 1
#print("Accuracy: ",(k-count)/k)
print("\nAccuracy")
accuracy_score(y_test,res)


# In[290]:


print("Correct predictions: ", k-count)
print("Wrong predictions: ",count)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Custom Data

# In[439]:


data = pd.read_csv('test.csv')
out = data.drop(['Loan_ID','Loan_Status'],axis='columns')
out.rename(columns={'ApplicantIncome': 'Applicant_Income', 'CoapplicantIncome': 'Coapplicant_Income', 'LoanAmount':'Loan_Amount'},inplace=True)
fo = data[['Loan_ID','Loan_Status']]
import os
import os.path
from os import path
if(path.exists("oo.csv")):   
    os.remove("oo.csv")


# In[440]:


nanColumnsO = []
for i in out:
    for j in out[i]:
        if(pd.isna(j)):
            nanColumnsO.append(i)
            break
            
print(nanColumnsO)


# In[441]:


#for i in  nanColumns:
#    out[i].fillna(out[i].mode()[0],inplace=True)
out['Gender'].fillna(out['Gender'].mode()[0],inplace=True)
out['Married'].fillna(out['Married'].mode()[0],inplace=True)
out['Dependents'].fillna(out['Dependents'].mode()[0],inplace=True)
out['Self_Employed'].fillna(out['Self_Employed'].mode()[0],inplace=True)
out['Loan_Amount'].fillna(out['Loan_Amount'].mean(),inplace=True)
out['Loan_Amount_Term'].fillna(out['Loan_Amount_Term'].mode()[0],inplace=True)
out['Credit_History'].fillna(out['Credit_History'].mode()[0],inplace=True)
out['Applicant_Income'].fillna(out['Applicant_Income'].mode()[0],inplace=True)
out['Coapplicant_Income'].fillna(out['Coapplicant_Income'].mode()[0],inplace=True) 
out['Dependents'].replace('3+','3',inplace=True)


# In[442]:


out


# In[443]:


le_genderO = LabelEncoder()
le_marriedO = LabelEncoder()
le_educationO = LabelEncoder()
le_self_employedO = LabelEncoder()
le_property_areaO = LabelEncoder()

out['Gender_n'] = le_genderO.fit_transform(out['Gender'])
out['Married_n'] = le_marriedO.fit_transform(out['Married'])
out['Education_n'] = le_educationO.fit_transform(out['Education'])
out['Self_Employed_n'] = le_self_employedO.fit_transform(out['Self_Employed'])
out['Property_Area_n'] = le_property_areaO.fit_transform(out['Property_Area'])

out.drop(['Gender','Married','Education','Self_Employed','Property_Area'],axis="columns",inplace=True)


# In[444]:


res = model.predict(out)
a = list(res)


# In[445]:


o=0
for i in a:
    o += 1
    if(i): 
        fo.loc[o-1,"Loan_Status"]  = 'Y'
        print('y')
    else: 
        fo.loc[o-1,"Loan_Status"] = 'N'
        print('n')


# In[446]:


o


# In[447]:


fo['Loan_ID'] +" " +  fo['Loan_Status']


# In[448]:


fo.to_csv('oo.csv')


# In[ ]:




