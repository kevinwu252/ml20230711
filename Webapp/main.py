import streamlit as st
import numpy as np
import pandas as pd

from sklearn import datasets#資料集
from sklearn.model_selection import train_test_split#切割
from sklearn.decomposition import PCA #降維

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier#鄰近值
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score#評估指標
import matplotlib.pyplot as plt
st.title("人工智慧分類器")
#側邊欄
data_name=st.sidebar.selectbox(
    "請選擇資料集",
    ["iris","wine","cancer"]
)
classifier=st.sidebar.selectbox(
    "請選擇分類器",
    ["KNN","SVM","RandomForset"]
)
#下載資料集
def loadData(name):
    data=None 
    if name=="iris":
        data=datasets.load_iris()
    elif name=="wine":
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    X=data.data#資料本身
    y=data.target#資料標籤
    return X,y    
#顯示資料集資訊
X,y=loadData(data_name)
st.write("### 資料集結構:",X.shape)
st.write("## 資料集分類:",len(np.unique(y)))
# st.write("# 資料集分類:",np.unique(y))
st.write("### 前10筆資料:")
st.write(X[:10])
#定義模型參數
def parameter(clf):
    p={}
    if clf =="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        p["C"]=C#key和值
    elif clf=="KNN":
        K=st.sidebar.slider("K",1,20)
        p["K"]=K
    else:
        max_depth=st.sidebar.slider("max_depth",2,15)#深度
        p["dep"]=max_depth
        trees =st.sidebar.slider("n_estimators",1,100)#估算器
        p["trees"]=trees
    return p 
#取得參數
params=parameter(classifier)

#建立分類器模型
def getClassifier(clf, p):
    now_clf=None#設定一個名稱好讓之後丟東西進去
    if clf=="SVM":
        now_clf=SVC(C=params["C"])#"C"是key
        #如果是建立線性回歸不需要放C=params[]
    elif clf=="KNN":
        now_clf=KNeighborsClassifier(n_neighbors=params["K"])
    else:
        now_clf=RandomForestClassifier(n_estimators=params["trees"],
                                       max_depth=params["dep"],
                                       random_state=123)
    return now_clf
#取得模型物件
clf=getClassifier(classifier,params)

#分割資料集
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.25,
                                               random_state=123,
                                               stratify=y)
                                               #需要有相同比例的資料
                                        
#訓練資料
clf.fit(X_train,y_train)
#預測
y_pred=clf.predict(X_test)
#評估準確率
acc=accuracy_score(y_test,y_pred)#真實的先放，後面的放預測
#顯示結果
st.write("#### 準確率:",acc)

#PCA 降維
pca=PCA(2)
new_X=pca.fit_transform(X)

x1=new_X[:,0]
x2=new_X[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.7,)#c=y顏色
plt.xlabel("X軸")
plt.ylabel("Y軸"    )

#plt.show
st.pyplot(fig)


