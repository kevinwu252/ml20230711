import numpy as np
import pandas as pd
import streamlit as st

st.title("測試首頁")
st.write("AAAA")
a=100
st.write(a)
st.write("-------------------------")
st.write("表格-----------------")
df = pd.DataFrame({"F1":[1,2,4,5],"F2":[11,55,22,33]})
st.write(df)

st.write("-------------------------")

st.write("核取方塊(check box)")
cb=st.checkbox("是否參加?")
if cb:
    st.info("參加")
st.write("-------------------------")

st.write("選項案鈕")
gender=st.radio("選擇性別:",("男","女","空白"))
st.write(gender)
st.success(gender)
st.write("-------------------------")

st.write("下拉選單")
option= st.selectbox("請選擇食物",["麵包","牛排","香蕉","漢堡"])
st.write(option)
st.success(option)
"我想吃",option

# st.write("-------------------------")
# st.write("進度條")
# import time
# load=st.empty()
# bar=st.progress(0)
# for i in range(100):
#     load.text(f"目前的進度:{i+1}%")
#     bar.progress(i+1)
#     time.sleep(0.1)#每0.1秒跑1%
st.write("-------------------------")

def aa():
    st.text("我愛吃牛排")

st.write("按鈕")
btn=st.button("確定")
if btn:
    st.info("已確認")
    aa()
else:
    st.info("未確認")    
st.write("-------------------------")

st.write("滑桿")
num=st.slider("請選擇數量:",1,5)
"num=",num
st.write("num=",num)
st.write("-------------------------")

st.write("檔案上傳")
loader=st.file_uploader("請選擇CSV檔:")


if loader is not None:
    df2=pd.read_csv(loader,header=None)
    st.dataframe(df2)
    st.table(df2.iloc[:2])
# df2=pd.read_csv(loader)
# st.dataframe(df2)
st.write("-------------------------")

st.write("隱藏欄位(文字")
hidden=st.expander("按下之後展開")
hidden.write("0123156489")
st.write("-------------------------")

st.write("圖片上傳+圖片展示")
img=st.file_uploader("請選擇圖檔:",type=['png','jpg','jpeg'])
if img is not None:
    st.image(img)
st.write("-------------------------")

st.write("側邊攔")
side01=st.sidebar.button("click me")
side02=st.sidebar.checkbox("OK?")
st.write("-------------------------")

st.write("分欄")
left,right=st.columns(2)
left.write("aaa")
right.write(btn)


