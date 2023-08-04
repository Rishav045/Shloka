from flask import Flask , request , redirect
from flask_cors import CORS, cross_origin
app=Flask(__name__)
cors=CORS(app)
app.config['CORS_HEADERS']='Content-Type'
import pickle
import sklearn
import pandas as pd
vectorizer = pickle.load(open('vectorizer.pkl','rb'))
vecs=pickle.load(open('vecs_tf_idf.pkl','rb'))
from sklearn.metrics.pairwise import cosine_similarity
df1=pd.read_csv("Bhagwad_Gita.csv")
@app.route('/')
def root():
    return {"messsage":"Here is the home page"}
@app.route('/slok',methods=['POST'])
def topFive():
  if request.method=='POST':
    data=request.json
    queryLanguage=data['queryLanguage']
    query=data['query']
    query=query.lower()
    queryLanguage= queryLanguage.lower()
    q=vectorizer.transform([query])
    arr=[]
    for i in range (0,vecs.shape[0]):
        sim = cosine_similarity(q,vecs[i])
        temp= [i,sim[0][0]]
        arr.append(temp)
    arr=sorted(arr,key=lambda x:x[1])
    arr=arr[::-1]
    result=[]
    for i in range (0,5):
        if(queryLanguage=='hindi'):
            result.append(df1['HinMeaning'][arr[i][0]])
        
        elif (queryLanguage=='english'):
            result.append(df1['EngMeaning'][arr[i][0]])
       
        elif (queryLanguage=='sanskrit'):
            result.append(df1['Shloka'][arr[i][0]])
        
    return result



if __name__ == '_main_':
    app.run(debug=True)