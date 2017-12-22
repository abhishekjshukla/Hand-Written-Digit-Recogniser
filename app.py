from flask import Flask,render_template,request,flash,session,g,redirect,url_for
import cv2,json
from PIL import Image
import pandas as pd
from base64 import b64decode
import numpy as np
from sklearn.externals import joblib
app=Flask(__name__)
names = ["NearestNeighbors",
		 "DecisionTree", "RandomForest", "NeuralNet",
		 "NaiveBayes"]
clf=[0]*5
for i in range(0,len(names)):
	clf[i]=joblib.load(names[i]+".pkl")
final=99
@app.route('/',methods = ['POST','GET'])
def home():
	print("methpda ",request.method)
	global final
	session["clf"]=0
	ans=final
	if(request.method=="POST"):
		jsdata="ssss"
		jsdata = request.form['data']
		session['jsdata']=jsdata
		final=get_img(session['jsdata'])
		ans=final
		ans=str(ans)
		
	print("ans is " ,ans,final)
	return render_template("index.html",ans=final)
def get_img(png_arr):
	png_arr=session['jsdata']
	png_arr = png_arr.split(",")
	png_arr = png_arr[1]
	fh = open("image.png", "wb")
	fh.write(b64decode(png_arr))
	fh.close()
	img = cv2.imread('image.png')
	height, width = img.shape[:2]
	res = cv2.resize(img,(width//6, height//6), interpolation =  cv2.INTER_NEAREST)
	cv2.imwrite('digit.png',res)
	test=cv2.imread("digit.png")
	gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
	gray=np.reshape(gray,(1,784))
	ans=clf[int(session["clf"])].predict(gray)
	temp=int(ans[0])
	return temp
@app.route("/get_val", methods=['GET', 'POST'])
def get_val():
	select = request.form.get("classifiers")
	session["clf"]=select
	return(str(select)) 
if __name__ == '__main__':
	app.secret_key = 'my key'
	app.run(debug=True)
