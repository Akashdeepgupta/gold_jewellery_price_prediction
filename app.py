from flask import Flask, request, render_template#,redirect,url_for
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors=CORS(app)
model=pickle.load(open('model_pickle.pkl','rb'))
car=pd.read_csv('GoldData_Sheet_updeted.csv')


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/flask',methods = ['POST'])
def hello_flask():
    type = request.form.get('Jewellery type')
    weight = request.form.get('weight')
    carat = request.form.get('carat')
    color = request.form.get('color')
    hallmark = request.form.get('Hallmark')

    prediction=model.predict(pd.DataFrame(columns=['Jewellery type', 'Weight(in grams)', 'Purity(Carat)', 'Color', 'Hallmarked '], data=np.array([type,weight,carat,color,hallmark]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__ == '__main__':
    app.run(debug='True') 