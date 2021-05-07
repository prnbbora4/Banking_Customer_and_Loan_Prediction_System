from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import numpy as np
import pandas as pd

# import math


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/loan_predict')
def loan_predict():
    model = load_model('loan_predict.h5')
    # sc = MinMaxScaler()
    Gender = request.args.get('Gender')
    Married = request.args.get('Married')
    Dependents = request.args.get('Dependents', type=float)
    Education = request.args.get('Education')
    Self_Employed = request.args.get('Self_Employed')
    ApplicantIncome = request.args.get('ApplicantIncome', type=int)
    CoapplicantIncome = request.args.get('CoapplicantIncome', type=float)
    LoanAmount = request.args.get('LoanAmount', type=int)
    Loan_Amount_Term = request.args.get('Loan_Amount_Term', type=float)
    Credit_History = request.args.get('Credit_History', type=float)
    Property_Area = request.args.get('Property_Area')

    predict = request.args.get('predict', type=str)

    if predict == "predict":
        print(type(Dependents))
        inputs = {'Dependents': [float(Dependents or 0)],
                  'ApplicantIncome': [int(ApplicantIncome or 0)],
                  'CoapplicantIncome': [float(CoapplicantIncome or 0)],
                  'LoanAmount': [int(LoanAmount or 0)],
                  'Loan_Amount_Term': [float(Loan_Amount_Term or 0)],
                  'Credit_History': [float(Credit_History or 0)],
                  'Gender_Female': [1 if Gender =="Female" else 0],
                  'Gender_Male': [1 if Gender =="Male" else 0],
                  'Married_No': [1 if Married =="No" else 0],
                  'Married_Yes': [1 if Married =="Yes" else 0],
                  'Education_Graduate': [1 if Education =="Graduate" else 0],
                  'Education_Not Graduate': [1 if Education =="Not Graduate" else 0],
                  'Self_Employed_No': [1 if Self_Employed =="No" else 0],
                  'Self_Employed_Yes': [1 if Self_Employed =="Yes" else 0],
                  'Property_Area_Rural': [1 if Property_Area =="Rural" else 0],
                  'Property_Area_Semiurban': [1 if Property_Area =="Semiurban" else 0],
                  'Property_Area_Urban': [1 if Property_Area =="Urban" else 0]
                  }

        # inputs = np.array(
        #     [[Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
        #      Loan_Amount_Term, Credit_History, Property_Area]]).reshape(-1, 1)

        # inputs = {'Gender':[Gender],
        #     'Married':[Married],
        #     'Dependents':[Dependents],
        #     'Education':[Education],
        #     'Self_Employed':[Self_Employed],
        #     'ApplicantIncome':[ApplicantIncome],
        #     'CoapplicantIncome':[CoapplicantIncome],
        #     'LoanAmount':[LoanAmount],
        #     'Loan_Amount_Term':[Loan_Amount_Term],
        #     'Credit_History':[Credit_History],
        #     'Property_Area':[Property_Area]}


        # df = pd.DataFrame(inputs,
        #                   columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
        #                            'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        #                            'Property_Area'])
        inputs = pd.DataFrame(inputs)
        print(inputs)

        sc = MinMaxScaler()

        final_features = sc.fit_transform(inputs)
        print(final_features)

        #
        prediction = model.predict(final_features)
        prediction = (prediction>0.58)

        # X=pd.read_excel('Test_loan.xlsx')
        # X_test = sc.fit_transform(X)
        # y_pred = model.predict(X_test)
        # y_pred = (y_pred>0.58)
        # output = round(prediction[0], 2)
        return render_template('loan_predict.html', p_text="Prediction : " + ("Approved" if prediction[0] else "Not Approved"))
    return render_template('loan_predict.html')


@app.route('/customer_predict')
def customer_predict():
    return render_template('customer_predict.html')


if __name__ == '__main__':
    app.run(debug=True)
# activate chatbot
