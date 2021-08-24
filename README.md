# Textual-Judgemental-Analysis
A deep learning model for classification of judgemental and non-judgemental statements; to overcome challenges of identifying factual content, compared to fictional/nonfactual content.

We have solved this problem using a pre trained model called DistilBERT. The output of DistilBERT model is used in Logistic Regression Model to predict the class. DistilBERT is a transformer based model which is used for classification of text.

We have used two models here :  
Model Selection : Supervised Learning > Binary Class Classification > DistilBERT > Logistic Regression  
DistilBERT processes the sentence and passes along some information it extracted from it on to the next model. DistilBERT is a smaller version of BERT developed and open sourced by the team at HuggingFace. It’s a lighter and faster version of BERT that roughly matches its performance.

The next model, a basic Logistic Regression model from scikit learn will take in the result of DistilBERT’s processing, and classify the sentence as either Judgemental or Non Judgemental (0 or 1, respectively).

The data we pass between the two models is a vector of size 768. We can think of this of vector as an embedding for the sentence that we can use for classification.

<b>
Refer, Judgemental_Non_Judgemental_Anjali_Hardik_Jayant.pptx for detailed description on the project  
</b>
  
# Model Evaluation Metrics – Accuracy and ROC AUC Score  
We built 4 different models and finally selected DistilBERT  
<br/>
![ModelComparison](https://user-images.githubusercontent.com/53824674/130576412-8d82de04-ca51-4677-8d92-4fa6baa2b4a9.png)

# Deployment/Demo  
Deployed at : https://judgement-prediction-streamlit.herokuapp.com/  
<br/>
![ModelDeployment](https://user-images.githubusercontent.com/53824674/130576560-00891ec7-fd5e-4b20-a72f-5af8a3a7445d.png)


Deployment Files: The models were deployed on Heroku with UI built via StreamLit
<br/>

<b>Refer, 2021-07-14_StreamLit & Heroku_HardikRaja.pptx for commands and background on Deployment</b>

<br/>
judgement_predictor.pkl - Saved Logistic Regression pickle file  
Procfile - Executing the StreamLit App  
requirements.txt - Required by Heroku  
setup.sh - File required to deploy StreamLit App on Heroku  
streamlit_app.py - The StreamLit App  
