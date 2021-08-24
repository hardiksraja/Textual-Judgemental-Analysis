import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import transformers as transfs
import torch
import time
import joblib


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    st.title('Judgemental/Non-Judgemental Statement Classification')

    statement = st.text_input('Enter your statement', help='Enter the statement that you will like to classify as Judgemental or Non-Judgemental')

    pressed = st.button('Predict', help ='Submit to get results')
    
    if pressed:
        if(statement == ''):
            st.error('Error!! : Kindly enter text before submitting')
        else:
            with st.spinner(text='In progress'):
                result = evaluateModel(statement)
                st.write('Model Result : ', '**',result,'**')
                st.balloons()
        
@st.cache
def loadModel():
    
    #Importing pre-trained DistilBERT model and tokenizer
    model_class, tokenizer_class, pretrained_weights = (transfs.DistilBertModel, transfs.DistilBertTokenizer, 'distilbert-base-uncased')
    
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    
    lr_clf = joblib.load('judgement_predictor.pkl')
    
    return tokenizer, model, lr_clf   

def evaluateModel(inputStatement):
    tokenizer, model, lr_clf = loadModel()
    
    sentence = tokenizer.encode(inputStatement, add_special_tokens=True)
    
    input_ids = torch.tensor(np.array(sentence).reshape(1,len(sentence)))
    
    with torch.no_grad():
        last_hidden_states = model(input_ids)
        
    # Slice the output for the first position for all the sequences, take all hidden unit outputs
    features = last_hidden_states[0][:,0,:].numpy()
    
    expander = st.beta_expander("Model Intermediate Results")
    expander.write("Output of DistilBERT Model")
    
    df = pd.DataFrame(features, columns=('Feature %d' % i for i in range(features.size)))
    expander.dataframe(df)  # Same as st.write(df)

    y_prob = lr_clf.predict_proba(features)
    predicted_indice = (y_prob[0][1] > 0.5).astype(int)
    
    labels = ['Judgemental','Non Judgemental']
      
    expander.write("Output of Logistic Regression Model")
    df_1 = pd.DataFrame(y_prob, columns=[labels])
    
    expander.write(df_1)
    
    percentage = np.round(100*y_prob[0][predicted_indice], 2)
    result = '{0} with {1}%'.format(labels[predicted_indice], percentage)
    return result

if __name__ == "__main__":
    main()