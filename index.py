import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

model= pickle.load(open('mnb_tfid.pkl','rb'))
vectorizer= pickle.load(open('tfid_Vactorizer.pkl','rb'))
def transforms_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    temp = []
    for word in text:
        if word.isalnum():
            temp.append(word)

    text = temp[:]
    temp.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            temp.append(word)

    text = temp[:]
    temp.clear()

    for word in text:
        temp.append((ps.stem(word)))

    return " ".join(temp)


st.title("SMS / Email Spam Massage Classifer")
input_massage = st.text_area('Enter Massage Here')
if st.button("Predict"):
    # 1: Preprosessing
    transforms_massage = transforms_text(input_massage)
    # 2: Vactorize
    vactor_input = vectorizer.transform([transforms_massage])
    # 3. predict
    result = model.predict(vactor_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")









