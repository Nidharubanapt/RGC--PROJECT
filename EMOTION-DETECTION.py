#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import speech_recognition as sr
import pyttsx3


# In[2]:


df=pd.read_csv("emotion_dataset_raw.csv")


# In[3]:


df1=pd.read_csv("emotional text new.txt")
df1


# In[4]:


data = pd.concat([df,df1], ignore_index=True)
data


# In[5]:


data['Emotion'].value_counts()


# In[6]:


sns.countplot(x='Emotion',data=data)


# In[7]:


selected_Emotion = ['joy', 'sadness','anger','surprise','neutral']
data_subset = data[data['Emotion'].isin(selected_Emotion)]
data_subset


# In[8]:


import neattext.functions as nfx
data_subset['Clean_Text']=data_subset['Text'].apply(nfx.remove_userhandles)


# In[9]:


dir(nfx)


# In[10]:


data_subset['Clean_Text']=data_subset['Clean_Text'].apply(nfx.remove_stopwords)


# In[11]:


data_subset


# In[12]:


x=data_subset['Clean_Text']
y=data_subset['Emotion']


# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[14]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[15]:


pipe_lr=Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
pipe_lr.fit(x_train,y_train)
pipe_lr.score(x_test,y_test)


# In[16]:


svm_classifier = SVC(kernel='rbf', C=10, probability=True)


# In[17]:


count_vectorizer = CountVectorizer()


# In[18]:


pipe_svm = Pipeline([
    ('cv', count_vectorizer),
    ('svc', svm_classifier)
])


# In[19]:


pipe_svm.fit(x_train, y_train)


# In[20]:


accuracy = pipe_svm.score(x_test, y_test)


# In[21]:


print("Accuracy:", accuracy)


# In[22]:


pipe_rf=Pipeline(steps=[('cv',CountVectorizer()),('rf',RandomForestClassifier(n_estimators=10))])
pipe_rf.fit(x_train,y_train)
pipe_rf.score(x_test,y_test)


# In[23]:


recognizer = sr.Recognizer()
text_to_speech_engine = pyttsx3.init()


# In[24]:


def capture_voice():
    with sr.Microphone() as source:
        print("Speak something:")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None


# In[25]:


def text_to_speech(text):
    text_to_speech_engine.say(text)
    text_to_speech_engine.runAndWait()


# In[41]:


voice_input = capture_voice()
if voice_input:
    # Preprocess the voice input
    clean_voice_input = nfx.remove_userhandles(voice_input)
    clean_voice_input = nfx.remove_stopwords(clean_voice_input)

    # Perform emotion detection
    predicted_emotion = pipe_lr.predict([clean_voice_input])[0]
    # Display the predicted emotion
    print("Predicted Emotion:", predicted_emotion)

    # Convert the predicted emotion to speech
    text_to_speech(predicted_emotion)


# In[39]:


import joblib
pipeline_file = open("text_emotion_new.pkl","wb")
joblib.dump(pipe_lr,pipeline_file)
pipeline_file.close()


# In[ ]:




