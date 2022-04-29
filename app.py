import streamlit as st
import pandas as pd
# import cv2
# import numpy as np

# img_file_buffer = st.camera_input("Take a picture")

# if img_file_buffer is not None:s
#     # To read image file buffer with OpenCV:
#     bytes_data = img_file_buffer.getvalue()
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#     # Check the type of cv2_img:
#     # Should output: <class 'numpy.ndarray'>
#     st.write(type(cv2_img))

#     # Check the shape of cv2_img:
#     # Should output shape: (height, width, channels)
#     st.write(cv2_img.shape)

st.set_page_config (
    page_title="Minor Project",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.balloons()
st.title('Human Behaviour Analysis by Body Language using Machine Learning')
df = pd.read_csv('humanBehaviourFace.csv')

st.write("Human Behavioral analysis via body language is primarily concerned with two areas of computational intelligence: computer vision and multivariate classification techniques. Humans have a natural inclination to communicate through their body language, which implies that a person's body posture and facial expression can help us predict their behavior. In this small project, we employed some of the most cutting-edge open-source technology to predict an individual's behavior simply by using an algorithm based on that person's facial expression and body position.")

st.subheader('Motivation')
st.write("Our motivation for working on this project stems from the fact that in order to create intelligent conversational bots, we need to create a system that can understand nonverbal human communication. We need a system to understand human body language and predict the mood of the person. In many different fields, the synchronisation of verbal and nonverbal human behaviour during interpersonal encounters has been explored. Human behaviour, such as facial expressions and body actions, can be adapted to make communication more fluid, effective, and clear. Body language, which includes facial expressions, head, limb, and other postures and motions, is an important part of nonverbal communication in humans. It expresses a person's views and sentiments towards everyone else. Depending on the individual, various body movements and mannerisms transmit different meanings.​ We've trained the system to read the movements and gestures in order to predict human conduct and interact with them non-verbally. Because human behaviour is circumstantial, it can be investigated using the system\'s analysis.")


coll, colll = st.columns(2)
with coll: 
	st.image('pose.gif')
with colll:
	st.image('holistic.gif')

st.subheader('The Custom dataset for Facial Emotion Detection by using Face Mesh')
st.write(df.head())
st.write(df.iloc[200:300])
st.subheader('Tools Used')
col, col2, col3 = st.columns(3)
with col:
	st.write('MEDIAPIPE Framework')
	st.image('mediapipe.png')
with col2:
	st.write('Keras Library')
	st.image('keras.png')
with col3:
	st.write('Tensorflow Framework')
	st.image('tensorflow.png')
col9, col8, col7 = st.columns(3)
with col9:
	st.write('OpenCV Library')
	st.image('opencv.png')
with col8:
	st.write('PandasLibrary')
	st.image('pandas.png')
with col7:
	st.write('Pickle Library')
	st.image('pickle.png')

st.subheader('Work done till now')
st.write("Till now we have created a customized dataset wherein, the facial and pose estimation landmarks for 18 features have been captured. ​For now, to classify the features (both facial and pose) we have used simple Random Forest Classifier. ​The probability of different features is not accurate; therefore, we will work upon the model's accuracy and merge the complete facial and pose landmarks.")
st.balloons()
