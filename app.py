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

st.subheader('The code implemented for face landmarks and pose landmarks')
st.code('''
cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            #converting the landmarks to a list of numpy array
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            # Concate both face and pose landmarks
            row = pose_row+face_row
            
            # Append class name 
            row.insert(0, class_name)
            
            # Export to CSV using append mode
            with open('Features.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
            
        except:
            pass
                        
        cv2.imshow('Human Behavioral Analysis by Body Language', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()''', language='python')

st.subheader('Plan for the Next Phase')
st.image('Next phase.png')


st.balloons()


