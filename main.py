import streamlit as st
import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
    st.header("Live stream processing")

    sign_language_det = "Sign Language Live Detector"
    app_mode = st.sidebar.selectbox( "Choose the app mode",
        [
            sign_language_det
        ],
    )

    st.subheader(app_mode)

    if app_mode == sign_language_det:
        sign_language_detector()
 

def sign_language_detector():

    class OpenCVVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            mp_holistic = mp.solutions.holistic # Holistic model
            mp_drawing = mp.solutions.drawing_utils # Drawing utilities
            
            # Actions that we try to detect
            actions = np.array(['hello', 'thanks', 'iloveyou'])

            # Load the model from Modelo folder:

            model = load_model('model.h5',actions)

            # 1. New detection variables
            sequence = []
            sentence = []
            threshold = 0.8

            # Set mediapipe model 
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while True:
                    #img = frame.to_ndarray(format="bgr24")
                    flip_img = cv2.flip(img,1)

                    # Make detections
                    image, results = mediapipe_detection(flip_img, holistic)
                    
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # 2. Prediction logic
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]
                    
                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        #print(actions[np.argmax(res)])

                    #3. Viz logic
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5: 
                            sentence = sentence[-5:]

                        # Viz probabilities
                        image = prob_viz(res, actions, image)

                    cv2.rectangle(image, (0,0), (640, 40), (234, 234, 77), 1)
                    cv2.putText(image, ' '.join(sentence), (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    return av.VideoFrame.from_ndarray(image,format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
    )


if __name__ == "__main__":
    main()

