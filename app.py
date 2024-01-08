import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load the trained iris model
model = pickle.load(open('xg_boost_model.pkl','rb'))

def main():

    # Title of the app page
    st.title('Flight Engine Safety')

    # Add a heading for input features
    st.subheader('Enter Features of the Flight!')

    # Rquest for input fatures, but replod with some default values
    time =  st.text_input('Time', 11.50)
    speed =  st.text_input('Speed', 16000)
    torque =  st.text_input('Torque', 100)
    f1 =  st.text_input('f1', -70.5672)
    f2 =  st.text_input('f2', 25.375701)
    f3 =  st.text_input('f3', 137.921211)
    f4 =  st.text_input('f4', 9.112071)
    f5 =  st.text_input('f5', -4.012787)
    f6 =  st.text_input('f6', 89.006213)
    f7 =  st.text_input('f7', -9.112071)
    f8 =  st.text_input('f8', 1.665377)

    # Get predictions when the button is pressed
    if st.button('Get Prediction'):

        # run predictions
        pred = model.predict(np.array([[float(time),float(speed),float(torque),float(f1),float(f2),float(f3),float(f4),float(f5),float(f6),float(f1),float(f7),float(f8)]]))

        st.success('Predicted Safety:' + pred[0])


if __name__ == "__main__":
    main()
