import numpy as np
import pandas as pd
import pickle 
import streamlit as st

def main():        
    st.title('Airlines Price Prediction')
    st.subheader('This project will predict Indian Flight Prices')
    st.info('@author : Kartik Khandelwal')
    st.markdown('<h6><u>Enter Details Here:-</u></h6>',
                     unsafe_allow_html=True)

    #model
    model = pickle.load(open('model.pkl', 'rb'))

    #series
    series_airlines = pd.Series(['Air Asia', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways',
        'Jet Airways Business', 'Multiple carriers', 'Multiple carriers Premium economy',
        'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy'])
    series_source = pd.Series(['Banglore','Chennai', 'Delhi', 'Kolkata', 'Mumbai'])
    series_dest = pd.Series(['Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi'])
    series_stops = pd.Series(['1 Stop', '2 Stops', '3 Stops', '4 Stops', 'Non-Stop'])

    #streamlit front-end
    date_of_journey = st.date_input('Date of Journey')
    col1, col2 = st.columns(2)
    with col1:
        arrival_time = st.time_input('Arrival Time')
        airlines = st.selectbox('Airlines', series_airlines)
        source = st.radio('Source',     series_source)

    with col2:
        dep_time = st.time_input('Departure Time')
        stops = st.selectbox('Total Stops',     series_stops)
        dest = st.radio('Destination',  series_dest)

    #features for prediction
    day = int(date_of_journey.day)
    month = int(date_of_journey.month)

    arrival_hour = int(arrival_time.hour)
    arrival_min = int(arrival_time.minute)

    dep_hour = int(dep_time.hour)
    dep_min = int(dep_time.minute)

    hour = int(arrival_hour - dep_hour)
    minute = int(arrival_min - dep_min)

    airlines_feat = []
    source_feat = []
    dest_feat = []
    stops_feat = []

    for feat, series, name in zip([airlines_feat, source_feat, dest_feat, stops_feat], [series_airlines, series_source, series_dest, series_stops], [airlines, source, dest, stops]):
        for i in series:
            if i == name:
                feat.append(1)
            else:
                feat.append(0)

    features = np.array([day, month, arrival_hour, arrival_min, dep_hour, dep_min, hour, minute, *airlines_feat, *source_feat, *dest_feat, *stops_feat]).reshape(1,-1)

    pred = int(model.predict(features)[0])
    result = f'Your Flight Price is ₹{int(np.exp(pred))}.'

    if st.button('Result', help='Prediction'):
        if source != dest:
            st.success(result)
        else:
            st.error('Source and Destination can not be same.')


if __name__=='__main__':
    main()