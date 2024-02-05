import streamlit as st
from pickle import load
import numpy as np
import pandas as pd


st.markdown('<style>h1 { color: navy; text-align: center;}</style>', unsafe_allow_html=True)

st.title("L.A. CRIME")

image = "../data/images/POLICE.png"

st.image(image, output_format="PNG")

type_of_crime = st.selectbox(
    "Crime committed: ",
    ("Select a type of crime","Burglary", "Child Abuse", "Homicide", "Rape", "Robbery")
)

descent_dict = {"Other Asian":4, "Black":0, "Chinese":6, "Cambodian":12, "Filipino":7, 
                "Guamanian":16, "Hispanic/Latin/Mexican":1, "American Indian/Alaskan Native":9, 
                "Japanese":11, "Korean":8, "Laotian":18, "Other":5, "Pacific Islander":15, 
                "Samoan":17, "Hawaiian":14, "Vietnamese":10, "White":2, "Unknown":3, 
                "Asian Indian":13}

sex_dict = {'Female':0, 'Male':1, 'Other':2}

area_name_dict = {"Southwest":0, "Central":1, "N Hollywood":2, "Mission":3, "Harbor":4,
                 "Rampart":5, "Hollenbeck":6, "77th Street":7, "Wilshire":8, "Hollywood":9, 
                 "Northeast":10, "West LA":11, "Van Nuys":12, "West Valley":13, "Newton":14,
                 "Olympic":15, "Southeast":16, "Topanga":17, "Foothill":18, "Pacific":19, 
                 "Devonshire":20
}

class_dict={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}

if type_of_crime == "Burglary":

    latitude = st.slider('Select latitude', min_value=33.978735, max_value=34.328900, step=0.00001)
    longitude = st.slider('Select longitude', min_value=-118.667600, max_value=-118.260900, step=0.00001)
    
    if st.button("prediction"):
        # Define 'row' inside the button click event
        row = [longitude, latitude]
        
        array = np.array(row).reshape(1, -1)  # Reshape to 2D array
        
        # load model
        model = load(open("../models/best_model_burglary.pk", "rb"))
        y_pred = model.predict(array)[0]
        pred_class = class_dict[y_pred]

        st.text("Possible burglary gang: " + pred_class)

        data = {'latitude': [latitude], 'longitude': [longitude]}

        df = pd.DataFrame(data)
        st.map(df, size=20, color='#0044ff')

elif type_of_crime == "Robbery":

    latitude = st.slider('Select latitude', min_value=33.978735, max_value=34.328900, step=0.00001)
    longitude = st.slider('Select longitude', min_value=-118.667600, max_value=-118.260900, step=0.00001)
    
    if st.button("prediction"):
        # Define 'row' inside the button click event
        row = [longitude, latitude]
        
        array = np.array(row).reshape(1, -1)  # Reshape to 2D array
        
        # load model
        model = load(open("../models/model_robbery.pk", "rb"))
        y_pred = model.predict(array)[0]
        pred_class = class_dict[y_pred]

        st.text("Possible robbery gang: " + pred_class))

        data = {'latitude': [latitude], 'longitude': [longitude]}
        
        df = pd.DataFrame(data)
        st.map(df, size=20, color='#0044ff')

elif type_of_crime == "Rape":

    age = st.select_slider("Select Age: ", np.arange(1, 99))
    descent = st.selectbox("Select Descent: ", [key for key in sorted(descent_dict)])
    area = st.selectbox("Select Area: ", [key for key in sorted(area_name_dict)])

    fact_descent=descent_dict[descent]
    fact_area=area_name_dict[area]

    if st.button("prediction"):
        # Define 'row' inside the button click event
        row = [age, fact_descent, fact_area]
        
        array = np.array(row).reshape(1, -1)  # Reshape to 2D array
        
        # load model
        model = load(open("../models/best_model_rapist.pk", "rb"))
        y_pred = model.predict(array)[0]
        pred_class = class_dict[y_pred]

        st.text("Possible rapist profile: " + pred_class))

elif type_of_crime == "Homicide":

    age = st.select_slider("Select Age: ", np.arange(1, 99))
    sex = st.selectbox("Select Sex: ", [key for key in sorted(sex_dict)])
    descent = st.selectbox("Select Descent: ", [key for key in sorted(descent_dict)])
    area = st.selectbox("Select Area: ", [key for key in sorted(area_name_dict)])

    fact_descent=descent_dict[descent]
    fact_sex=sex_dict[sex]
    fact_area=area_name_dict[area]

    if st.button("prediction"):
        # Define 'row' inside the button click event
        row = [age, fact_sex, fact_descent, fact_area]
        
        array = np.array(row).reshape(1, -1)  # Reshape to 2D array
        
        # load model
        model = load(open("../models/best_model_killers.pk", "rb"))
        y_pred = model.predict(array)[0]
        pred_class = class_dict[y_pred]

        st.text("Possible homicide profile: " + pred_class)
    

elif type_of_crime == "Child Abuse":

    age = st.select_slider("Select Age: ", np.arange(1, 17))
    sex = st.selectbox("Select Sex: ", [key for key in sorted(sex_dict)])
    descent = st.selectbox("Select Descent: ", [key for key in sorted(descent_dict)])

    fact_descent=descent_dict[descent]
    fact_sex=sex_dict[sex]

    if st.button("prediction"):
        # Define 'row' inside the button click event
        row = [age, fact_sex, fact_descent]
        
        array = np.array(row).reshape(1, -1)  # Reshape to 2D array
        
        # load model
        model = load(open("../models/best_model_child.pk", "rb"))
        y_pred = model.predict(array)[0]
        pred_class = class_dict[y_pred]

        st.text("Possible child molester: " + pred_class)

else:
    st.text("No crime selected")