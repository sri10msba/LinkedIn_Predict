import streamlit as st 

#DATA - pulling it in, cleaning it, and training the model
import pandas as pd
import numpy as np

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return np.where(x == 1,1,0)
ss = ss = pd.DataFrame({
    "income": np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent": np.where(s["par"], 1, 0),
    "married": np.where(s["marital"] == 1, 1, 0),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"] > 98, np.nan, s["age"]),
    "sm_li": clean_sm(s["web1h"])
})
ss = ss.dropna() 
ss.head()

y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "age", "female"]]

# Split data into train and test
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=123) # set for reproducibility

#LOGISTIC REGRESSION MODEL
# Initialize algorithm 
lr = LogisticRegression(class_weight='balanced')
# Fit algorithm to training data
lr.fit(x_train, y_train)


#STREAMLIT APP INPUT/INTERFACE
st.image('linkedin.jpg', caption='Image from: https://kinsta.com/blog/how-to-create-a-company-page-on-linkedin/')
st.markdown("# LinkedIn User Predictor!")
st.markdown("## Welcome to the app that let's you check if an individual maybe a LinkedIn User")
st.markdown("#### Enter/Select the correct parameter below and see your results!")

#Assign the values for the input categories for the interface
income_categories = {"Less than $10,000": 1,
                     "10 to under $20,000": 2,
                     "20 to under $30,000": 3,
                     "30 to under $40,000": 4,
                     "40 to under $50,000": 5,
                     "50 to under $75,000": 6,
                     "75 to under $100,000": 7,
                     "100 to under $150,000": 8,
                     "$150,000 or more": 9,
                     "Don't know": 98,
                     "I would rather not say": 99}
education_categories = {"Less than high school (Grades 1-8 or no formal schooling)": 1,
                        "High school incomplete (Grades 9-11 or Grade 12 with No diploma": 2,
                        "High school graduate (Grade 12 with diploma or GED certificate)":3,
                        "Some college, no degree (includes some community college)":4,
                        "Two-year associate degree from a college or university":5,
                        "Four-year college or university degree/bachelor's degree (e.g., BS, BA, AB)":6,
                        "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":7,
                        "Postgraduate or professional degree, including master's, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)":8,
                        "Don't know": 98,
                        "I would rather not say":99}
parent_categories = {"Yes":1, "No":2, "Don't know":8, "I would rather not say":9}
married_categories = {"Married":1, "Living with a partner":2,"Divorced":3, "Separated":4, "Widowed":5, "Never been married":6, "Don't know":8, "I would rather not say":9}
gender_categories = {"Male":1, "Female":2, "Don't know":98, " I would rather not say":99}

   
#Grab inputs from user in the app Interface & displaying category and value to the user
income = st.selectbox("What is the individual's household income?", list(income_categories.keys()))
income_num = income_categories[income]
    
education = st.selectbox("What is the individual's highest level of education/degree completed?", list(education_categories.keys()))
educ_num = education_categories[education] 
                            
parent = st.selectbox("Is the individual a parent of a child under 18 living in their home?", list(parent_categories.keys()))
parent_num = parent_categories[parent]
                           
married = st.selectbox("What is the individual's marital status?", list(married_categories.keys()))
married_num = married_categories[married]

gender = st.radio("What is the individual's gender?", list(gender_categories.keys()))
gender_num = gender_categories[gender]

age = st.slider("What is the individual's age?", min_value = 0, max_value = 100, step = 1)
if age < 97:
    age_num = age
elif age >= 97:
    age_num = 97
else:
    age_num = 98

st.markdown("### Let's summerize your selections:")
st.write(f"For income you selected: {income}.")
st.write(f"For education level you selected: {education}.") 
st.write(f"For the parenting question you selected: {parent}.")
st.write(f"For the marriage question you selected: {married}.")
st.write(f"For gender you selected: {gender}.")
st.write(f"For age you selected: {age}.")

    
#PASSING APP DATA TO MODEL FOR PREDICTION    
# New data for predictions
appdata = pd.DataFrame({
    "income":[income_num],
    "education":[educ_num],
    "parent":[parent_num],
    "married":[married_num],
    "age":[age_num],
    "female":[gender_num]})

PredictionResult = lr.predict(appdata)

if PredictionResult == 1:
    st.markdown("#### 👍 This individual is likely to be on LinkedIn!")
else:
    st.markdown("#### 👎 This individual is *not* likely to be on LinkedIn!")