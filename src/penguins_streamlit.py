import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set the title and description of the app
st.title(body="Penguin Classifier")
st.markdown(body="""This app uses 6 inputs to predict the species of penguin using 
            a model built on the Palmer's Penguins dataset. Use the form below to get started!""")

# Load the data
penguin_file = "./data/penguins.csv"
penguin_df = pd.read_csv(filepath_or_buffer=penguin_file)

# Drop null values
penguin_df.dropna(inplace=True)

# Split the data into features and output
features = penguin_df.drop(columns=["species", "year"])
output = penguin_df["species"]

# One-hot encode the features
features = pd.get_dummies(data=features)

# One-hot encode the output
output, uniques = pd.factorize(values=output)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.8)

# Create a random forest classifier
rfc = RandomForestClassifier(random_state=15)

# Train the classifier
rfc.fit(X=X_train, y=y_train)

# Make predictions
y_pred = rfc.predict(X=X_test)

# Calculate the accuracy
score = accuracy_score(y_true=y_test, y_pred=y_pred)   

# Print the score
st.write(f"The accuracy of the model is: {100*score:.2f}%") 

# Create the form
with st.form(key="penguin_form"):
    island = st.selectbox(label="Penguin Island", 
                        options=["Biscoe", "Dream", "Torgerson"])

    sex = st.selectbox(label="Sex", 
                    options=["Female", "Male"])

    bill_length_mm = st.number_input(label="Bill Length (mm)",
                                    min_value=0.0)

    bill_depth_mm = st.number_input(label="Bill Depth (mm)", 
                                    min_value=0.0)

    flipper_length_mm = st.number_input(label="Flipper Length (mm)",
                                        min_value=0.0)

    body_mass_g = st.number_input(label="Body Mass (g)",
                                min_value=0.0)

    # Submit the form
    st.form_submit_button()

# Set initial values
sex_female, sex_male = 0, 0
island_Biscoe, island_Dream, island_Torgerson = 0, 0, 0

if island == "Biscoe":
    island_Biscoe = 1
elif island == "Dream":
    island_Dream = 1
elif island == "Torgerson":
    island_Torgerson = 1

if sex == "Female":
    sex_female = 1
elif sex == "Male":
    sex_male = 1

# Make a prediction
sample_array = np.array(object=[bill_length_mm, bill_depth_mm, flipper_length_mm, 
                                body_mass_g, island_Biscoe, island_Dream, 
                                island_Torgerson, sex_female, sex_male])
prediction = rfc.predict(sample_array.reshape(1, -1))
species = uniques[prediction[0]]
st.subheader(body="Predicting your penguin species:")
st.write(f"We predict your pengiun is of the {species} species.")
st.write("""We used a Random Forest Classifier to make this prediction.
         The features used in this prediction are ranked by importance in the plot below.""")

# Display the feature importance plot
st.image(image="./img/feature_importance.png")

st.write("""Below are the histograms for each continuous feature
         separated by species. The vertical lines represent the user input.""")

# Display the distributions
fig, ax = plt.subplots()
ax = sns.displot(data=penguin_df, x="bill_length_mm", hue="species")
plt.axvline(x=bill_length_mm)
plt.title(label="Bill Length by Species")
st.pyplot(fig=ax)

fig, ax = plt.subplots()
ax = sns.displot(data=penguin_df, x="bill_depth_mm", hue="species")
plt.axvline(x=bill_depth_mm)
plt.title(label="Bill Depth by Species")
st.pyplot(fig=ax)

fig, ax = plt.subplots()
ax = sns.displot(data=penguin_df, x="flipper_length_mm",  hue="species")
plt.axvline(x=flipper_length_mm)
plt.title(label="Flipper Length by Species")
st.pyplot(fig=ax)