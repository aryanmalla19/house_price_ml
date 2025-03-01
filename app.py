import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------------------------------------
# Step 1: Load data.csv to compute encoding mappings
# -----------------------------------------------------------
try:
    df_data = pd.read_csv("data.csv")
    # Assume df_data has columns: LOCATION, Price, etc.
    # And that you have already split LOCATION into 'District' and 'Street'
    # For this example, we assume 'District', 'Street', and 'FACING' exist in the raw data.
    
    # Compute target encoding mappings: use mean Price for each District and Street.
    district_mean = df_data.groupby("District")["Price"].mean().to_dict()
    street_mean   = df_data.groupby("Street")["Price"].mean().to_dict()
    
    # For FACING, use your custom ranking order
    custom_order = ['missing', 'north', 'north east', 'east', 'south east', 
                    'south', 'south west', 'west', 'north west']
    order_map = {direction: idx for idx, direction in enumerate(custom_order)}
    
    # Get sorted unique options for the selectboxes:
    district_options = sorted(district_mean.keys())
    street_options   = sorted(street_mean.keys())
    facing_options   = custom_order  # Use custom_order as available options
    
except Exception as e:
    st.error("Error loading data.csv for encoding mappings: " + str(e))
    st.stop()

# -----------------------------------------------------------
# Step 2: Load the individual models from pickle files
# -----------------------------------------------------------
try:
    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("cat_model.pkl", "rb") as f:
        cat_model = pickle.load(f)
    with open("lr_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open("xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("ridge_model.pkl", "rb") as f:
        ridge_model = pickle.load(f)
    with open("lgb_model.pkl", "rb") as f:
        lgb_model = pickle.load(f)
except Exception as e:
    st.error("Error loading model pickle files: " + str(e))
    st.stop()

# -----------------------------------------------------------
# Define feature names (must match what was used during training)
# -----------------------------------------------------------
feature_names = [
    "Land Area (Aana)", 
    "Road Access (Feet)", 
    "FACING",      # Encoded using order_map
    "FLOOR", 
    "BEDROOM", 
    "BATHROOM", 
    "Street",    # Target encoded using street_mean
    "District",  # Target encoded using district_mean
    "House Age"
]

# -----------------------------------------------------------
# Define the ensemble prediction function
# -----------------------------------------------------------
def ensemble_predict(X):
    # Ensure X is a DataFrame with the correct columns
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)
    X_lgb = X.copy()  # Avoid slicing warnings for LightGBM
    # Average predictions from six models (assumes models were trained on log-price)
    log_preds = (
        rf_model.predict(X) +
        cat_model.predict(X) +
        lr_model.predict(X) +
        xgb_model.predict(X) +
        ridge_model.predict(X) +
        lgb_model.predict(X_lgb)
    ) / 6.0
    # Exponentiate to convert log-price back to original scale
    return np.exp(log_preds)

# -----------------------------------------------------------
# Streamlit User Interface
# -----------------------------------------------------------
st.title("House Price Prediction - Averaging Ensemble")
st.write("Enter the details of the house below:")

with st.form("prediction_form"):
    # Numeric inputs
    land_area   = st.number_input("Land Area (Aana)", min_value=0.0, value=4.0)
    road_access = st.number_input("Road Access (Feet)", min_value=0.0, value=12.0)
    floor       = st.number_input("FLOOR", min_value=0.0, value=2.5)
    bedroom     = st.number_input("BEDROOM", min_value=0, value=5)
    bathroom    = st.number_input("BATHROOM", min_value=0, value=4)
    house_age   = st.number_input("House Age", min_value=0, value=5)
    
    # Categorical inputs using selectbox:
    # For FACING, we apply label encoding using our custom order:
    facing_input = st.selectbox("FACING", facing_options)
    # For District and Street, use the unique options from our mapping
    district_input = st.selectbox("District", district_options)
    street_input   = st.selectbox("Street", street_options)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # -----------------------------------------------------------
    # Transform the categorical inputs:
    # For FACING: Lowercase, fill missing, then map using order_map.
    facing_value = facing_input.lower() if facing_input else "missing"
    facing_encoded = order_map.get(facing_value, len(custom_order))
    
    # For District and Street: Apply target encoding using the mappings from data.csv.
    district_encoded = district_mean.get(district_input, np.nan)
    street_encoded   = street_mean.get(street_input, np.nan)
    
    # Ensure we have valid encodings:
    if pd.isna(district_encoded) or pd.isna(street_encoded):
        st.error("Error: Selected District/Street not found in training data mappings.")
        st.stop()
    
    # -----------------------------------------------------------
    # Build the input DataFrame using the encoded values.
    # -----------------------------------------------------------
    input_data = pd.DataFrame({
        "Land Area (Aana)": [land_area],
        "Road Access (Feet)": [road_access],
        "FACING": [facing_encoded],
        "FLOOR": [floor],
        "BEDROOM": [bedroom],
        "BATHROOM": [bathroom],
        "Street": [street_encoded],
        "District": [district_encoded],
        "House Age": [house_age]
    })
    
    # Get the ensemble prediction (in original scale)
    predicted_price = ensemble_predict(input_data)
    
    st.success(f"Predicted House Price (Original Scale): {predicted_price[0]:.2f}")
