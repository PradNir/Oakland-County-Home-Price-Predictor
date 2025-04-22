import streamlit as st
import numpy as np
import pickle

# Load trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set page config and title
st.set_page_config(page_title="Real Estate PPSF Predictor", page_icon="🏠")
st.title("🏠 Real Estate Price per SqFt Predictor")
st.markdown("Estimate the **Price per Square Foot** using smart features and a trained model.")

# --------------------------
# USER INPUTS
# --------------------------

beds = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
baths = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, value=2.0)
sqft = st.number_input("Total Living Area (Square Feet)", min_value=300, max_value=10000, value=2000)
lot_size = st.number_input("Lot Size (Square Feet)", min_value=500, max_value=30000, value=5000)
hoa = st.number_input("HOA per Month ($)", min_value=0, max_value=1000, value=0)
home_age = st.slider("Home Age (Years)", 0, 150, 20)
sale_qtr = st.selectbox("Quarter Sold", ["Q1 (Jan–Mar)", "Q2 (Apr–Jun)", "Q3 (Jul–Sep)", "Q4 (Oct–Dec)"])
sale_qtr_num = ["Q1 (Jan–Mar)", "Q2 (Apr–Jun)", "Q3 (Jul–Sep)", "Q4 (Oct–Dec)"].index(sale_qtr) + 1

# --------------------------
# CITY Dropdown (Frequency Encoded)
# --------------------------
city_freq_map = {
    "Troy": 1200,
    "Royal Oak": 950,
    "Rochester Hills": 875,
    "Farmington Hills": 790,
    "Southfield": 670,
    "Oak Park": 540,
    "Auburn Hills": 480,
    "Pontiac": 400
}

city_to_zip_map = {
    "Troy": {"48083": 850, "48084": 780},
    "Royal Oak": {"48067": 720},
    "Rochester Hills": {"48307": 900, "48309": 870},
    "Auburn Hills": {"48326": 610},
    "Pontiac": {"48341": 560},
    "Southfield": {"48076": 660},
    "Farmington Hills": {"48336": 540},
    "Oak Park": {"48237": 430}
}

zip_to_avg_ppsf = {
    "48083": 195.0,
    "48084": 220.0,
    "48307": 210.0,
    "48309": 205.0,
    "48067": 230.0,
    "48341": 180.0,
    "48326": 190.0,
    "48076": 200.0,
    "48336": 185.0,
    "48237": 175.0
}

# Select City
city = st.selectbox("Select City", list(city_freq_map.keys()))
city_encoded = city_freq_map[city]

# Select ZIP based on City
available_zips = list(city_to_zip_map[city].keys())
zip_code = st.selectbox("Select ZIP Code", available_zips)
zip_encoded = city_to_zip_map[city][zip_code]

# Auto-fill Avg PPSF using ZIP
avg_ppsf_knn = zip_to_avg_ppsf.get(zip_code, 200.0)
st.markdown(f"💡 Estimated Avg PPSF Nearby (based on ZIP): **${avg_ppsf_knn:.2f}**")

# --------------------------
# Property Type Dropdown (Frequency Encoded)
# --------------------------
property_type_freq_map = {
    "Single Family Residential": 15000,
    "Townhouse": 2400,
    "Condominium": 3200,
    "Duplex": 800,
    "Mobile/Manufactured Home": 500
}
property_type = st.selectbox("Select Property Type", list(property_type_freq_map.keys()))
property_type_encoded = property_type_freq_map[property_type]

# --------------------------
# DERIVED FEATURES
# --------------------------

bed_bath_product = beds * baths
lot_to_home_ratio = lot_size / sqft
bed_to_bath_ratio = beds / baths
has_hoa = 1 if hoa > 0 else 0
is_luxury = 1 if sqft >= 3500 and baths >= 3.5 and hoa >= 250 else 0

# Final input vector (matches your model's 16 features)
features = np.array([[
    beds, baths, sqft, lot_size, hoa,
    bed_bath_product, home_age, sale_qtr_num, has_hoa,
    city_encoded, zip_encoded, property_type_encoded,
    lot_to_home_ratio, bed_to_bath_ratio, is_luxury,
    avg_ppsf_knn
]])

# --------------------------
# PREDICT
# --------------------------
if st.button("💰 Predict Price per SqFt"):
    y_log_pred = model.predict(features)[0]
    ppsf = np.expm1(y_log_pred)
    st.success(f"Estimated Price per SqFt: **${ppsf:.2f}**")
    st.markdown(f"🏡 Estimated Total Price: **${ppsf * sqft:,.2f}**")
