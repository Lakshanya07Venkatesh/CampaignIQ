import pandas as pd
import joblib


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load(
    "models/best_conversion_model.pkl"
)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_conversion_rate(
    budget,
    duration,
    platform,
    content_type,
    target_age,
    target_gender,
    region,
    month
):

    # Calculate quarter from month
    quarter = (month - 1) // 3 + 1

    campaign = pd.DataFrame([{
        "Budget": budget,
        "Duration": duration,
        "Platform": platform,
        "Content_Type": content_type,
        "Target_Age": target_age,
        "Target_Gender": target_gender,
        "Region": region,
        "Month": month,
        "Quarter": quarter
    }])

    prediction = model.predict(campaign)[0]

    return prediction


# ==========================================
# TEST PREDICTION
# ==========================================

prediction = predict_conversion_rate(
    budget=6000,
    duration=15,
    platform="Instagram",
    content_type="Video",
    target_age="18-24",
    target_gender="Female",
    region="Asia",
    month=9
)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n================================")
print("CAMPAIGN PERFORMANCE PREDICTION")
print("================================")

print(f"Predicted Conversion Rate: {prediction:.2%}")