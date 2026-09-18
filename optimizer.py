import pandas as pd
import joblib
import itertools


# Load trained model
model = joblib.load("models/best_conversion_model.pkl")


def optimize_campaign(budget, duration, month):

    platforms = [
        "Instagram",
        "LinkedIn",
        "Facebook",
        "YouTube",
        "Google"
    ]

    content_types = [
        "Carousel",
        "Text",
        "Video",
        "Image"
    ]

    target_ages = [
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55+"
    ]

    target_genders = [
        "Male",
        "Female",
        "Other"
    ]

    regions = [
        "Asia",
        "Africa",
        "Europe",
        "North America",
        "South America"
    ]

    # Calculate quarter from month
    quarter = (month - 1) // 3 + 1

    # Generate every possible combination
    combinations = list(
        itertools.product(
            platforms,
            content_types,
            target_ages,
            target_genders,
            regions
        )
    )

    # Create one dataframe containing all 1,500 combinations
    campaigns = pd.DataFrame(
        combinations,
        columns=[
            "Platform",
            "Content_Type",
            "Target_Age",
            "Target_Gender",
            "Region"
        ]
    )

    # Add fixed campaign values
    campaigns["Budget"] = budget
    campaigns["Duration"] = duration
    campaigns["Month"] = month
    campaigns["Quarter"] = quarter

    # Keep exact order expected by the trained model
    campaigns = campaigns[
        [
            "Budget",
            "Duration",
            "Platform",
            "Content_Type",
            "Target_Age",
            "Target_Gender",
            "Region",
            "Month",
            "Quarter"
        ]
    ]

    # Predict ALL configurations at once
    predictions = model.predict(campaigns)

    # Add predictions
    results = campaigns.copy()

    results["Predicted Conversion Rate"] = predictions

    # Sort highest prediction first
    results = results.sort_values(
        by="Predicted Conversion Rate",
        ascending=False
    ).reset_index(drop=True)

    # Get best configuration
    best_row = results.iloc[0]

    best_config = {
        "Platform": best_row["Platform"],
        "Content Type": best_row["Content_Type"],
        "Target Age": best_row["Target_Age"],
        "Target Gender": best_row["Target_Gender"],
        "Region": best_row["Region"],
        "Predicted Conversion Rate": best_row[
            "Predicted Conversion Rate"
        ]
    }

    return best_config, results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    best_config, results = optimize_campaign(
        budget=6000,
        duration=15,
        month=9
    )

    print("\n======================================")
    print("OPTIMIZED CAMPAIGN RECOMMENDATION")
    print("======================================")

    print("Platform:", best_config["Platform"])
    print("Content Type:", best_config["Content Type"])
    print("Target Age:", best_config["Target Age"])
    print("Target Gender:", best_config["Target Gender"])
    print("Region:", best_config["Region"])

    print(
        f"Predicted Conversion Rate: "
        f"{best_config['Predicted Conversion Rate']:.2%}"
    )

    print(
        "Total combinations tested:",
        len(results)
    )