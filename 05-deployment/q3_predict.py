import pickle

# Customer lead supplied by the homework
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

# Load the already-trained model
with open("pipeline.bin", "rb") as model_file:
    pipeline = pickle.load(model_file)

# Predict the probability of conversion
probability = pipeline.predict_proba([lead])[0, 1]

print(f"Conversion probability: {probability:.3f}")