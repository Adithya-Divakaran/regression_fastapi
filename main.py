from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware # Import CORS Middleware

# Load the trained regression model
with open('Ridge Regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI(title="Average Order Value Predictor")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Input model for the API
class CustomerInfo(BaseModel):
    age: float
    gender: str  # Expecting 'Male' or 'Female'

@app.get("/")
def read_root():
    return {"message": "API is live. Use /predict to get order value prediction."}

# Gender encoder
def encode_gender(gender: str) -> int:
    return 1 if gender.lower() == "male" else 0

# POST endpoint
@app.post("/predict")
def predict(data: CustomerInfo):
    try:
        gender_encoded = encode_gender(data.gender)
        input_array = np.array([[data.age, gender_encoded]])
        prediction = model.predict(input_array)[0]
        return {"predicted_avg_order_value": prediction}
    except Exception as e:
        return {"error": str(e)}