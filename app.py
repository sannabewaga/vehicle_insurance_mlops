from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from uvicorn import run as app_run

from typing import Optional

from src.datascience.constants import APP_HOST, APP_PORT
from src.datascience.pipeline.training_pipeline import TrainingPipeline
from src.datascience.pipeline.prediction_pipeline import VehicleData, VehiclePredictor
application = FastAPI()
app = application
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Gender: Optional[int] = None
        self.Age: Optional[int] = None
        self.Driving_License: Optional[int] = None
        self.Region_Code: Optional[float] = None
        self.Previously_Insured: Optional[int] = None
        self.Annual_Premium: Optional[float] = None
        self.Policy_Sales_Channel: Optional[float] = None
        self.Vintage: Optional[int] = None
        self.Vehicle_Age_lt_1_Year: Optional[int] = None
        self.Vehicle_Age_gt_2_Years: Optional[int] = None
        self.Vehicle_Damage_Yes: Optional[int] = None

    async def get_vehicle_data(self):
        form = await self.request.form()
        self.Gender = form.get("Gender")
        self.Age = form.get("Age")
        self.Driving_License = form.get("Driving_License")
        self.Region_Code = form.get("Region_Code")
        self.Previously_Insured = form.get("Previously_Insured")
        self.Annual_Premium = form.get("Annual_Premium")
        self.Policy_Sales_Channel = form.get("Policy_Sales_Channel")
        self.Vintage = form.get("Vintage")
        self.Vehicle_Age_lt_1_Year = form.get("Vehicle_Age_lt_1_Year")
        self.Vehicle_Age_gt_2_Years = form.get("Vehicle_Age_gt_2_Years")
        self.Vehicle_Damage_Yes = form.get("Vehicle_Damage_Yes")

@app.get("/", tags=["form"])
async def index(request: Request):
    return templates.TemplateResponse("vehicledata.html", {"request": request, "context": "Rendering"})

@app.get("/train")
async def train_model():
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        return Response("Training successful!")
    except Exception as e:
        return Response(f"Training failed: {e}")

@app.post("/")
async def predict(request: Request):
    try:
        form = DataForm(request)
        await form.get_vehicle_data()

        vehicle_data = VehicleData(
            Gender=form.Gender,
            Age=form.Age,
            Driving_License=form.Driving_License,
            Region_Code=form.Region_Code,
            Previously_Insured=form.Previously_Insured,
            Annual_Premium=form.Annual_Premium,
            Policy_Sales_Channel=form.Policy_Sales_Channel,
            Vintage=form.Vintage,
            Vehicle_Age_lt_1_Year=form.Vehicle_Age_lt_1_Year,
            Vehicle_Age_gt_2_Years=form.Vehicle_Age_gt_2_Years,
            Vehicle_Damage_Yes=form.Vehicle_Damage_Yes
        )

        df = vehicle_data.get_vehicle_input_dataframe()

        predictor = VehiclePredictor()
        prediction = predictor.predict(df)

        response_text = "Response-Yes" if prediction == "1" else "Response-No"

        return templates.TemplateResponse("vehicledata.html", {
            "request": request,
            "context": response_text
        })

    except Exception as e:
        return templates.TemplateResponse("vehicledata.html", {
            "request": request,
            "context": f"Prediction Failed: {e}"
        })

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
