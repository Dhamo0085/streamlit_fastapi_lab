import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from data import TARGET_NAMES
from predict import ModelNotAvailableError, load_model, model_is_available, predict_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("iris_api")

# Hard limits: anything outside (0, 10] cm is physically implausible for an
# iris flower, so it is rejected with 422 before it ever reaches the model.
FEATURE_MIN_EXCLUSIVE = 0.0
FEATURE_MAX_INCLUSIVE = 10.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup so the first request is not slow."""
    try:
        load_model()
        logger.info("Model loaded at startup")
    except ModelNotAvailableError as exc:
        # Keep the API up: GET / reports model_loaded=false and /predict returns 503.
        logger.warning("Model not available at startup: %s", exc)
    yield


app = FastAPI(
    title="Iris Classifier API",
    description=(
        "Serves a scikit-learn DecisionTreeClassifier trained on the Iris dataset. "
        "Adapted from the MLOps course FastAPI lab with input validation, "
        "proper status codes and richer responses."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


def _json_safe(value):
    """Replace NaN/Infinity (not valid JSON) with strings, recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # FastAPI's default handler echoes the rejected input back. For NaN or
    # Infinity that echo is not valid JSON and the 422 turned into a 500,
    # so the offending values are converted to strings first.
    errors = []
    for error in exc.errors():
        if isinstance(error.get("ctx", {}).get("error"), Exception):
            # Exceptions would otherwise serialise as an empty {}.
            error = {**error, "ctx": {**error["ctx"], "error": str(error["ctx"]["error"])}}
        errors.append(error)
    errors = _json_safe(jsonable_encoder(errors))
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": errors})


def _feature_field(description: str):
    return Field(
        ...,
        gt=FEATURE_MIN_EXCLUSIVE,
        le=FEATURE_MAX_INCLUSIVE,
        allow_inf_nan=False,
        description=f"{description} in cm (must be > 0 and <= 10)",
    )


class IrisData(BaseModel):
    """
    Pydantic BaseModel representing iris flower measurements.

    Attributes:
        petal_length (float): The length of the iris flower's petal in centimeters.
        sepal_length (float): The length of the iris flower's sepal in centimeters.
        petal_width (float): The width of the iris flower's petal in centimeters.
        sepal_width (float): The width of the iris flower's sepal in centimeters.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
            ]
        },
    )

    petal_length: float = _feature_field("Petal length")
    sepal_length: float = _feature_field("Sepal length")
    petal_width: float = _feature_field("Petal width")
    sepal_width: float = _feature_field("Sepal width")

    @field_validator("*", mode="before")
    @classmethod
    def reject_booleans(cls, value):
        # Pydantic would silently turn true/false into 1.0/0.0; a boolean is
        # never a valid measurement, so treat it as a type error instead.
        if isinstance(value, bool):
            raise ValueError("must be a number, not a boolean")
        return value

    def to_features(self) -> list[float]:
        """Feature vector in the column order the model was trained on."""
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class IrisResponse(BaseModel):
    response: int
    species: str = ""
    warnings: list[str] = []


"""Modern web apps use a technique named routing. This helps the user remember the URLs.
For instance, instead of having /booking.php they see /booking/. Instead of /account.asp?id=1234/
they’d see /account/1234/."""

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Concurrent (multiple tasks can run simultaneously)"""
    return {"status": "healthy", "model_loaded": model_is_available()}

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    """
    Predict the iris flower species based on provided features.
    This endpoint accepts iris flower measurements and returns the predicted species class.
    Args:
        iris_features (IrisData): An IrisData object containing:
            - sepal_length (float): Length of the sepal in cm
            - sepal_width (float): Width of the sepal in cm
            - petal_length (float): Length of the petal in cm
            - petal_width (float): Width of the petal in cm
    Returns:
        IrisResponse: A response object containing:
            - response (int): The predicted iris species class (0, 1, or 2)
            - species (str): The species name for that class
            - warnings (list[str]): Notes about unusual (but accepted) inputs
    Raises:
        HTTPException: 422 for invalid input, 503 if the model file is missing,
            500 if prediction fails unexpectedly.
    Example:
        POST /predict
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        Response:
        {
            "response": 0,
            "species": "setosa",
            "warnings": []
        }
    """
    try:
        features = [iris_features.to_features()]

        prediction = predict_data(features)
        class_id = int(prediction[0])
        logger.info("Prediction %s -> %s", features[0], TARGET_NAMES[class_id])
        return IrisResponse(response=class_id, species=TARGET_NAMES[class_id], warnings=[])

    except ModelNotAvailableError as e:
        logger.warning("Prediction requested but model is unavailable: %s", e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected prediction error")
        raise HTTPException(status_code=500, detail=str(e))
