import logging
import math
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI, Request, status, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from data import FEATURE_NAMES, TARGET_NAMES
from predict import (
    ModelNotAvailableError,
    check_training_range,
    load_metrics,
    load_model,
    model_is_available,
    predict_data,
    predict_proba_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("iris_api")

# Hard limits: anything outside (0, 10] cm is physically implausible for an
# iris flower, so it is rejected with 422 before it ever reaches the model.
FEATURE_MIN_EXCLUSIVE = 0.0
FEATURE_MAX_INCLUSIVE = 10.0
MAX_BATCH_SIZE = 100


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

    def range_warnings(self) -> list[str]:
        """Soft check: valid values that lie outside the training data range."""
        return check_training_range(
            {
                "sepal_length": self.sepal_length,
                "sepal_width": self.sepal_width,
                "petal_length": self.petal_length,
                "petal_width": self.petal_width,
            }
        )


class IrisResponse(BaseModel):
    response: int
    species: str = ""
    warnings: list[str] = []


class IrisProbaResponse(BaseModel):
    response: int
    species: str
    probabilities: dict[str, float]
    warnings: list[str] = []


class BatchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "samples": [
                        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                        {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3},
                    ]
                }
            ]
        },
    )

    samples: list[IrisData] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class BatchItem(BaseModel):
    index: int
    response: int
    species: str
    warnings: list[str] = []


class BatchResponse(BaseModel):
    count: int
    predictions: list[BatchItem]


class ModelInfo(BaseModel):
    # Older Pydantic 2.x releases warn about fields starting with "model_".
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    max_depth: int | None
    classes: list[str]
    feature_names: list[str]
    metrics_available: bool
    metrics: dict | None = None


@contextmanager
def model_errors_as_http():
    """Map model failures to HTTP errors: missing model -> 503, anything else -> 500."""
    try:
        yield
    except ModelNotAvailableError as e:
        logger.warning("Prediction requested but model is unavailable: %s", e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected prediction error")
        raise HTTPException(status_code=500, detail=str(e))


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
    with model_errors_as_http():
        features = [iris_features.to_features()]

        prediction = predict_data(features)
        class_id = int(prediction[0])
        warnings = iris_features.range_warnings()
        logger.info("Prediction %s -> %s (warnings=%d)", features[0], TARGET_NAMES[class_id], len(warnings))
        return IrisResponse(response=class_id, species=TARGET_NAMES[class_id], warnings=warnings)


@app.post("/predict/proba", response_model=IrisProbaResponse)
async def predict_iris_proba(iris_features: IrisData):
    """
    Same input as /predict, plus the probability of each species (they sum to 1).
    `response` is the most likely class, so it always agrees with /predict.
    """
    with model_errors_as_http():
        proba = predict_proba_data([iris_features.to_features()])[0]
        class_id = int(proba.argmax())
        logger.info("Proba prediction -> %s (p=%.3f)", TARGET_NAMES[class_id], proba[class_id])
        return IrisProbaResponse(
            response=class_id,
            species=TARGET_NAMES[class_id],
            probabilities={name: float(p) for name, p in zip(TARGET_NAMES, proba)},
            warnings=iris_features.range_warnings(),
        )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_iris_batch(batch: BatchRequest):
    """
    Predict 1 to 100 samples in one call. If any sample is invalid the whole
    request is rejected with 422, and the error `loc` points at that sample.
    """
    with model_errors_as_http():
        predictions = predict_data([sample.to_features() for sample in batch.samples])
        items = [
            BatchItem(
                index=i,
                response=int(class_id),
                species=TARGET_NAMES[int(class_id)],
                warnings=sample.range_warnings(),
            )
            for i, (sample, class_id) in enumerate(zip(batch.samples, predictions))
        ]
        logger.info("Batch prediction for %d samples", len(items))
        return BatchResponse(count=len(items), predictions=items)


@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    """
    Describe the served model. Training metrics come from model/metrics.json;
    if that file is missing, `metrics` is null and `metrics_available` is false.
    """
    with model_errors_as_http():
        model = load_model()
        metrics = load_metrics()
        return ModelInfo(
            model_type=type(model).__name__,
            max_depth=model.get_params().get("max_depth"),
            classes=[TARGET_NAMES[int(c)] for c in model.classes_],
            feature_names=FEATURE_NAMES,
            metrics_available=metrics is not None,
            metrics=metrics,
        )
