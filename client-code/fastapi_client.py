import uvicorn
from fastapi import FastAPI, HTTPException
import grpc
import os
from starlette.responses import RedirectResponse
from tensorflow_serving.apis import predict_pb2
import tensorflow as tf

app = FastAPI()


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/prediction")
async def predict_api():
    request = predict_pb2.PredictRequest()
    request.model_spec.name = os.getenv("MODEL_NAME", None)
    request.model_spec.signature_name = (
        tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
    request.inputs[tf.saved_model.signature_constants.PREDICT_INPUTS].CopyFrom(
        tf.contrib.util.make_tensor_proto([[1]], shape=[1, 1])
    )
    try:
        result = self.stub.Predict(request, 10)
        return {"result": result.outputs["outputs"].int_val[0]}
    except grpc.RpcError as e:
        raise HTTPException(status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
