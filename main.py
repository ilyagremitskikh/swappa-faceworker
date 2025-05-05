import uvicorn
from fastapi import FastAPI, status
from fastapi.responses import Response
from faststream.nats.fastapi import NatsRouter, Logger
from face_operations.models import FaceOperationJobFile, FaceSwapJobRequest
from faststream.nats import NatsMessage
from face_operations.service import FaceOperationsService

router = NatsRouter("nats://test-mq.orb.local:4222")


@router.post("/start-face-swap")
async def start_face_swap(request: FaceSwapJobRequest):
    await router.broker.publish(request, "gpu-tasks")
    return {"message": "Face swap job started"}


@router.post("/face-detection")
async def create_face_detection_job(request: FaceOperationJobFile):
    response: NatsMessage = await router.broker.request(request, "cpu-tasks", timeout=30)
    return response.decoded_body
    

@router.subscriber("cpu-tasks")
async def cpu_tasks_handler(request: FaceOperationJobFile, logger: Logger):
    logger.info(request)
    try:
        result = FaceOperationsService.run_face_verification(request)
        logger.info(result)
        return result
    except Exception as e:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=str(e))


@router.subscriber("gpu-tasks")
async def gpu_tasks_handler(request: FaceSwapJobRequest, logger: Logger):
    logger.info(request)
    try:
        result = FaceOperationsService.run_face_swap(request)
        logger.info(result)
        return result
    except Exception as e:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=str(e))

app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)