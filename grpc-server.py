import logging

from concurrent import futures
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import grpc
import requests
from PIL import Image
import inference_pb2
import inference_pb2_grpc


class InstanceDetector(inference_pb2_grpc.InstanceDetectorServicer):
    def __init__(self):
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.9)
        self.model.eval()
        self.preprocess = self.weights.transforms()

    def Predict(self, request, context):
        img = Image.open(requests.get(request.url, stream=True).raw)
        batch = [self.preprocess(img)]
        prediction = self.model(batch)[0]
        labels = [self.weights.meta["categories"][i] for i in prediction["labels"]]
        return inference_pb2.InstanceDetectorOutput(objects=labels)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InstanceDetectorServicer_to_server(InstanceDetector(), server)
    server.add_insecure_port('[::]:9090')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    print("start serving...")
    serve()