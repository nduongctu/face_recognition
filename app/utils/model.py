import insightface
from insightface.model_zoo import model_zoo

DET_MODEL_PATH = "/root/.insightface/models/det_10g.onnx"
REC_MODEL_PATH = "/root/.insightface/models/w600k_r50.onnx"
MASK_MODEL_PATH = "/root/.insightface/models/mask_detector112.onnx"
det_model = model_zoo.get_model(DET_MODEL_PATH)
rec_model = model_zoo.get_model(REC_MODEL_PATH)
#mask_model = model_zoo.get_model(MASK_MODEL_PATH)

det_model.prepare(ctx_id=0, input_size=(640, 640))
rec_model.prepare(ctx_id=0)
#mask_model.prepare(ctx_id=0)