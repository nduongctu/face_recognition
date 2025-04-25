import insightface
from insightface.model_zoo import model_zoo

det_model_path = 'det_10g.onnx'
rec_model_path = 'w600k_r50.onnx'

det_model = model_zoo.get_model(f'app/weights/{det_model_path}')
rec_model = model_zoo.get_model(f'app/weights/{rec_model_path}')

det_model.prepare(ctx_id=0, input_size=(640, 640), det_thres=0.8)
