import insightface

model_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model_app.prepare(ctx_id=0, det_size=(640, 640))
