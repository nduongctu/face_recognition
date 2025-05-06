import cv2
import numpy as np
import insightface
from insightface.app.common import Face
from app.utils.align_face import align_face
from app.utils.model import det_model, rec_model
from app.utils.preprocess import resize_with_padding, expand_bbox_px


def extract_vector(img_np, det_thresh=0.6):
    """
    Trích xuất bbox và embedding cho 1 khuôn mặt có score cao nhất trong ảnh (nếu có).
    Trả về: dict gồm bbox (trên ảnh gốc), embedding (list); hoặc None nếu không có khuôn mặt nào.
    """
    orig_h, orig_w = img_np.shape[:2]
    resized_result = resize_with_padding(img_np, (640, 640), return_info=True)
    if isinstance(resized_result, tuple) and len(resized_result) == 2:
        img_resize, (scale, (pad_w, pad_h)) = resized_result
    else:
        img_resize = resized_result
        scale, pad_w, pad_h = 1, 0, 0

    bboxes, kpss = det_model.detect(img_resize, max_num=0, metric='default')
    if bboxes is None or len(bboxes) == 0:
        return None

    scores = bboxes[:, 4]
    mask = scores >= det_thresh
    if not np.any(mask):
        return None

    idx_max = np.argmax(scores * mask)
    det_score = float(bboxes[idx_max, 4])
    bbox_resize = bboxes[idx_max, :4].astype(int).tolist()
    kps = kpss[idx_max]

    bbox_resize = expand_bbox_px(bbox_resize, 4, img_resize.shape[1], img_resize.shape[0])

    x1, y1, x2, y2 = bbox_resize
    x1 = max((x1 - pad_w) / scale, 0)
    y1 = max((y1 - pad_h) / scale, 0)
    x2 = min((x2 - pad_w) / scale, orig_w)
    y2 = min((y2 - pad_h) / scale, orig_h)
    bbox_orig = [int(x1), int(y1), int(x2), int(y2)]

    face = Face(bbox=bbox_resize, kps=kps, det_score=det_score)
    rec_model.get(img_resize, face)
    embedding = face.normed_embedding
    if embedding is not None:
        return {
            "bbox": bbox_orig,
            "embedding": embedding.tolist(),
        }
    return None


def extract_vector_align(img_np, det_thresh=0.6):
    """
    Trích xuất bbox có align ảnh và embedding cho 1 khuôn mặt có score cao nhất trong ảnh (nếu có).
    Trả về: dict gồm bbox (trên ảnh gốc), embedding (list); hoặc None nếu không có khuôn mặt nào.
    """
    orig_h, orig_w = img_np.shape[:2]
    resized_result = resize_with_padding(img_np, (640, 640), return_info=True)
    if isinstance(resized_result, tuple) and len(resized_result) == 2:
        img_resize, (scale, (pad_w, pad_h)) = resized_result
    else:
        img_resize = resized_result
        scale, pad_w, pad_h = 1, 0, 0

    bboxes, kpss = det_model.detect(img_resize, max_num=0, metric='default')
    if bboxes is None or len(bboxes) == 0:
        return None

    scores = bboxes[:, 4]
    mask = scores >= det_thresh
    if not np.any(mask):
        return None

    idx_max = np.argmax(scores * mask)
    det_score = float(bboxes[idx_max, 4])
    bbox_resize = bboxes[idx_max, :4].astype(int).tolist()
    kps = kpss[idx_max]

    bbox_resize = expand_bbox_px(bbox_resize, 4, img_resize.shape[1], img_resize.shape[0])
    x1, y1, x2, y2 = bbox_resize
    x1 = max((x1 - pad_w) / scale, 0)
    y1 = max((y1 - pad_h) / scale, 0)
    x2 = min((x2 - pad_w) / scale, orig_w)
    y2 = min((y2 - pad_h) / scale, orig_h)
    bbox_orig = [int(x1), int(y1), int(x2), int(y2)]

    kps_orig = []
    for (x, y) in kps:
        x_orig = (x - pad_w) / scale
        y_orig = (y - pad_h) / scale
        kps_orig.append([x_orig, y_orig])
    kps_orig = np.array(kps_orig, dtype=np.float32)

    aligned_face_img = align_face(img_np, kps_orig)

    face = Face(bbox=bbox_orig, kps=kps_orig, det_score=det_score)
    rec_model.get(aligned_face_img, face)

    embedding = face.normed_embedding
    if embedding is not None:
        return {
            "bbox": bbox_orig,
            "embedding": embedding.tolist(),
        }
    return None
