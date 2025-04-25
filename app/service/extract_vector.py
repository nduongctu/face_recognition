import cv2
import numpy as np
import insightface
from insightface.app.common import Face
from app.service.fas_model import is_real_face
from app.utils.preprocess import resize_with_padding
from app.utils.model_utils import det_model, rec_model
from concurrent.futures import ThreadPoolExecutor


def _extract_single(idx, bbox_resize, det_score, kps, img_resize, pad_w, pad_h, scale, orig_w, orig_h):
    # Tạo bbox gốc
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
            "bbox": bbox_orig,  # bbox gốc trên ảnh ban đầu
            "embedding": embedding.tolist()
        }
    return None


def extract_vector(img_np, det_thresh=0.6):
    """
    Trích xuất danh sách bbox và embedding cho tất cả khuôn mặt trong ảnh, có hỗ trợ đa luồng.
    Trả về: list dict, mỗi dict gồm bbox (ảnh gốc), bbox_resize, embedding.
    """
    results = []
    orig_h, orig_w = img_np.shape[:2]
    resized_result = resize_with_padding(img_np, (640, 640), return_info=True)
    if isinstance(resized_result, tuple) and len(resized_result) == 2:
        img_resize, (scale, (pad_w, pad_h)) = resized_result
    else:
        # Giả sử ảnh đã resize, ko lấy padding
        img_resize = resized_result
        scale, pad_w, pad_h = 1, 0, 0

    bboxes, kpss = det_model.detect(img_resize, max_num=0, metric='default')
    if bboxes is None or len(bboxes) == 0:
        return "Không tìm thấy khuôn mặt nào"

    requests = []
    for idx in range(len(bboxes)):
        det_score = float(bboxes[idx, 4])
        if det_score < det_thresh:
            continue
        bbox_resize = bboxes[idx, :4].astype(int).tolist()
        kps = kpss[idx]
        requests.append((idx, bbox_resize, det_score, kps, img_resize, pad_w, pad_h, scale, orig_w, orig_h))

    # Dùng ThreadPool xử lý embedding song song
    with ThreadPoolExecutor() as executor:
        joblist = [executor.submit(_extract_single, *args) for args in requests]
        for job in joblist:
            ret = job.result()
            if ret is not None:
                results.append(ret)
    return results
