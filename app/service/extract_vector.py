import cv2
import numpy as np
import insightface
from insightface.app.common import Face
from app.service.fas_model import is_real_face
from app.utils.preprocess import resize_with_padding
from app.utils.model_utils import det_model, rec_model
from concurrent.futures import ThreadPoolExecutor

tracked_faces = []
IOU_THRESHOLD = 0.5
MAX_FRAME_MISS = 3


def iou(bbox1, bbox2):
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    boxBArea = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0
    return interArea / unionArea


def _extract_single(idx, bbox_resize, det_score, kps, img_resize, pad_w, pad_h, scale, orig_w, orig_h):
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
            "bbox_orig": bbox_orig,
            "embedding": embedding.tolist()
        }
    return None


def extract_vector(img_np, det_thresh=0.6):
    """
    Trích xuất danh sách bbox_orig và embedding cho các khuôn mặt mới/tracked trong ảnh, tracking liên tục dựa trên bbox_orig.
    """
    global tracked_faces
    orig_h, orig_w = img_np.shape[:2]
    resized_result = resize_with_padding(img_np, (640, 640), return_info=True)
    if isinstance(resized_result, tuple) and len(resized_result) == 2:
        img_resize, (scale, (pad_w, pad_h)) = resized_result
    else:
        img_resize = resized_result
        scale, pad_w, pad_h = 1, 0, 0

    bboxes, kpss = det_model.detect(img_resize, max_num=0, metric='default')
    if bboxes is None or len(bboxes) == 0:
        tracked_faces.clear()
        return "Không tìm thấy khuôn mặt nào"

    requests = []
    new_tracked = []
    detection_results = []
    # So sánh IOU tất cả dựa trên bbox_orig
    for idx in range(len(bboxes)):
        det_score = float(bboxes[idx, 4])
        if det_score < det_thresh:
            continue
        bbox_resize = bboxes[idx, :4].astype(int).tolist()
        kps = kpss[idx]
        # Chuyển bbox về bbox_orig
        x1 = max((bbox_resize[0] - pad_w) / scale, 0)
        y1 = max((bbox_resize[1] - pad_h) / scale, 0)
        x2 = min((bbox_resize[2] - pad_w) / scale, orig_w)
        y2 = min((bbox_resize[3] - pad_h) / scale, orig_h)
        bbox_orig = [int(x1), int(y1), int(x2), int(y2)]

        matched = False
        for tracked in tracked_faces:
            if iou(bbox_orig, tracked["bbox_orig"]) > IOU_THRESHOLD:
                new_tracked.append({
                    "bbox_orig": bbox_orig,
                    "embedding": tracked.get("embedding"),
                    "frame_miss": 0
                })
                matched = True
                break
        if not matched:
            requests.append((idx, bbox_resize, det_score, kps, img_resize, pad_w, pad_h, scale, orig_w, orig_h))
            detection_results.append({"bbox_orig": bbox_orig})

    with ThreadPoolExecutor() as executor:
        joblist = [executor.submit(_extract_single, *args) for args in requests]
        for idx, job in enumerate(joblist):
            ret = job.result()
            if ret is not None:
                detection_results[idx]["embedding"] = ret["embedding"]
                detection_results[idx]["frame_miss"] = 0
                new_tracked.append(detection_results[idx])

    # Tăng frame_miss cho các track cũ không còn xuất hiện ở frame hiện tại (vẫn phải dựa vào bbox_orig)
    for tracked in tracked_faces:
        is_still = any(iou(tracked["bbox_orig"], t["bbox_orig"]) > IOU_THRESHOLD for t in new_tracked)
        if not is_still and tracked.get("frame_miss", 0) < MAX_FRAME_MISS:
            t_miss = tracked.copy()
            t_miss["frame_miss"] = t_miss.get("frame_miss", 0) + 1
            new_tracked.append(t_miss)

    tracked_faces = [t for t in new_tracked if t.get("frame_miss", 0) < MAX_FRAME_MISS]

    return [{"bbox": t["bbox_orig"], "embedding": t["embedding"]} for t in tracked_faces if
            t.get("embedding") is not None]
