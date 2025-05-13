from insightface.app.common import Face
from face_recognize.align_face import align_face
from face_recognize.model import det_model, rec_model
from concurrent.futures import ThreadPoolExecutor
from face_recognize.preprocess import resize_with_padding, expand_bbox_px

tracked_faces = []
IOU_THRESHOLD = 0.3
MAX_FRAME_MISS = 15


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

    bbox_resize_expanded = expand_bbox_px(bbox_resize, 4, img_resize.shape[1], img_resize.shape[0])

    aligned_face_img = align_face(img_resize, kps)
    face = Face(bbox=bbox_resize_expanded, kps=kps, det_score=det_score)
    rec_model.get(aligned_face_img, face)
    embedding = face.normed_embedding
    if embedding is not None:
        return {
            "bbox_orig": bbox_orig,
            "embedding": embedding.tolist()
        }
    return None


def extract_vector(img_np, det_thresh=0.6):
    global tracked_faces
    orig_h, orig_w = img_np.shape[:2]
    resized_result = resize_with_padding(img_np, (640, 640), return_info=True)
    if isinstance(resized_result, tuple) and len(resized_result) == 2:
        img_resize, (scale, (pad_w, pad_h)) = resized_result
    else:
        img_resize = resized_result
        scale, pad_w, pad_h = 1, 0, 0

    bboxes, kpss = det_model.detect(img_resize, max_num=0, metric='default')

    # Lọc khuôn mặt cần trả về: trả về tất cả khuôn mặt được phát hiện
    faces_to_return = []

    if bboxes is None or len(bboxes) == 0:
        # Không thấy mặt nào => chỉ tăng frame_miss
        for tracked in tracked_faces:
            tracked["frame_miss"] = tracked.get("frame_miss", 0) + 1

        # Cập nhật tracked_faces
        tracked_faces = [
            t for t in tracked_faces
            if t.get("frame_miss", 0) < MAX_FRAME_MISS and (
                    (not t.get("is_identified", False) and t.get("frame_count", 0) <= 30)
                    or (t.get("is_identified", False) and t.get("frame_count", 0) <= 210)
            )
        ]

        # Trả về tất cả khuôn mặt đã tracked dù không nhìn thấy trong frame này
        for t in tracked_faces:
            if t.get("embedding") is None:
                continue

            detail = None
            if t.get("user_id") is None and t.get("frame_count", 0) >= 30:
                detail = "Không tìm thấy người phù hợp"
                if not t.get("reported", False):
                    t["reported"] = True

            faces_to_return.append({
                "bbox": t["bbox_orig"],
                "embedding": t.get("embedding"),
                "user_id": t.get("user_id"),
                "is_identified": t.get("is_identified", False),
                "num_unidentified": t.get("num_unidentified", 0),
                "detail": detail,
                "frame_count": t.get("frame_count", 0)
            })

        return faces_to_return

    requests = []
    detection_results = []
    new_tracked = []
    matched_indices = set()

    # Xử lý tất cả khuôn mặt được phát hiện
    for idx in range(len(bboxes)):
        det_score = float(bboxes[idx, 4])
        if det_score < det_thresh:
            continue
        bbox_resize = bboxes[idx, :4].astype(int).tolist()
        kps = kpss[idx]

        x1 = max((bbox_resize[0] - pad_w) / scale, 0)
        y1 = max((bbox_resize[1] - pad_h) / scale, 0)
        x2 = min((bbox_resize[2] - pad_w) / scale, orig_w)
        y2 = min((bbox_resize[3] - pad_h) / scale, orig_h)
        bbox_orig = [int(x1), int(y1), int(x2), int(y2)]

        matched = False
        for tidx, tracked in enumerate(tracked_faces):
            if iou(bbox_orig, tracked["bbox_orig"]) > IOU_THRESHOLD:
                user_id = tracked.get("user_id")
                is_identified = user_id is not None
                frame_count = tracked.get("frame_count", 0) + 1

                # Tạo tracked mới với bbox mới, giữ lại embedding cũ
                new_t = {
                    "bbox_orig": bbox_orig,
                    "embedding": tracked.get("embedding"),
                    "frame_miss": 0,
                    "user_id": user_id,
                    "num_unidentified": tracked.get("num_unidentified", 0),
                    "is_identified": is_identified,
                    "frame_count": frame_count,
                    "reported": tracked.get("reported", False)  # Giữ trạng thái đã báo cáo
                }

                new_tracked.append(new_t)
                matched = True
                matched_indices.add(tidx)

                detail = None
                if not is_identified and frame_count >= 30:
                    detail = "Không tìm thấy người phù hợp"
                    if not new_t["reported"]:
                        new_t["reported"] = True

                faces_to_return.append({
                    "bbox": bbox_orig,
                    "embedding": tracked.get("embedding"),
                    "user_id": user_id,
                    "is_identified": is_identified,
                    "num_unidentified": tracked.get("num_unidentified", 0),
                    "detail": detail,
                    "frame_count": frame_count
                })
                break

        if not matched:
            # Khuôn mặt mới - cần trích xuất vector
            detection_results.append({
                "bbox_orig": bbox_orig,
                "user_id": None,
                "num_unidentified": 0,
                "is_identified": False,
                "frame_count": 1,
                "reported": False  # Chưa báo cáo
            })
            requests.append((idx, bbox_resize, det_score, kps, img_resize, pad_w, pad_h, scale, orig_w, orig_h))

            # Thêm khuôn mặt mới vào danh sách trả về ngay cả khi chưa có embedding
            faces_to_return.append({
                "bbox": bbox_orig,
                "embedding": None,  # Sẽ được cập nhật sau khi trích xuất
                "user_id": None,
                "is_identified": False,
                "num_unidentified": 0,
                "detail": "Khuôn mặt mới phát hiện",
                "frame_count": 1
            })

    # Trích xuất vector cho khuôn mặt mới
    with ThreadPoolExecutor() as executor:
        joblist = [executor.submit(_extract_single, *args) for args in requests]
        for idx, job in enumerate(joblist):
            ret = job.result()
            if ret is not None:
                detection_results[idx]["embedding"] = ret["embedding"]
                detection_results[idx]["frame_miss"] = 0
                new_tracked.append(detection_results[idx])

                # Cập nhật embedding cho khuôn mặt mới trong danh sách trả về
                for face in faces_to_return:
                    if face["bbox"] == ret["bbox_orig"] and face["embedding"] is None:
                        face["embedding"] = ret["embedding"]
                        break

    # Cập nhật trạng thái identified
    for t in new_tracked:
        if t.get("user_id") is not None:
            t["is_identified"] = True
        else:
            t["is_identified"] = False

    # Cập nhật danh sách tracked_faces
    tracked_faces = [
        t for t in new_tracked
        if t.get("frame_miss", 0) < MAX_FRAME_MISS and (
                (not t.get("is_identified", False) and t.get("frame_count", 0) <= 10)
                or (t.get("is_identified", False) and t.get("frame_count", 0) <= 210)
        )
    ]

    return faces_to_return
