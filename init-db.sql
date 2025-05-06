-- Tạo bảng lưu kết quả nhận dạng
CREATE TABLE IF NOT EXISTS recognize_results
(
    result_id      SERIAL PRIMARY KEY,
    user_id        TEXT      NOT NULL,
    bbox           JSONB     NOT NULL,
    detection_time TIMESTAMP NOT NULL,
    confidence     FLOAT,
    object_name    TEXT
);

-- Tạo bảng lưu ảnh khuôn mặt
CREATE TABLE IF NOT EXISTS face_frames
(
    frame_id  UUID PRIMARY KEY,
    result_id INTEGER REFERENCES recognize_results (result_id) ON DELETE CASCADE,
    idx_frame INTEGER NOT NULL
);

-- Tạo bảng theo dõi hiện diện của người dùng
CREATE TABLE IF NOT EXISTS user_presence
(
    presence_id SERIAL PRIMARY KEY,
    user_id     TEXT      NOT NULL,
    entry_time  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    exit_time   TIMESTAMP,
    is_active   BOOLEAN            DEFAULT TRUE
);

-- Tạo các indexes cho tối ưu hiệu suất
CREATE INDEX IF NOT EXISTS idx_recognize_user_time ON recognize_results (user_id, detection_time);
CREATE INDEX IF NOT EXISTS idx_user_presence_active ON user_presence (user_id, is_active);

-- -- Tạo function để lọc trùng lặp
-- CREATE OR REPLACE FUNCTION filter_duplicate_detections()
-- RETURNS TRIGGER AS $$
-- DECLARE
--     last_detection TIMESTAMPTZ;
--     current_presence_id INTEGER;
-- BEGIN
--     -- Kiểm tra lần nhận dạng cuối cùng của người dùng này
--     SELECT MAX(detection_time)
--     INTO last_detection
--     FROM recognize_results
--     WHERE user_id = NEW.user_id
--       AND detection_time > (NEW.detection_time - INTERVAL '30 seconds');
--
--     -- Nếu không có hoặc đã quá 10 giây, lưu lại và cập nhật presence
--     IF last_detection IS NULL THEN
--         -- Kiểm tra xem người dùng có đang active không
--         SELECT presence_id
--         INTO current_presence_id
--         FROM user_presence
--         WHERE user_id = NEW.user_id
--           AND is_active = TRUE;
--
--         -- Nếu chưa có presence active, tạo mới
--         IF current_presence_id IS NULL THEN
--             INSERT INTO user_presence(user_id) VALUES (NEW.user_id);
--         END IF;
--
--         RETURN NEW;
--     ELSE
--         -- Cập nhật thời gian hoạt động của user_presence
--         UPDATE user_presence
--         SET exit_time = NULL
--         WHERE user_id = NEW.user_id
--           AND is_active = TRUE;
--
--         -- Bỏ qua insert nếu phát hiện quá gần nhau
--         RETURN NULL;
--     END IF;
-- END;
-- $$
-- LANGUAGE plpgsql;
--
-- -- Tạo trigger để áp dụng logic lọc
-- CREATE TRIGGER filter_frequent_detections
-- BEFORE INSERT ON recognize_results
-- FOR EACH ROW
-- EXECUTE FUNCTION filter_duplicate_detections();
--
-- -- Tạo function dọn dẹp dữ liệu cũ
-- CREATE OR REPLACE FUNCTION cleanup_old_face_data()
-- RETURNS void AS $$
-- BEGIN
--     -- Xóa các kết quả nhận dạng cũ không còn hình ảnh liên kết
--     DELETE FROM recognize_results
--     WHERE result_id NOT IN (
--         SELECT DISTINCT result_id FROM face_images
--     )
--     AND detection_time < NOW() - INTERVAL '7 days';
--
--     -- Đóng các user_presence cũ
--     UPDATE user_presence
--     SET is_active = FALSE,
--         exit_time = NOW()
--     WHERE is_active = TRUE
--       AND entry_time < NOW() - INTERVAL '15 minutes';
-- END;
-- $$
-- LANGUAGE plpgsql;
