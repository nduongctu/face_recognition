import numpy as np
from PIL import Image


def resize_image(img_np, target_size=(112, 112)):
    """
    Resize ảnh numpy array về kích thước target_size
    :param img_np: ảnh đầu vào dạng numpy array (RGB)
    :param target_size: kích thước mong muốn (mặc định là 112x112)
    :return: ảnh đã resize dưới dạng numpy array
    """
    # Chuyển ảnh numpy array thành đối tượng PIL Image
    image = Image.fromarray(img_np)

    # Resize ảnh
    image_resized = image.resize(target_size)

    # Chuyển lại thành numpy array
    img_resized_np = np.array(image_resized)

    # Đảm bảo ảnh có 3 kênh màu (RGB)
    if img_resized_np.shape[-1] != 3:
        img_resized_np = np.stack([img_resized_np] * 3, axis=-1)  # Chuyển thành 3 kênh nếu cần

    return img_resized_np
