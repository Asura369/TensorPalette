from PIL import Image


def load_image(filename, size=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = _resize_cover_center_crop(img, size)
    return img


def _resize_cover_center_crop(img, size):
    """Resize so the shorter side matches size, then center-crop a square.

    Preserves aspect ratio (no distortion); crops painting edges equally on
    both sides.
    """
    w, h = img.size
    factor = max(size / w, size / h)
    img = img.resize((int(round(w * factor)), int(round(h * factor))), Image.Resampling.LANCZOS)
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div(255.0)
    return (batch - mean) / std
