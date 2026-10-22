"""One image policy shared by extraction and serving."""
from dataclasses import asdict, dataclass
from io import BytesIO

from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImagePolicy:
    version: int = 1
    normalize_jpeg: bool = False
    jpeg_quality: int = 90
    exif_transpose: bool = True
    max_pixels: int = 40_000_000

    def __post_init__(self):
        if (type(self.normalize_jpeg) is not bool or type(self.exif_transpose) is not bool
                or any(type(x) is not int for x in (self.version, self.jpeg_quality, self.max_pixels))
                or self.version != 1 or not 1 <= self.jpeg_quality <= 100 or self.max_pixels < 1):
            raise ValueError("Invalid image policy")

    def to_dict(self):
        return asdict(self)


def validate_image(image, policy):
    """Check a still-image header before allocating pixels or loading a model.

    This does not transpose, convert, re-encode or decode the image. Callers at
    an upload boundary must also call ``image.load()`` to reject truncated data.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Expected a decoded Pillow image")
    if image.width * image.height > policy.max_pixels:
        raise ValueError("Image exceeds pixel limit")
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("Multi-frame images are not supported")


def prepare_image(image, policy):
    validate_image(image, policy)
    image = ImageOps.exif_transpose(image) if policy.exif_transpose else image
    image = image.convert("RGB").copy()
    if policy.normalize_jpeg:
        # This standardizes one encoding step; it does NOT eliminate source bias.
        with BytesIO() as buf:
            image.save(buf, format="JPEG", quality=policy.jpeg_quality)
            buf.seek(0)
            with Image.open(buf) as decoded:
                image = decoded.convert("RGB").copy()
    return image


def load_image(path, policy):
    with Image.open(path) as image:
        return prepare_image(image, policy)
