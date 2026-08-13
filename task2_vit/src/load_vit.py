import os
import torch
import requests
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

def load_vit(model_name="google/vit-base-patch16-224"):
    """
    Load pretrained ViT classification model and matching image processor.
    Uses eager attention implementation so output_attentions=True returns full attention maps.
    """
    processor = ViTImageProcessor.from_pretrained(model_name)
    try:
        model = ViTForImageClassification.from_pretrained(model_name, attn_implementation="eager")
    except Exception:
        model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()
    return model, processor

def preprocess_image(processor, image):
    """
    Preprocess PIL image into model expected pixel values tensor.
    """
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values']

def get_imagenet_labels(model):
    """
    Return the id->label mapping from the model config, supporting both int and str key lookup.
    """
    id2label = model.config.id2label
    norm = {}
    for k, v in id2label.items():
        norm[int(k)] = v
        norm[str(k)] = v
    return norm

def download_sample_images(save_dir=None):
    """
    Download 3 sample images (golden retriever, tabby cat, sports car)
    to save_dir and return list of (filepath, description) tuples.
    If save_dir is None, defaults to task2_vit/images/ relative to workspace root.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")
    os.makedirs(save_dir, exist_ok=True)

    samples = [
        {
            "filename": "golden_retriever.jpg",
            "url": "https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_640.jpg",
            "description": "Golden Retriever"
        },
        {
            "filename": "tabby_cat.jpg",
            "url": "https://cdn.pixabay.com/photo/2014/11/30/14/11/cat-551554_640.jpg",
            "description": "Tabby Cat"
        },
        {
            "filename": "sports_car.jpg",
            "url": "https://cdn.pixabay.com/photo/2012/11/02/13/02/car-63930_640.jpg",
            "description": "Sports Car"
        },
    ]


    results = []
    for s in samples:
        fpath = os.path.join(save_dir, s["filename"])
        if not os.path.exists(fpath):
            print(f"Downloading {s['description']}...")
            try:
                resp = requests.get(s["url"], timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                with open(fpath, "wb") as f:
                    f.write(resp.content)
            except Exception as e:
                print(f"  Warning: Failed to download {s['filename']}: {e}")
                # Create a fallback image with text
                from PIL import ImageDraw, ImageFont
                img = Image.new("RGB", (224, 224), color=(200, 200, 200))
                draw = ImageDraw.Draw(img)
                draw.text((30, 100), s["description"], fill=(0, 0, 0))
                img.save(fpath)
        results.append((fpath, s["description"]))
    return results

def load_sample_images(images_dir=None):
    """
    Load already downloaded sample images from disk. Returns list of (PIL.Image, description).
    """
    if images_dir is None:
        images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")

    samples = [
        ("golden_retriever.jpg", "Golden Retriever"),
        ("tabby_cat.jpg", "Tabby Cat"),
        ("sports_car.jpg", "Sports Car"),
    ]

    loaded = []
    for fname, desc in samples:
        fpath = os.path.join(images_dir, fname)
        if os.path.exists(fpath):
            img = Image.open(fpath).convert("RGB")
            loaded.append((img, desc))
    return loaded
