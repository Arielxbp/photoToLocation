import time
import json
from io import BytesIO
from pathlib import Path
from typing import Optional
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

def load_image(source: str) -> Image.Image:
    """Load from a local path or HTTP(S) URL."""
    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(source).convert("RGB")


def kosmos_caption(
    image: Image.Image,
    prompt: str,
    processor,
    model,
    device: str,
    max_new_tokens: int = 256,
) -> tuple[str, list[dict]]:
    """
    Run a single Kosmos-2 grounded-captioning pass.

    Returns:
        caption   – plain text caption
        entities  – list of {"phrase": str, "bboxes": [[x1,y1,x2,y2], …]}
                    (bboxes are normalised [0, 1])
    """
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    generated = processor.batch_decode(ids, skip_special_tokens=False)[0]
    caption, raw_entities = processor.post_process_generation(generated)
    entities = [
        {"phrase": phrase, "bboxes": list(bboxes)}
        for phrase, _span, bboxes in raw_entities
    ]
    return caption, entities


def identify_language(text: str) -> list[dict]:
    """Run langdetect + langid and return ranked candidates."""
    results = []
    try:
        from langdetect import detect_langs
        for lang in detect_langs(text):
            results.append({"lang": lang.lang, "prob": round(lang.prob, 4), "source": "langdetect"})
    except Exception:
        pass
    try:
        import langid
        lang, score = langid.classify(text)
        results.append({"lang": lang, "prob": round(float(score), 4), "source": "langid"})
    except Exception:
        pass
    return results or [{"lang": "unknown", "prob": 0.0, "source": "none"}]


def infer_driving_side(entities: list[dict]) -> dict:
    """
    Heuristic: count vehicles on left vs right half of image.
    Bboxes are normalised [0,1], so centre_x < 0.5 → left half.
    """
    keywords = {"car", "truck", "bus", "van", "vehicle", "taxi", "lorry", "motorbike"}
    left, right = 0, 0
    for ent in entities:
        if any(k in ent["phrase"].lower() for k in keywords):
            for box in ent["bboxes"]:
                cx = (box[0] + box[2]) / 2
                if cx < 0.5:
                    left += 1
                else:
                    right += 1
    total = left + right
    if total == 0:
        return {"side": "unknown", "confidence": "low", "left": 0, "right": 0}
    ratio = left / total
    if ratio > 0.65:
        side, conf = "LHT (drive on left)", "medium"
    elif ratio < 0.35:
        side, conf = "RHT (drive on right)", "medium"
    else:
        side, conf = "uncertain", "low"
    return {"side": side, "confidence": conf, "left": left, "right": right}


IMAGE_SOURCE = "data/screenshots/round_streetview.png" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
image = load_image(IMAGE_SOURCE)
W, H = image.size
print(f"Image size: {W} × {H}")

MODEL_ID = "microsoft/kosmos-2-patch14-224"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Kosmos2ForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

PROMPT_GENERAL = (
    "<grounding> Describe this street scene in detail, including all visible "
    "objects, signs, vehicles, buildings, and road markings."
)

gen_caption, gen_entities = kosmos_caption(
    image, PROMPT_GENERAL, processor, model, DEVICE, max_new_tokens=512
)

PROMPT_SIGNS = (
    "<grounding> List every traffic sign, road sign, and utility pole "
    "visible in the image."
)

sign_caption, sign_entities = kosmos_caption(
    image, PROMPT_SIGNS, processor, model, DEVICE, max_new_tokens=256
)

PROMPT_PLATES = (
    "<grounding> Identify every license plate in the image and read "
    "the text on each plate."
)

plate_caption, plate_entities = kosmos_caption(
    image, PROMPT_PLATES, processor, model, DEVICE, max_new_tokens=128
)

PROMPT_OCR = (
    "<grounding> Read all text visible in the image, including signs, "
    "storefronts, labels, and license plates."
)

ocr_caption, ocr_entities = kosmos_caption(
    image, PROMPT_OCR, processor, model, DEVICE, max_new_tokens=256
)

# Aggregate all OCR text for language detection
all_text = ocr_caption + " " + " ".join(e["phrase"] for e in ocr_entities)
lang_candidates = identify_language(all_text.strip())

driving = infer_driving_side(gen_entities)

PROMPT_SEG = (
    "<grounding> Segment and label the major regions of this image: "
    "road, sidewalk, buildings, sky, vegetation, vehicles, pedestrians, signs."
)

seg_caption, seg_entities = kosmos_caption(
    image, PROMPT_SEG, processor, model, DEVICE, max_new_tokens=256
)

results = {
    "general_caption": gen_caption,
    "grounded_entities": gen_entities,
    "traffic_signs": {"caption": sign_caption, "entities": sign_entities},
    "license_plates": {"caption": plate_caption, "entities": plate_entities},
    "ocr": {
        "caption": ocr_caption,
        "entities": ocr_entities,
        "language_candidates": lang_candidates,
    },
    "driving_side": driving,
    "scene_segmentation": {"caption": seg_caption, "entities": seg_entities},
}

out_path = Path("streetview_results.json")
out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))