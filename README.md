![1.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/_04t9LUTuMdTlMqHAONno.png)

# open-deepfake-detection

> open-deepfake-detection is a vision-language encoder model fine-tuned from `siglip2-base-patch16-512` for binary image classification. It is trained to detect whether an image is fake or real using the *OpenDeepfake-Preview* dataset. The model uses the `SiglipForImageClassification` architecture.

> \[!note]
> *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*
> [https://arxiv.org/pdf/2502.14786](https://arxiv.org/pdf/2502.14786)

> \[!important]
Experimental Model 

```py
Classification Report:
              precision    recall  f1-score   support

        Fake     0.9718    0.9155    0.9428     10000
        Real     0.9201    0.9734    0.9460      9999

    accuracy                         0.9444     19999
   macro avg     0.9459    0.9444    0.9444     19999
weighted avg     0.9459    0.9444    0.9444     19999
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/KIQGQnaSxrY1F2TQNpRLR.png)

---

## Label Space: 2 Classes

The model classifies an image as either:

```
Class 0: Fake  
Class 1: Real
```

---

## Install Dependencies

```bash
pip install -q transformers torch pillow gradio hf_xet
```

---

## Inference Code

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/open-deepfake-detection"  # Updated model name
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Updated label mapping
id2label = {
    "0": "Fake",
    "1": "Real"
}

def classify_image(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2, label="Deepfake Detection"),
    title="open-deepfake-detection",
    description="Upload an image to detect whether it is AI-generated (Fake) or a real photograph (Real), using the OpenDeepfake-Preview dataset."
)

if __name__ == "__main__":
    iface.launch()
```

---

## Demo Inference

> [!warning]
real

![Screenshot 2025-05-20 at 14-01-01 Deepfake Detection Model.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/0HPpoJmqIhHMqPo80ZIdc.png)
![Screenshot 2025-05-20 at 14-01-41 Deepfake Detection Model.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/fHB6TCDTHFI5wI7OBNOPZ.png)

> [!warning]
fake

![Screenshot 2025-05-20 at 14-04-22 Deepfake Detection Model.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/wNS6sFeGKroHlPvMyDqJe.png)
![Screenshot 2025-05-20 at 14-08-07 Deepfake Detection Model.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/sKKph7D8MLLhnfjtatnrw.png)

## Intended Use

`open-deepfake-detection` is designed for:

* **Deepfake Detection** – Identify AI-generated or manipulated images.
* **Content Moderation** – Flag synthetic or fake visual content.
* **Dataset Curation** – Remove synthetic samples from mixed datasets.
* **Visual Authenticity Verification** – Check the integrity of visual media.
* **Digital Forensics** – Support image source verification and traceability.
