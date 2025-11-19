"""
Simple Object Detector using OWLv2 - Following Official HuggingFace Documentation
Based on: https://huggingface.co/docs/transformers/en/model_doc/owlv2
"""
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import sys
import os

class SimpleOWLv2Detector:
    def __init__(self, model_name="google/owlv2-base-patch16-ensemble", threshold=0.1):
        """
        Initialize OWLv2 detector following official docs
        Args:
            model_name: HuggingFace model name
            threshold: Detection confidence threshold
        """
        print(f"Loading model: {model_name}")
        print(f"Threshold: {threshold}")

        # Load processor and model (exactly as in HF docs)
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.threshold = threshold

        print("✓ Model loaded successfully!\n")

    def detect(self, image_path, text_queries):
        """
        Detect objects in image using text queries
        Follows official HuggingFace example exactly
        """
        # Load image and ensure it's RGB
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Format text queries as list of lists (required by OWLv2)
        # Example: ["cat", "dog"] -> [["cat", "dog"]]
        texts = [text_queries]

        # Process inputs (image + text)
        inputs = self.processor(text=texts, images=image, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get bounding boxes
        # Target size is (height, width)
        target_sizes = torch.Tensor([image.size[::-1]])

        # Use the official post_process method with text labels
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.threshold,
            text_labels=texts
        )

        return results[0], image

    def visualize(self, image, result, output_path="annotated_image.jpg"):
        """
        Draw bounding boxes on image
        """
        # Ensure image is a copy so we don't modify the original
        image = image.copy()
        draw = ImageDraw.Draw(image)

        # Get detections
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["text_labels"]

        # Colors for different objects
        colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple"]

        print(f"\nFound {len(boxes)} objects:")
        print("-" * 70)

        # Try to load a better font
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            # Convert box to list
            box = box.tolist()
            score = score.item()

            print(f"  [{idx}] {label:20s} confidence: {score:.3f}  bbox: {[int(x) for x in box]}")

            # Draw box
            color = colors[idx % len(colors)]
            draw.rectangle(box, outline=color, width=4)

            # Draw label with background
            text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((box[0], box[1] - 20), text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((box[0], box[1] - 20), text, fill="white", font=font)

        # Save image
        image.save(output_path)
        print(f"\n✓ Saved annotated image to: {output_path}")

        return output_path


if __name__ == "__main__":
    print("="*70)
    print("SIMPLE OWLv2 OBJECT DETECTOR")
    print("Following: https://huggingface.co/docs/transformers/en/model_doc/owlv2")
    print("="*70)

    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "scene.png"

    if not os.path.exists(image_path):
        print(f"\n❌ Error: Image '{image_path}' not found!")
        print("Usage: python object_detector.py <image_path>")
        sys.exit(1)

    print(f"\nImage: {image_path}")

    # Define what to look for
    # These are the text queries - OWLv2 will look for these objects
    text_queries = [
        "bowl",
        "plate", 
        "cup",
        "ramekin",
        "cabinet",
        "stove"
    ]

    print(f"Looking for: {text_queries}\n")

    # Initialize detector
    detector = SimpleOWLv2Detector(threshold=0.15)

    # Detect objects
    print("="*70)
    print("RUNNING DETECTION...")
    print("="*70)

    result, image = detector.detect(image_path, text_queries)

    # Visualize results
    output_path = "annotated_" + os.path.basename(image_path)
    detector.visualize(image, result, output_path)

    print("\n" + "="*70)
    print("✓ DONE!")
    print("="*70)
