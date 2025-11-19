#!/usr/bin/env python3
"""
Complete Scene Graph Pipeline with LIBERO Task Classification
Extracts objects → Detects in image → Generates scene graph → Classifies LIBERO task

Usage:
    python main.py --text "pick_up_the_black_bowl_and_place_it_on_the_plate" --image scene.png
"""

import argparse
import os
import sys
from pathlib import Path
import json

from object_extractor import ObjectExtractor
from object_detector import SimpleOWLv2Detector
from scene_graph_generator import SceneGraphGenerator
from libero_classifier import SceneGraphLIBEROClassifier


class CompleteSceneGraphPipeline:
    def __init__(self, threshold=0.15):
        """
        Initialize the complete pipeline with LIBERO classification
        Args:
            threshold: Detection confidence threshold for OWLv2
        """
        print("="*70)
        print("INITIALIZING COMPLETE SCENE GRAPH PIPELINE")
        print("="*70)

        # Initialize modules
        print("\n[1/4] Loading Object Extractor...")
        self.object_extractor = ObjectExtractor()

        print("\n[2/4] Loading OWLv2 Detector...")
        self.object_detector = SimpleOWLv2Detector(threshold=threshold)

        print("\n[3/4] Loading Scene Graph Generator...")
        self.scene_graph_generator = SceneGraphGenerator()

        print("\n[4/4] Loading LIBERO Task Classifier...")
        self.libero_classifier = SceneGraphLIBEROClassifier()

        print("\n" + "="*70)
        print("✓ PIPELINE READY!")
        print("="*70 + "\n")

    def run(self, text: str, image_path: str, output_dir: str = "./output"):
        """
        Run complete pipeline with classification
        Args:
            text: Task description text
            image_path: Path to input image
            output_dir: Directory to save outputs
        Returns:
            Dictionary with all results including classification
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*70)
        print("STEP 1: OBJECT EXTRACTION FROM TEXT")
        print("="*70)
        print(f"Task: {text}")

        # Step 1: Extract objects from text
        objects = self.object_extractor.extract_from_text(text)
        print(f"\nExtracted Objects: {objects}")

        if not objects:
            print("\n⚠️  Warning: No objects extracted from text!")
            print("Using default queries: ['object', 'item', 'thing']")
            objects = ['object', 'item', 'thing']

        # Step 2: Detect objects in image
        print("\n" + "="*70)
        print("STEP 2: OBJECT DETECTION WITH OWLv2")
        print("="*70)
        print(f"Image: {image_path}")
        print(f"Looking for: {objects}\n")

        result, image = self.object_detector.detect(image_path, objects)

        num_detections = len(result['boxes'])
        print(f"\n✓ Detected {num_detections} objects")

        # Check if no objects detected
        if num_detections == 0:
            print("\n" + "="*70)
            print("❌ ERROR: NO OBJECTS DETECTED")
            print("="*70)
            print("\nCannot generate scene graph or classify task without detected objects.")
            print("\nPossible solutions:")
            print("  1. Lower detection threshold: --threshold 0.05")
            print("  2. Try different object queries (check extracted objects)")
            print("  3. Verify image contains the mentioned objects")
            print("  4. Check image quality and lighting")
            print()

            # Create error result
            error_result = {
                "status": "error",
                "error_type": "no_objects_detected",
                "message": "Cannot determine task category - no objects detected in image",
                "text": text,
                "image_path": image_path,
                "extracted_objects": objects,
                "num_detections": 0,
                "suggestions": [
                    "Lower detection threshold",
                    "Verify image contains mentioned objects",
                    "Check image quality"
                ]
            }

            # Save error result
            error_path = os.path.join(output_dir, "error_result.json")
            with open(error_path, 'w') as f:
                json.dump(error_result, f, indent=2)
            print(f"Error details saved to: {error_path}\n")

            return error_result

        # Step 3: Visualize detections
        print("\n" + "="*70)
        print("STEP 3: VISUALIZATION")
        print("="*70)

        annotated_path = os.path.join(output_dir, "annotated_image.jpg")
        self.object_detector.visualize(image, result, annotated_path)

        # Step 4: Generate scene graph
        print("\n" + "="*70)
        print("STEP 4: SCENE GRAPH GENERATION")
        print("="*70)

        scene_graph = self.scene_graph_generator.generate_from_owlv2_result(
            result, image_path=image_path
        )

        # Print summary
        self.scene_graph_generator.print_scene_graph_summary(scene_graph)

        # Save scene graph
        scene_graph_path = os.path.join(output_dir, "scene_graph.json")
        self.scene_graph_generator.save_scene_graph(scene_graph, scene_graph_path)

        # Step 5: LIBERO Task Classification
        print("\n" + "="*70)
        print("STEP 5: LIBERO TASK CLASSIFICATION")
        print("="*70)

        classification = self.libero_classifier.classify(
            task_text=text,
            scene_graph=scene_graph
        )

        self.libero_classifier.print_classification(classification)

        # Save classification result
        classification_path = os.path.join(output_dir, "classification.json")
        with open(classification_path, 'w') as f:
            json.dump(classification, f, indent=2)
        print(f"\nClassification saved to: {classification_path}")

        # Return complete results
        results = {
            "status": "success",
            "text": text,
            "image_path": image_path,
            "extracted_objects": objects,
            "num_detections": num_detections,
            "scene_graph": scene_graph,
            "classification": classification,
            "outputs": {
                "annotated_image": annotated_path,
                "scene_graph": scene_graph_path,
                "classification": classification_path
            }
        }

        print("\n" + "="*70)
        print("✓ COMPLETE PIPELINE FINISHED!")
        print("="*70)
        print(f"\nAll outputs saved to: {output_dir}/")
        print(f"  - Annotated Image:  {annotated_path}")
        print(f"  - Scene Graph:      {scene_graph_path}")
        print(f"  - Classification:   {classification_path}")
        print(f"\nTask classified as: {classification['category'].upper()} "
              f"(confidence: {classification['confidence']:.2f})")
        print()

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Complete Scene Graph Pipeline with LIBERO Task Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single task
  python main.py --text "pick_up_the_black_bowl_and_place_it_on_the_plate" --image scene.png

  # Basket task (should classify as libero_object)
  python main.py --text "pick_up_the_bowl_and_place_it_in_the_basket" --image scene.png

  # Spatial task
  python main.py --text "pick_up_the_bowl_between_the_plate_and_the_cup" --image scene.png

  # Goal task
  python main.py --text "open_the_middle_drawer_of_the_cabinet" --image scene.png

  # Lower threshold if no detections
  python main.py --text "..." --image scene.png --threshold 0.05
        """
    )

    # Input arguments
    parser.add_argument("--text", type=str, help="Task description text")
    parser.add_argument("--text_file", type=str, help="Path to file with task descriptions")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")

    # Output arguments
    parser.add_argument("--output", type=str, default="./output", 
                       help="Output directory (default: ./output)")

    # Model arguments
    parser.add_argument("--threshold", type=float, default=0.15,
                       help="Detection confidence threshold (default: 0.15)")

    args = parser.parse_args()

    # Validate inputs
    if not args.text and not args.text_file:
        parser.error("Either --text or --text_file must be provided")

    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        sys.exit(1)

    # Get task descriptions
    if args.text:
        tasks = [args.text]
    else:
        with open(args.text_file, 'r') as f:
            tasks = [line.strip() for line in f if line.strip()]

    # Initialize pipeline
    pipeline = CompleteSceneGraphPipeline(threshold=args.threshold)

    # Process first task (you can loop for multiple tasks)
    if len(tasks) > 1:
        print(f"\nNote: Processing first task from {len(tasks)} total tasks.")
        print("To process all tasks, modify the loop below.\n")

    task = tasks[0]
    results = pipeline.run(task, args.image, args.output)

    # Exit with appropriate code
    if results.get("status") == "error":
        sys.exit(1)

    return results


if __name__ == "__main__":
    main()
