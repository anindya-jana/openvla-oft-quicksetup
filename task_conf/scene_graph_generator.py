"""
Scene Graph Generator Module
Generates scene graphs from object detections with spatial relationships
Compatible with simplified OWLv2 detector
"""
import json
import numpy as np
from typing import List, Dict, Tuple

class SceneGraphGenerator:
    def __init__(self):
        """Initialize scene graph generator"""
        self.spatial_relations = [
            "left_of", "right_of", "above", "below", 
            "near", "far_from"
        ]

        # Thresholds for spatial relationship detection
        self.near_threshold = 150  # pixels
        self.far_threshold = 400   # pixels

    def get_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center coordinates of bounding box"""
        xmin, ymin, xmax, ymax = bbox
        return ((xmin + xmax) / 2, (ymin + ymax) / 2)

    def compute_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Euclidean distance between centers of two boxes"""
        c1 = self.get_center(bbox1)
        c2 = self.get_center(bbox2)
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def determine_spatial_relations(self, bbox1: List[float], bbox2: List[float]) -> List[str]:
        """
        Determine spatial relationships between two objects
        Args:
            bbox1, bbox2: Bounding boxes [xmin, ymin, xmax, ymax]
        Returns:
            List of relationship names
        """
        relations = []

        c1 = self.get_center(bbox1)
        c2 = self.get_center(bbox2)
        distance = self.compute_distance(bbox1, bbox2)

        # Horizontal relationships (with threshold)
        horizontal_threshold = 80
        if c1[0] < c2[0] - horizontal_threshold:
            relations.append("left_of")
        elif c1[0] > c2[0] + horizontal_threshold:
            relations.append("right_of")

        # Vertical relationships (with threshold)
        vertical_threshold = 80
        if c1[1] < c2[1] - vertical_threshold:
            relations.append("above")
        elif c1[1] > c2[1] + vertical_threshold:
            relations.append("below")

        # Proximity relationships
        if distance < self.near_threshold:
            relations.append("near")
        elif distance > self.far_threshold:
            relations.append("far_from")

        # Default to "near" if no other relations found
        return relations if relations else ["near"]

    def generate_from_owlv2_result(self, result: Dict, image_path: str = None) -> Dict:
        """
        Generate scene graph from OWLv2 detection result
        Args:
            result: OWLv2 detection result dict with 'boxes', 'scores', 'text_labels'
            image_path: Optional path to source image
        Returns:
            Scene graph dictionary
        """
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["text_labels"]

        # Convert tensors to lists if needed
        if hasattr(boxes, 'tolist'):
            boxes = [box.tolist() for box in boxes]
        if hasattr(scores, 'tolist'):
            scores = [score.item() for score in scores]

        # Initialize scene graph structure
        scene_graph = {
            "objects": [],
            "relationships": [],
            "metadata": {
                "num_objects": len(boxes),
                "image_path": image_path
            }
        }

        # Add objects (nodes)
        for idx, (bbox, score, label) in enumerate(zip(boxes, scores, labels)):
            obj_node = {
                "object_id": idx,
                "label": label,
                "bbox": bbox,
                "score": float(score),
                "attributes": {
                    "center": list(self.get_center(bbox)),
                    "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                }
            }
            scene_graph["objects"].append(obj_node)

        # Add relationships (edges)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                bbox1 = boxes[i]
                bbox2 = boxes[j]
                label1 = labels[i]
                label2 = labels[j]

                # Compute spatial relationships
                relations = self.determine_spatial_relations(bbox1, bbox2)

                for relation in relations:
                    relationship = {
                        "subject_id": i,
                        "subject": label1,
                        "predicate": relation,
                        "object_id": j,
                        "object": label2,
                        "triplet": f"{label1} - {relation} - {label2}"
                    }
                    scene_graph["relationships"].append(relationship)

        scene_graph["metadata"]["num_relationships"] = len(scene_graph["relationships"])

        return scene_graph

    def save_scene_graph(self, scene_graph: Dict, output_path: str = "scene_graph.json"):
        """
        Save scene graph to JSON file
        Args:
            scene_graph: Scene graph dictionary
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(scene_graph, f, indent=2)
        print(f"Scene graph saved to: {output_path}")

        return output_path

    def print_scene_graph_summary(self, scene_graph: Dict):
        """Print human-readable summary of scene graph"""
        print("\n" + "="*70)
        print("SCENE GRAPH SUMMARY")
        print("="*70)

        print(f"\nObjects ({scene_graph['metadata']['num_objects']}):")
        print("-" * 70)
        for obj in scene_graph["objects"]:
            center = obj['attributes']['center']
            print(f"  [{obj['object_id']}] {obj['label']:20s} score: {obj['score']:.3f}  "
                  f"center: ({int(center[0])}, {int(center[1])})")

        print(f"\nRelationships ({scene_graph['metadata']['num_relationships']}):")
        print("-" * 70)
        for rel in scene_graph["relationships"]:
            print(f"  {rel['triplet']}")

        print("\n" + "="*70)


# Test function
if __name__ == "__main__":
    # Test with mock OWLv2 result
    print("Testing Scene Graph Generator...")
    print("="*70)

    # Mock detection result (as returned by simplified OWLv2 detector)
    mock_result = {
        "boxes": [[100, 100, 200, 200], [250, 150, 350, 250], [400, 100, 500, 200]],
        "scores": [0.95, 0.90, 0.85],
        "text_labels": ["bowl", "plate", "cup"]
    }

    generator = SceneGraphGenerator()
    scene_graph = generator.generate_from_owlv2_result(mock_result, "test_image.jpg")

    generator.print_scene_graph_summary(scene_graph)
    generator.save_scene_graph(scene_graph, "test_scene_graph.json")

    print("\n✓ Scene graph generation test completed!")
