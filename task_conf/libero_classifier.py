"""
LIBERO Task Classifier using Scene Graph
Classifies tasks into: libero_spatial, libero_object, libero_goal, libero_10
Based on scene graph features instead of LLM
"""
import json
from typing import Dict, List

class SceneGraphLIBEROClassifier:
    def __init__(self):
        """Initialize the scene graph based classifier"""

        # LIBERO task category definitions
        self.categories = {
            "libero_spatial": {
                "description": "Tasks requiring spatial reasoning (between, next to, above, below)",
                "keywords": ["between", "next_to", "on_the", "in_the", "under", "above", "below", "beside"]
            },
            "libero_object": {
                "description": "Tasks centered on manipulating different objects (basket placement)",
                "keywords": ["basket", "place_it_in_the_basket", "put_in_basket"]
            },
            "libero_goal": {
                "description": "Goal-conditioned tasks (open, close, turn on, push)",
                "keywords": ["open", "close", "turn", "push", "pull", "put"]
            },
            "libero_10": {
                "description": "10 long-horizon tasks (scene prefixes)",
                "keywords": ["KITCHEN_SCENE", "LIVING_ROOM_SCENE", "STUDY_SCENE", "_SCENE"]
            }
        }

    def analyze_scene_graph(self, scene_graph: Dict) -> Dict:
        """
        Analyze scene graph features for classification
        Args:
            scene_graph: Scene graph dictionary
        Returns:
            Dictionary with scene graph features
        """
        features = {
            "num_objects": scene_graph["metadata"]["num_objects"],
            "num_relationships": scene_graph["metadata"]["num_relationships"],
            "objects": [obj["label"] for obj in scene_graph["objects"]],
            "relationships": [],
            "spatial_relations": [],
            "has_basket": False,
            "has_cabinet": False,
            "has_drawer": False,
            "relationship_complexity": 0
        }

        # Extract relationships and spatial patterns
        for rel in scene_graph["relationships"]:
            rel_type = rel["predicate"]
            features["relationships"].append(rel_type)

            # Track spatial relationships
            if rel_type in ["left_of", "right_of", "above", "below", "near", "between"]:
                features["spatial_relations"].append(rel["triplet"])

        # Check for specific objects
        features["has_basket"] = "basket" in features["objects"]
        features["has_cabinet"] = "cabinet" in features["objects"]
        features["has_drawer"] = "drawer" in features["objects"]

        # Compute relationship complexity (ratio of relationships to objects)
        if features["num_objects"] > 0:
            features["relationship_complexity"] = features["num_relationships"] / features["num_objects"]

        return features

    def classify_from_text(self, task_text: str) -> Dict:
        """
        Classify task from text description (rule-based)
        Args:
            task_text: Task description string
        Returns:
            Classification result
        """
        task_lower = task_text.lower()

        # Rule 1: Scene prefix → libero_10 (100% accuracy)
        if any(prefix in task_text for prefix in ["KITCHEN_SCENE", "LIVING_ROOM_SCENE", "STUDY_SCENE"]):
            return {
                "category": "libero_10",
                "confidence": 1.0,
                "reasoning": "Has scene prefix",
                "method": "text_rule"
            }

        # Rule 2: Basket action → libero_object (100% accuracy)
        if "place_it_in_the_basket" in task_lower or "basket" in task_lower:
            return {
                "category": "libero_object",
                "confidence": 1.0,
                "reasoning": "Basket manipulation task",
                "method": "text_rule"
            }

        # Rule 3: Goal verbs → libero_goal (90% accuracy)
        goal_verbs = ["open", "close", "turn", "push", "pull"]
        if any(task_lower.startswith(kw) for kw in goal_verbs):
            return {
                "category": "libero_goal",
                "confidence": 0.90,
                "reasoning": f"Starts with goal verb",
                "method": "text_rule"
            }

        # Rule 4: Spatial keywords + pick/place → libero_spatial (95% accuracy)
        spatial_kw = ["between", "next_to", "on_the", "in_the", "under_the", "above_the"]
        if any(kw in task_lower for kw in spatial_kw) and "pick_up" in task_lower:
            return {
                "category": "libero_spatial",
                "confidence": 0.95,
                "reasoning": "Spatial preposition + pick/place action",
                "method": "text_rule"
            }

        # Default: libero_goal
        return {
            "category": "libero_goal",
            "confidence": 0.5,
            "reasoning": "Default classification",
            "method": "text_rule"
        }

    def classify_from_scene_graph(self, scene_graph: Dict, task_text: str = None) -> Dict:
        """
        Classify task using scene graph features
        Args:
            scene_graph: Scene graph dictionary
            task_text: Optional task text for additional context
        Returns:
            Classification result with reasoning
        """
        # Analyze scene graph features
        features = self.analyze_scene_graph(scene_graph)

        # First try text-based rules if available
        if task_text:
            text_result = self.classify_from_text(task_text)
            # If high confidence from text, return it
            if text_result["confidence"] >= 0.9:
                text_result["scene_graph_features"] = features
                return text_result

        # Scene graph based classification
        scores = {
            "libero_spatial": 0.0,
            "libero_object": 0.0,
            "libero_goal": 0.0,
            "libero_10": 0.0
        }

        reasoning_parts = []

        # Feature 1: Basket presence → libero_object
        if features["has_basket"]:
            scores["libero_object"] += 0.8
            reasoning_parts.append("basket detected")

        # Feature 2: Cabinet/drawer → libero_goal
        if features["has_cabinet"] or features["has_drawer"]:
            scores["libero_goal"] += 0.6
            reasoning_parts.append("cabinet/drawer detected (goal-oriented)")

        # Feature 3: High spatial relationship density → libero_spatial
        if len(features["spatial_relations"]) >= 2:
            scores["libero_spatial"] += 0.7
            reasoning_parts.append(f"{len(features['spatial_relations'])} spatial relationships")

        # Feature 4: Multiple objects with relationships → libero_spatial
        if features["relationship_complexity"] >= 1.5:
            scores["libero_spatial"] += 0.5
            reasoning_parts.append(f"high relationship complexity ({features['relationship_complexity']:.2f})")

        # Feature 5: Few objects, simple task → libero_goal
        if features["num_objects"] <= 2:
            scores["libero_goal"] += 0.4
            reasoning_parts.append("simple object configuration")

        # Feature 6: Many objects → libero_10 or libero_spatial
        if features["num_objects"] >= 4:
            scores["libero_10"] += 0.3
            scores["libero_spatial"] += 0.3
            reasoning_parts.append("complex scene with many objects")

        # Find best category
        best_category = max(scores, key=scores.get)
        confidence = scores[best_category]

        # Normalize confidence to 0-1 range
        confidence = min(1.0, confidence)

        # If no strong signal, use text-based fallback
        if confidence < 0.3 and task_text:
            return self.classify_from_text(task_text)

        result = {
            "category": best_category,
            "confidence": confidence,
            "reasoning": " | ".join(reasoning_parts) if reasoning_parts else "Scene graph analysis",
            "method": "scene_graph",
            "scene_graph_features": features,
            "all_scores": scores
        }

        return result

    def classify(self, task_text: str = None, scene_graph: Dict = None, 
                scene_graph_path: str = None) -> Dict:
        """
        Main classification method - uses both text and scene graph
        Args:
            task_text: Task description text
            scene_graph: Scene graph dictionary
            scene_graph_path: Path to scene graph JSON file
        Returns:
            Classification result
        """
        # Load scene graph from file if path provided
        if scene_graph_path and not scene_graph:
            with open(scene_graph_path, 'r') as f:
                scene_graph = json.load(f)

        # Classify using available information
        if scene_graph:
            return self.classify_from_scene_graph(scene_graph, task_text)
        elif task_text:
            return self.classify_from_text(task_text)
        else:
            raise ValueError("Must provide either task_text or scene_graph")

    def print_classification(self, result: Dict):
        """Print classification result in readable format"""
        print("\n" + "="*70)
        print("LIBERO TASK CLASSIFICATION")
        print("="*70)
        print(f"Category:   {result['category'].upper()}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Method:     {result['method']}")
        print(f"Reasoning:  {result['reasoning']}")

        if "all_scores" in result:
            print(f"\nAll Scores:")
            for cat, score in result["all_scores"].items():
                print(f"  {cat:20s}: {score:.3f}")

        if "scene_graph_features" in result:
            features = result["scene_graph_features"]
            print(f"\nScene Graph Features:")
            print(f"  Objects: {features['num_objects']} - {features['objects']}")
            print(f"  Relationships: {features['num_relationships']}")
            print(f"  Spatial Relations: {len(features['spatial_relations'])}")
            if features['spatial_relations']:
                for rel in features['spatial_relations'][:3]:
                    print(f"    - {rel}")

        print("="*70)


# Test the classifier
if __name__ == "__main__":
    # Test with example scene graph
    test_scene_graph = {
        "objects": [
            {"object_id": 0, "label": "bowl", "bbox": [100, 100, 200, 200], "score": 0.95},
            {"object_id": 1, "label": "plate", "bbox": [250, 150, 350, 250], "score": 0.90},
            {"object_id": 2, "label": "basket", "bbox": [400, 100, 500, 200], "score": 0.85}
        ],
        "relationships": [
            {"subject": "bowl", "predicate": "left_of", "object": "plate", "triplet": "bowl - left_of - plate"},
            {"subject": "bowl", "predicate": "near", "object": "basket", "triplet": "bowl - near - basket"},
            {"subject": "plate", "predicate": "left_of", "object": "basket", "triplet": "plate - left_of - basket"}
        ],
        "metadata": {"num_objects": 3, "num_relationships": 3}
    }

    classifier = SceneGraphLIBEROClassifier()

    print("Testing LIBERO Task Classifier with Scene Graph")
    print("="*70)

    # Test 1: With scene graph only
    print("\n[Test 1] Scene graph only (basket present):")
    result1 = classifier.classify(scene_graph=test_scene_graph)
    classifier.print_classification(result1)

    # Test 2: With task text only
    print("\n[Test 2] Text only:")
    result2 = classifier.classify(task_text="pick_up_the_black_bowl_between_the_plate_and_the_ramekin")
    classifier.print_classification(result2)

    # Test 3: With both
    print("\n[Test 3] Both scene graph and text:")
    result3 = classifier.classify(
        task_text="pick_up_the_bowl_and_place_it_in_the_basket",
        scene_graph=test_scene_graph
    )
    classifier.print_classification(result3)
