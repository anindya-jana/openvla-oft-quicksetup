#!/usr/bin/env python3
"""
LIBERO Task Classifier Benchmark Script
Tests categorize_libero_llm.py against ground truth LIBERO tasks
"""

import subprocess
import json
from collections import defaultdict

# Ground truth LIBERO tasks
libero_task_map = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ],
    "libero_10": [
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ],
}


def run_classifier(task_text):
    """Run the classifier on a task and get prediction"""
    try:
        result = subprocess.run(
            ['python', 'categorize_libero_llm.py', '--text', task_text, '--json'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Parse JSON output
            output = json.loads(result.stdout)
            return output
        else:
            return {"category": "error", "confidence": 0.0, "reasoning": "Command failed"}
    except subprocess.TimeoutExpired:
        return {"category": "error", "confidence": 0.0, "reasoning": "Timeout"}
    except json.JSONDecodeError:
        return {"category": "error", "confidence": 0.0, "reasoning": "JSON parse error"}
    except Exception as e:
        return {"category": "error", "confidence": 0.0, "reasoning": str(e)}


def run_benchmark():
    """Run benchmark on all LIBERO tasks"""
    
    print("="*80)
    print("LIBERO TASK CLASSIFIER BENCHMARK")
    print("="*80)
    print()
    
    # Statistics
    stats = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "errors": 0,
        "by_category": defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0})
    }
    
    # Store misclassifications
    misclassified = []
    
    # Test each category
    for true_category, tasks in libero_task_map.items():
        print(f"\n{'='*80}")
        print(f"Testing: {true_category.upper()} ({len(tasks)} tasks)")
        print(f"{'='*80}")
        
        for i, task in enumerate(tasks, 1):
            stats["total"] += 1
            stats["by_category"][true_category]["total"] += 1
            
            # Run classifier
            result = run_classifier(task)
            predicted_category = result["category"]
            confidence = result["confidence"]
            
            # Check correctness
            is_correct = (predicted_category == true_category)
            
            if predicted_category == "error":
                stats["errors"] += 1
                status = "❌ ERROR"
            elif is_correct:
                stats["correct"] += 1
                stats["by_category"][true_category]["correct"] += 1
                status = "✓"
            else:
                stats["incorrect"] += 1
                stats["by_category"][true_category]["incorrect"] += 1
                status = "✗"
                misclassified.append({
                    "task": task,
                    "true": true_category,
                    "predicted": predicted_category,
                    "confidence": confidence
                })
            
            # Print result
            task_short = task[:65] + "..." if len(task) > 65 else task
            print(f"[{i:2d}] {status} {task_short}")
            if not is_correct:
                print(f"     Expected: {true_category} | Got: {predicted_category} (conf: {confidence:.2f})")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Overall accuracy
    accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
    print(f"\nOverall Accuracy: {stats['correct']}/{stats['total']} = {accuracy:.2f}%")
    print(f"Correct: {stats['correct']}")
    print(f"Incorrect: {stats['incorrect']}")
    print(f"Errors: {stats['errors']}")
    
    # Per-category accuracy
    print(f"\n{'Category':<20} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 50)
    for category in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        cat_stats = stats["by_category"][category]
        cat_acc = (cat_stats["correct"] / cat_stats["total"] * 100) if cat_stats["total"] > 0 else 0
        print(f"{category:<20} {cat_stats['total']:<8} {cat_stats['correct']:<8} {cat_acc:.2f}%")
    
    # Confusion matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    
    confusion = defaultdict(lambda: defaultdict(int))
    for item in misclassified:
        confusion[item["true"]][item["predicted"]] += 1
    
    if confusion:
        categories = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
        # Fix: Use variable instead of backslash in f-string
        header = "True / Predicted"
        print(f"\n{header:<20} ", end="")
        for cat in categories:
            print(f"{cat.replace('libero_', ''):<12}", end="")
        print()
        print("-" * 80)
        
        for true_cat in categories:
            print(f"{true_cat.replace('libero_', ''):<20} ", end="")
            for pred_cat in categories:
                count = confusion[true_cat][pred_cat]
                print(f"{count:<12}", end="")
            print()
    else:
        print("\n✓ Perfect classification! No errors to show in confusion matrix.")
    
    # Misclassified tasks
    if misclassified:
        print("\n" + "="*80)
        print(f"MISCLASSIFIED TASKS ({len(misclassified)} tasks)")
        print("="*80)
        
        for i, item in enumerate(misclassified, 1):
            print(f"\n[{i}] {item['task'][:70]}...")
            print(f"    True: {item['true']} | Predicted: {item['predicted']} (confidence: {item['confidence']:.2f})")
    
    # Save results to file
    results = {
        "overall_accuracy": accuracy,
        "total_tasks": stats["total"],
        "correct": stats["correct"],
        "incorrect": stats["incorrect"],
        "errors": stats["errors"],
        "per_category": {
            cat: {
                "total": stats["by_category"][cat]["total"],
                "correct": stats["by_category"][cat]["correct"],
                "accuracy": (stats["by_category"][cat]["correct"] / stats["by_category"][cat]["total"] * 100) 
                           if stats["by_category"][cat]["total"] > 0 else 0
            }
            for cat in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
        },
        "misclassified": misclassified
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ Benchmark complete! Results saved to benchmark_results.json")
    print("="*80)


if __name__ == "__main__":
    run_benchmark()
