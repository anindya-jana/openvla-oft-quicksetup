#!/usr/bin/env python3
"""
LIBERO Task Classifier - FIXED VERSION
Usage: python categorize_libero_llm.py --text "pick_up_the_bowl_and_place_it_in_the_basket"
"""

from groq import Groq
import argparse
import json

client = Groq(api_key="YOUR API KEY HERE")

def classify_libero_task(task):
    """Rule-based classifier with LLM fallback - FIXED"""
    
    task_lower = task.lower()
    
    # ============ RULE-BASED PRE-FILTERING (Apply in strict order) ============
    
    # Rule 1: Scene prefix → libero_10 (Must check FIRST and be strict)
    # Only match if it STARTS with scene prefix
    if task.startswith("KITCHEN_SCENE") or task.startswith("LIVING_ROOM_SCENE") or \
       task.startswith("STUDY_SCENE") or task.startswith("BEDROOM_SCENE"):
        return {
            "category": "libero_10",
            "confidence": 1.0,
            "reasoning": "Scene prefix detected (rule-based)"
        }
    
    # Rule 2: Basket action → libero_object (100% accuracy)
    if "place_it_in_the_basket" in task_lower:
        return {
            "category": "libero_object",
            "confidence": 1.0,
            "reasoning": "Basket action detected (rule-based)"
        }
    
    # Rule 3: Goal verbs → libero_goal (Check before spatial)
    if any(task_lower.startswith(kw) for kw in ["open", "close", "turn", "push", "pull"]):
        return {
            "category": "libero_goal",
            "confidence": 1.0,
            "reasoning": "Goal verb detected (rule-based)"
        }
    
    # Rule 4: Spatial + pick/place → libero_spatial
    spatial_kw = ["between", "next_to", "on_the", "in_the", "under_the", "above_the", "from_"]
    has_spatial = any(kw in task_lower for kw in spatial_kw)
    has_pick_place = "pick_up" in task_lower and "place_it" in task_lower
    
    if has_spatial and has_pick_place:
        return {
            "category": "libero_spatial",
            "confidence": 1.0,
            "reasoning": "Spatial + pick/place detected (rule-based)"
        }
    
    # Rule 5: Tasks starting with "put" → libero_goal
    if task_lower.startswith("put"):
        return {
            "category": "libero_goal",
            "confidence": 0.95,
            "reasoning": "Put action detected (rule-based)"
        }
    
    # ============ LLM FALLBACK (Only for ambiguous cases) ============
    
    prompt = f"""Classify this robot task into ONE category: libero_spatial, libero_object, libero_goal, or libero_10

STRICT RULES:
1. libero_10: Task STARTS with KITCHEN_SCENE, LIVING_ROOM_SCENE, or STUDY_SCENE
2. libero_object: Task contains "place_it_in_the_basket"
3. libero_spatial: Task has spatial words (between, next_to, on_the, in_the) AND "pick_up"
4. libero_goal: Other manipulation actions (open, close, turn, push, put)

Task: {task}

Answer with ONLY the category name (libero_spatial, libero_object, libero_goal, or libero_10)"""

    try:
        # Make API call
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50
        )
        
        # FIXED: Correct way to extract response
        response_text = completion.choices[0].message.content.strip().lower()
        
        # Parse category
        if "libero_10" in response_text or "libero10" in response_text:
            category = "libero_10"
        elif "libero_object" in response_text:
            category = "libero_object"
        elif "libero_spatial" in response_text:
            category = "libero_spatial"
        elif "libero_goal" in response_text:
            category = "libero_goal"
        else:
            # Default to goal if unclear
            category = "libero_goal"
        
        return {
            "category": category,
            "confidence": 0.9,
            "reasoning": "LLM classified"
        }
        
    except Exception as e:
        # Ultimate fallback
        return {
            "category": "libero_goal",
            "confidence": 0.5,
            "reasoning": f"Error fallback: {str(e)[:30]}"
        }


def main():
    parser = argparse.ArgumentParser(description='Classify LIBERO robot tasks')
    parser.add_argument('--text', type=str, required=True, help='Task description')
    parser.add_argument('--json', action='store_true', help='JSON output only')
    
    args = parser.parse_args()
    
    result = classify_libero_task(args.text)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nTask: {args.text}")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reasoning: {result['reasoning']}\n")


if __name__ == "__main__":
    main()
