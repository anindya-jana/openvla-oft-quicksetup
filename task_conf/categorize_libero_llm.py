#!/usr/bin/env python3
"""
LIBERO Task Classifier - Pure LLM Version (No Pre-filtering)
Uses smallest available Groq model for maximum speed
Usage: python categorize_libero_llm.py --text "pick_up_the_bowl_and_place_it_in_the_basket"
"""

from groq import Groq
import argparse
import json

client = Groq(api_key="YOUR API KEY HERE")

def classify_libero_task(task):
    """Pure LLM classifier - No rule-based pre-filtering"""
    
    # Detailed prompt with explicit examples and decision logic
    prompt = f"""You are a robot task classifier. Classify this task into EXACTLY ONE category.

CATEGORIES AND RULES:

1. libero_10: Task STARTS with a scene prefix
   - Must begin with: KITCHEN_SCENE, LIVING_ROOM_SCENE
   - Examples:
     ✓ "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"

2. libero_object: Task ends with "place_it_in_the_basket"
   - Fixed destination: the basket
   - Examples:
     ✓ "pick_up_the_ketchup_and_place_it_in_the_basket"

3. libero_spatial: Task has spatial prepositions + pick/place action
   - Contains: between, next_to, on_the, in_the
   - Examples:
     ✓ "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"

4. libero_goal: Other manipulation goals
   - Actions: open, close, turn, push, pull, put
   - Examples:
     ✓ "open_the_middle_drawer_of_the_cabinet"
     ✓ "put_the_bowl_on_the_stove"

APPLY RULES IN THIS ORDER:
1. Check if starts with scene prefix → libero_10
2. Check if ends with "place_it_in_the_basket" → libero_object
3. Check for spatial words + pick/place → libero_spatial
4. Otherwise → libero_goal

Task to classify: "{task}"

Answer with ONLY the category name (libero_10, libero_object, libero_spatial, or libero_goal). No explanation."""

    try:
        models_to_try = [
            "llama-3.1-8b-instant",     
        ]
        
        response_text = None
        used_model = None
        
        for model in models_to_try:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a precise task classifier. Answer with only the category name."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=30
                )
                
                response_text = completion.choices[0].message.content.strip().lower()
                used_model = model
                break  # Success, exit loop
                
            except Exception as e:
                # Model not available, try next one
                if "does not exist" in str(e) or "not found" in str(e):
                    continue
                else:
                    raise e
        
        if response_text is None:
            raise Exception("All models failed")
        
        # Parse category from response
        if "libero_10" in response_text or "libero10" in response_text:
            category = "libero_10"
        elif "libero_object" in response_text:
            category = "libero_object"
        elif "libero_spatial" in response_text:
            category = "libero_spatial"
        elif "libero_goal" in response_text:
            category = "libero_goal"
        else:
            # Try to extract from first word
            first_word = response_text.split()[0] if response_text.split() else ""
            if "10" in first_word:
                category = "libero_10"
            elif "object" in first_word:
                category = "libero_object"
            elif "spatial" in first_word:
                category = "libero_spatial"
            else:
                category = "libero_goal"
        
        return {
            "category": category,
            "confidence": 0.9,
            "reasoning": f"LLM classified (model: {used_model})"
        }
        
    except Exception as e:
        return {
            "category": "error",
            "confidence": 0.0,
            "reasoning": f"LLM error: {str(e)[:50]}"
        }


def main():
    parser = argparse.ArgumentParser(description='Classify LIBERO robot tasks using pure LLM')
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
