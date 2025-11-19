"""
Improved Object Extractor Module with Comprehensive Knowledge Base
Extracts object names from task description texts using large vocabulary
"""
import re
from typing import List, Set

class ObjectExtractor:
    def __init__(self):
        """
        Initialize object extractor with comprehensive knowledge base
        """
        # Comprehensive object knowledge base combining multiple sources
        # Based on COCO, Visual Genome, kitchen objects, and manipulation objects
        self.object_knowledge_base = {
            # Kitchen & Dining
            'bowl', 'plate', 'dish', 'cup', 'mug', 'glass', 'fork', 'knife', 'spoon',
            'chopsticks', 'pot', 'pan', 'wok', 'skillet', 'kettle', 'teapot',
            'bottle', 'jar', 'container', 'lid', 'tray', 'platter', 'saucer',
            'ramekin', 'pitcher', 'jug', 'carafe', 'decanter', 'flask',

            # Appliances & Furniture
            'stove', 'oven', 'microwave', 'refrigerator', 'fridge', 'freezer',
            'dishwasher', 'toaster', 'blender', 'mixer', 'cabinet', 'drawer',
            'shelf', 'rack', 'counter', 'table', 'chair', 'stool', 'desk',

            # Food Items
            'bread', 'butter', 'cheese', 'milk', 'egg', 'meat', 'fish', 'chicken',
            'beef', 'pork', 'vegetable', 'fruit', 'apple', 'orange', 'banana',
            'tomato', 'potato', 'onion', 'carrot', 'lettuce', 'salad',
            'soup', 'sauce', 'ketchup', 'mustard', 'mayo', 'mayonnaise',
            'salt', 'pepper', 'sugar', 'flour', 'oil', 'vinegar',
            'rice', 'pasta', 'noodle', 'cereal', 'cookie', 'cake', 'pie',
            'chocolate', 'candy', 'snack', 'chip', 'cracker',

            # Packaged Foods (LIBERO specific)
            'alphabet_soup', 'cream_cheese', 'salad_dressing', 'bbq_sauce',
            'tomato_sauce', 'chocolate_pudding', 'orange_juice', 'wine_bottle',
            'coffee', 'tea', 'soda', 'juice', 'water',

            # Containers & Storage
            'box', 'bag', 'basket', 'bin', 'bucket', 'can', 'carton',
            'package', 'wrapper', 'foil', 'plastic_wrap',

            # Cooking Tools
            'spatula', 'ladle', 'whisk', 'tongs', 'peeler', 'grater',
            'cutting_board', 'chopping_board', 'colander', 'strainer',
            'measuring_cup', 'measuring_spoon', 'timer', 'thermometer',

            # Coffee/Tea Equipment
            'moka_pot', 'coffee_maker', 'espresso_machine', 'french_press',
            'tea_infuser', 'coffee_grinder',

            # Cleaning
            'sponge', 'towel', 'cloth', 'rag', 'soap', 'detergent',
            'brush', 'scrubber',

            # Living Room Items
            'sofa', 'couch', 'tv', 'television', 'remote', 'lamp', 'pillow',
            'cushion', 'blanket', 'rug', 'carpet', 'curtain', 'picture', 'frame',
            'book', 'magazine', 'newspaper', 'vase', 'plant', 'flower',

            # Study/Office Items
            'pen', 'pencil', 'paper', 'notebook', 'folder', 'binder',
            'stapler', 'tape', 'scissors', 'ruler', 'eraser', 'marker',
            'highlighter', 'calculator', 'computer', 'laptop', 'keyboard',
            'mouse', 'monitor', 'printer', 'phone', 'tablet',
            'caddy', 'organizer', 'file',

            # Common Objects
            'ball', 'toy', 'game', 'puzzle', 'card', 'dice',
            'key', 'wallet', 'purse', 'bag', 'backpack', 'luggage',
            'clock', 'watch', 'calendar', 'mirror', 'brush', 'comb',

            # Tools & Hardware
            'hammer', 'screwdriver', 'wrench', 'pliers', 'drill',
            'nail', 'screw', 'bolt', 'nut', 'washer',

            # Electronics
            'charger', 'cable', 'adapter', 'battery', 'speaker',
            'headphone', 'earbud', 'camera', 'remote_control',

            # Miscellaneous
            'door', 'window', 'handle', 'knob', 'switch', 'button',
            'hook', 'hanger', 'clip', 'pin', 'magnet', 'sticker',
        }

        # Color modifiers
        self.colors = {
            'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange',
            'purple', 'pink', 'brown', 'gray', 'grey', 'silver', 'gold',
            'beige', 'tan', 'navy', 'maroon', 'teal', 'cyan', 'magenta'
        }

        # Material modifiers
        self.materials = {
            'wooden', 'metal', 'plastic', 'glass', 'ceramic', 'paper',
            'cloth', 'leather', 'rubber', 'steel', 'aluminum', 'wood'
        }

        # Spatial/location words to filter out
        self.stop_words = {
            'top', 'bottom', 'middle', 'left', 'right', 'front', 'back',
            'between', 'next', 'center', 'inside', 'outside', 'on', 'in', 
            'of', 'the', 'and', 'a', 'an', 'to', 'from', 'at', 'by',
            'near', 'close', 'far', 'above', 'below', 'under', 'over',
            'beside', 'behind', 'around', 'through', 'into', 'onto'
        }

        print(f"Object Knowledge Base initialized with {len(self.object_knowledge_base)} objects")

    def normalize_text(self, text: str) -> str:
        """Normalize text by replacing underscores and converting to lowercase"""
        return text.lower().replace('_', ' ')

    def extract_objects_from_text(self, text: str) -> List[str]:
        """
        Extract objects using pattern matching and knowledge base lookup
        Args:
            text: Task description string
        Returns:
            List of extracted object names
        """
        text = self.normalize_text(text)
        extracted_objects = set()

        # Strategy 1: Direct match with knowledge base (handles compound words)
        for obj in self.object_knowledge_base:
            obj_normalized = obj.replace('_', ' ')
            if obj_normalized in text:
                extracted_objects.add(obj)

        # Strategy 2: Extract "the [adjective]* [object]" patterns
        # Pattern: "the" followed by optional color/material, then object
        pattern = r'the\s+(?:(' + '|'.join(self.colors | self.materials) + r')\s+)?(\w+(?:\s+\w+)?)'
        matches = re.findall(pattern, text)

        for modifier, obj_phrase in matches:
            # Check each word against knowledge base
            words = obj_phrase.split()

            # Check full phrase first
            if obj_phrase in self.object_knowledge_base:
                if modifier:  # If there's a color/material modifier
                    extracted_objects.add(f"{modifier} {obj_phrase}")
                else:
                    extracted_objects.add(obj_phrase)

            # Check individual words
            for word in words:
                if word in self.object_knowledge_base and word not in self.stop_words:
                    if modifier:
                        extracted_objects.add(f"{modifier} {word}")
                    else:
                        extracted_objects.add(word)

        # Strategy 3: Extract multi-word object names (like "moka pot", "wine bottle")
        # Check for two-word combinations in knowledge base
        words = text.split()
        for i in range(len(words) - 1):
            two_word = f"{words[i]} {words[i+1]}"
            # Remove "the" prefix if present
            two_word_clean = two_word.replace('the ', '')
            if two_word_clean.replace(' ', '_') in self.object_knowledge_base:
                extracted_objects.add(two_word_clean.replace(' ', '_'))

        return list(extracted_objects)

    def clean_extractions(self, objects: List[str]) -> List[str]:
        """
        Clean extracted objects to remove false positives
        Args:
            objects: List of extracted object names
        Returns:
            Cleaned list of objects
        """
        cleaned = []
        for obj in objects:
            obj_clean = obj.strip()

            # Skip if empty or too short
            if not obj_clean or len(obj_clean) < 2:
                continue

            # Skip if it's just a stop word
            if obj_clean in self.stop_words:
                continue

            # Skip if it ends with a stop word (like "bowl between")
            words = obj_clean.split()
            if len(words) > 1 and words[-1] in self.stop_words:
                # Try to salvage by removing the stop word
                obj_clean = ' '.join(words[:-1])
                if not obj_clean or obj_clean in self.stop_words:
                    continue

            # Verify at least one word is in knowledge base
            has_valid_object = False
            for word in obj_clean.replace('_', ' ').split():
                if word in self.object_knowledge_base or obj_clean.replace(' ', '_') in self.object_knowledge_base:
                    has_valid_object = True
                    break

            if has_valid_object:
                cleaned.append(obj_clean)

        return list(set(cleaned))  # Remove duplicates

    def extract_from_text(self, text: str) -> List[str]:
        """
        Main extraction method with cleaning
        Args:
            text: Task description string
        Returns:
            List of unique extracted object names
        """
        objects = self.extract_objects_from_text(text)
        cleaned = self.clean_extractions(objects)
        return sorted(cleaned)

    def extract_from_text_list(self, texts: List[str]) -> List[str]:
        """
        Extract objects from a list of task descriptions
        Args:
            texts: List of task description strings
        Returns:
            List of all unique extracted object names
        """
        all_objects = set()
        for text in texts:
            objects = self.extract_from_text(text)
            all_objects.update(objects)

        return sorted(list(all_objects))

# Test function
if __name__ == "__main__":
    extractor = ObjectExtractor()

    test_tasks = [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "put_the_cream_cheese_in_the_bowl",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    ]

    print("\nTesting Improved Object Extractor:")
    print("="*70)
    for task in test_tasks:
        objects = extractor.extract_from_text(task)
        print(f"\nTask: {task}")
        print(f"Extracted Objects: {objects}")
