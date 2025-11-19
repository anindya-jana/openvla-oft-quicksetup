from flask import Flask, render_template, jsonify, request, send_file
import random
import subprocess
from pathlib import Path
import re

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
SAVED_IMAGES_DIR = BASE_DIR / "saved_images"

LIBERO_TASK_MAP = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_r",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_pla",
        "pick_up_the_black_bowl_from_table_center_and_place",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wo",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_o",
        "pick_up_the_black_bowl_next_to_the_plate_and_place",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_p",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_bask",
        "pick_up_the_cream_cheese_and_place_it_in_the_baske",
        "pick_up_the_salad_dressing_and_place_it_in_the_bas",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_baske",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_",
        "pick_up_the_orange_juice_and_place_it_in_the_baske",
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
        "put_both_the_alphabet_soup_and_the_tomato_sauce_in",
        "put_both_the_cream_cheese_box_and_the_butter_in_th",
        "turn_on_the_stove_and_put_the_moka_pot_on_it",
        "put_the_black_bowl_in_the_bottom_drawer_of_the_cab",
        "put_the_white_mug_on_the_left_plate_and_put_the_ye",
        "pick_up_the_book_and_place_it_in_the_back_compartm",
        "put_the_white_mug_on_the_plate_and_put_the_chocola",
        "put_both_the_alphabet_soup_and_the_cream_cheese_bo",
        "put_both_moka_pots_on_the_stove",
        "put_the_yellow_and_white_mug_in_the_microwave_and_",
    ],
}

FULL_TASK_DESCRIPTIONS = {
    "libero_spatial": {
        "pick_up_the_black_bowl_between_the_plate_and_the_r": "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_pla": "pick up the black bowl next to the ramekin and place it on the plate",
        "pick_up_the_black_bowl_from_table_center_and_place": "pick up the black bowl from table center and place it on the plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place": "pick up the black bowl on the cookie box and place it on the plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wo": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it": "pick up the black bowl on the ramekin and place it on the plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_": "pick up the black bowl next to the cookie box and place it on the plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_o": "pick up the black bowl on the stove and place it on the plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place": "pick up the black bowl next to the plate and place it on the plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_p": "pick up the black bowl on the wooden cabinet and place it on the plate",
    },
    "libero_object": {
        "pick_up_the_alphabet_soup_and_place_it_in_the_bask": "pick up the alphabet soup and place it in the basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_baske": "pick up the cream cheese and place it in the basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_bas": "pick up the salad dressing and place it in the basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket": "pick up the bbq sauce and place it in the basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket": "pick up the ketchup and place it in the basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_baske": "pick up the tomato sauce and place it in the basket",
        "pick_up_the_butter_and_place_it_in_the_basket": "pick up the butter and place it in the basket",
        "pick_up_the_milk_and_place_it_in_the_basket": "pick up the milk and place it in the basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_": "pick up the chocolate pudding and place it in the basket",
        "pick_up_the_orange_juice_and_place_it_in_the_baske": "pick up the orange juice and place it in the basket",
    },
    "libero_goal": {
        "open_the_middle_drawer_of_the_cabinet": "open the middle drawer of the cabinet",
        "put_the_bowl_on_the_stove": "put the bowl on the stove",
        "put_the_wine_bottle_on_top_of_the_cabinet": "put the wine bottle on top of the cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside": "open the top drawer and put the bowl inside",
        "put_the_bowl_on_top_of_the_cabinet": "put the bowl on top of the cabinet",
        "push_the_plate_to_the_front_of_the_stove": "push the plate to the front of the stove",
        "put_the_cream_cheese_in_the_bowl": "put the cream cheese in the bowl",
        "turn_on_the_stove": "turn on the stove",
        "put_the_bowl_on_the_plate": "put the bowl on the plate",
        "put_the_wine_bottle_on_the_rack": "put the wine bottle on the rack",
    },
    "libero_10": {
        "put_both_the_alphabet_soup_and_the_tomato_sauce_in": "put both the alphabet soup and the tomato sauce in the basket",
        "put_both_the_cream_cheese_box_and_the_butter_in_th": "put both the cream cheese box and the butter in the basket",
        "turn_on_the_stove_and_put_the_moka_pot_on_it": "turn on the stove and put the moka pot on it",
        "put_the_black_bowl_in_the_bottom_drawer_of_the_cab": "put the black bowl in the bottom drawer of the cabinet and close it",
        "put_the_white_mug_on_the_left_plate_and_put_the_ye": "put the white mug on the left plate and put the yellow and white mug on the right plate",
        "pick_up_the_book_and_place_it_in_the_back_compartm": "pick up the book and place it in the back compartment of the caddy",
        "put_the_white_mug_on_the_plate_and_put_the_chocola": "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        "put_both_the_alphabet_soup_and_the_cream_cheese_bo": "put both the alphabet soup and the cream cheese box in the basket",
        "put_both_moka_pots_on_the_stove": "put both moka pots on the stove",
        "put_the_yellow_and_white_mug_in_the_microwave_and_": "put the yellow and white mug in the microwave and close it",
    },
}

def get_suite_display_name(suite_key):
    names = {
        "libero_spatial": "Libero spatial",
        "libero_object": "Libero object",
        "libero_goal": "Libero goal",
        "libero_10": "Libero 10"
    }
    return names.get(suite_key, suite_key)

def find_task_image(suite, task_folder):
    suite_display = get_suite_display_name(suite)
    task_path = SAVED_IMAGES_DIR / suite_display / task_folder / "episode_0000" / "full_image_step_0000.png"
    if task_path.exists():
        return f"images/{task_path.relative_to(BASE_DIR)}"
    return None

def get_all_tasks_with_images():
    tasks_data = []
    for suite, tasks in LIBERO_TASK_MAP.items():
        for idx, task_folder in enumerate(tasks):
            image_path = find_task_image(suite, task_folder)
            if image_path:
                desc = FULL_TASK_DESCRIPTIONS.get(suite, {}).get(task_folder, task_folder.replace("_", " "))
                tasks_data.append({
                    "suite": suite,
                    "task_id": idx,
                    "folder": task_folder,
                    "description": desc,
                    "image": image_path
                })
    return tasks_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/random-task')
def random_task():
    tasks = get_all_tasks_with_images()
    if tasks:
        task = random.choice(tasks)
        return jsonify(task)
    return jsonify({"error": "No tasks found"}), 404

@app.route('/api/run-analysis', methods=['POST'])
def run_analysis():
    data = request.json
    description = data.get('description')
    image_path = data.get('image')
    if not all([description, image_path]):
        return jsonify({"error": "Missing parameters"}), 400
    if image_path.startswith("images/"):
        image_path = image_path[len("images/"):]
    full_image_path = Path(__file__).parent / image_path
    if not full_image_path.exists():
        return jsonify({"error": f"Image not found: {full_image_path}"}), 404

    try:
        cmd = (
            f"conda run -n openvla-oft1 python main.py "
            f'--text "{description}" '
            f'--image "{str(full_image_path)}"'
        )
        result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent, capture_output=True, text=True, timeout=60)
        
        # Extract meaningful line from output
        lines = result.stdout.split('\n')
        classifier_line = ""
        for line in lines:
            if "Task classified as:" in line:
                classifier_line = line.strip()
                break
        
        return jsonify({
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "classifier_result": classifier_line  # Add extracted result
        })
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Analysis timed out"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-simulation', methods=['POST'])
def run_simulation():
    CATEGORY_MAP = {
        "LIBERO_OBJECT": ("libero_object", "moojink/openvla-7b-oft-finetuned-libero-object"),
        "LIBERO_SPATIAL": ("libero_spatial", "moojink/openvla-7b-oft-finetuned-libero-spatial"),
        "LIBERO_GOAL": ("libero_goal", "moojink/openvla-7b-oft-finetuned-libero-goal"),
        "LIBERO_10": ("libero_10", "moojink/openvla-7b-oft-finetuned-libero-10"),
    }
    
    data = request.json
    classifier_output = data.get('classifier_output', '')
    original_description = data.get('description', '')
    
    match = re.search(r"Task classified as:\s*([A-Z_]+)", classifier_output)
    if not match:
        return jsonify({"error": "Could not parse classifier output"}), 400
    
    category = match.group(1)
    predicted_suite, predicted_ckpt = CATEGORY_MAP.get(category, (None, None))
    if not predicted_suite or not predicted_ckpt:
        return jsonify({"error": "Unknown classifier category"}), 400
    
    task_id = None
    actual_suite = predicted_suite
    actual_ckpt = predicted_ckpt
    
    suite_tasks_map = FULL_TASK_DESCRIPTIONS.get(predicted_suite, {})
    for folder, desc in suite_tasks_map.items():
        if desc == original_description:
            if folder in LIBERO_TASK_MAP[predicted_suite]:
                task_id = LIBERO_TASK_MAP[predicted_suite].index(folder)
                break
    
    if task_id is None:
        for suite_name in ["libero_object", "libero_spatial", "libero_goal", "libero_10"]:
            if suite_name == predicted_suite:
                continue
            
            suite_tasks_map = FULL_TASK_DESCRIPTIONS.get(suite_name, {})
            for folder, desc in suite_tasks_map.items():
                if desc == original_description:
                    if folder in LIBERO_TASK_MAP[suite_name]:
                        task_id = LIBERO_TASK_MAP[suite_name].index(folder)
                        actual_suite = suite_name
                        for cat_key, (cat_suite, cat_ckpt) in CATEGORY_MAP.items():
                            if cat_suite == suite_name:
                                actual_ckpt = cat_ckpt
                                break
                        break
            
            if task_id is not None:
                break
    
    if task_id is None:
        return jsonify({"error": f"Could not find task with description: '{original_description}' in any suite"}), 400

    sim_cmd = (
        f"conda run -n openvla-oft bash -lc 'export PYTHONPATH=\"$PWD:$PWD/LIBERO\"; "
        f"export HF_HOME=\"$PWD/hf-cache\"; export TRANSFORMERS_CACHE=\"$PWD/hf-cache\"; "
        f"python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint {actual_ckpt} "
        f"--task_suite_name {actual_suite} --single_task_id {task_id} --num_trials_per_task 1 "
        f"--use_wandb False --center_crop True --onscreen_render True'"
    )
    result = subprocess.run(sim_cmd, shell=True, cwd="/home/server/openvla-oft", capture_output=True, text=True, timeout=180)
    
    success_match = re.search(r"Overall success rate:\s*([\d.]+)", result.stdout)
    success_rate = success_match.group(1) if success_match else "N/A"
    
    return jsonify({
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "used_suite": actual_suite,
        "predicted_suite": predicted_suite,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success_rate": success_rate
    })

@app.route('/images/<path:filepath>')
def serve_image(filepath):
    image_path = Path(__file__).parent / filepath
    if image_path.exists():
        return send_file(image_path)
    return "Image not found", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
