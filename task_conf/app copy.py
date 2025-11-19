from flask import Flask, render_template, jsonify, request, send_file
import json
import random
import subprocess
from pathlib import Path
import re

app = Flask(__name__)

# Base paths
BASE_DIR = Path(__file__).parent
SAVED_IMAGES_DIR = BASE_DIR / "saved_images"

# The task map and descriptions simplified for example
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
    # Add task description per your previous data; omitted here for brevity
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

def extract_summary(log_text):
    summary_lines = []
    lines = log_text.splitlines()
    capture = False
    for line in lines:
        if line.startswith("Saved rollout MP4 at path"):
            capture = True
        if capture:
            summary_lines.append(line)
    return "\n".join(summary_lines) if summary_lines else "No summary found."

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
        return jsonify({
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
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
    task_id = data.get('task_id')
    match = re.search(r"Task classified as:\s*([A-Z_]+)", classifier_output)
    if not (match and task_id is not None):
        return jsonify({"error": "Could not parse classifier output or missing task id"}), 400
    category = match.group(1)
    suite, ckpt = CATEGORY_MAP.get(category, (None, None))
    if not suite or not ckpt:
        return jsonify({"error": "Unknown classifier category"}), 400

    sim_cmd = (
        f"conda run -n openvla-oft bash -lc 'export PYTHONPATH=\"$PWD:$PWD/LIBERO\"; "
        f"export HF_HOME=\"$PWD/hf-cache\"; export TRANSFORMERS_CACHE=\"$PWD/hf-cache\"; "
        f"python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint {ckpt} "
        f"--task_suite_name {suite} --single_task_id {task_id} --num_trials_per_task 1 "
        f"--use_wandb False --center_crop True --onscreen_render True'"
    )
    result = subprocess.run(sim_cmd, shell=True, cwd="/home/server/openvla-oft", capture_output=True, text=True, timeout=180)
    summary = extract_summary(result.stdout)
    return jsonify({
        "success": True,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "summary": summary
    })

@app.route('/images/<path:filepath>')
def serve_image(filepath):
    image_path = Path(__file__).parent / filepath
    if image_path.exists():
        return send_file(image_path)
    return "Image not found", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
