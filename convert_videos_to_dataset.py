import os
import csv
import json
from pathlib import Path

# --- Config ---
VIDEO_BASE = "/data4/luowei/videos"
CSV_PATH = os.path.join(VIDEO_BASE, "4dworldbench/phygenbench_prompts.csv")
OUT_DIR = os.path.join(VIDEO_BASE, "dataset_jsons")
PROMPT_TXT_DIR = os.path.join(VIDEO_BASE, "prompts_txt")

# Models under 4dworldbench (dir name -> model_name for output)
MODELS_4DWB = {
    "wisa": "wisa",
    "wan_1_3b": "wan_1_3b",
    "wan_14b": "wan_14b",
    "cogvideox": "cogvideox",
    "ours_1_3b": "ours_1_3b",
    "ours_1_3bs2_720p": "ours_1_3bs2_720p",
    "ours_14b": "ours_14b",
    "ours_14bs2_720p": "ours_14bs2_720p",
    "videorepa": "videorepa",
}
HUNYUAN_MODEL = "hunyuan"


def main():
    # 1. Read CSV
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Read {len(rows)} prompts from CSV")

    # 2. Create txt files for each prompt and group by states_of_matter
    os.makedirs(PROMPT_TXT_DIR, exist_ok=True)

    groups = {}  # states_of_matter -> list of prompt dicts
    for idx, row in enumerate(rows):
        num = idx + 1  # 1-based: 0001 ~ 0160
        num_str = f"{num:04d}"
        caption = row["caption"].strip()
        som = row["states_of_matter"].strip()  # solid_fluid, solid_solid, fluid_fluid

        # Write caption to txt file
        txt_name = f"{num_str}.txt"
        txt_path = os.path.join(PROMPT_TXT_DIR, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)

        # Extract a short key from video_url (e.g., "001_A_cup_of_water...")
        video_url = row.get("video_url", "")
        basename = os.path.splitext(os.path.basename(video_url))[0] if video_url else num_str

        prompt_entry = {
            "prompt_id": num_str,
            "prompt_key": basename,
            "condition_content": txt_path,
            "condition_caption": caption,
            "generated_videos": [],  # filled per-model
        }

        groups.setdefault(som, []).append((num_str, prompt_entry))

    print(f"Groups by states_of_matter: { {k: len(v) for k, v in groups.items()} }")

    # 3. Build dataset JSON for 4dworldbench models
    model_dirs = {}
    for dir_name, model_name in MODELS_4DWB.items():
        dir_path = os.path.join(VIDEO_BASE, "4dworldbench", dir_name)
        if os.path.isdir(dir_path):
            model_dirs[model_name] = dir_path

    # Build for hunyuan too
    hunyuan_dir = os.path.join(VIDEO_BASE, "hunyuan")
    if os.path.isdir(hunyuan_dir):
        model_dirs[HUNYUAN_MODEL] = hunyuan_dir

    all_models = build_models(model_dirs, groups)

    # 4. Write output
    dataset_info = {
        "base_path": VIDEO_BASE,
        "model_type": "text-to-4D",
        "condition_type": "text",
    }

    dataset = {
        "dataset_info": dataset_info,
        "models": all_models,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "phygenbench_text_to_4d.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nSaved dataset JSON to: {out_path}")
    print(f"Models: {[m['model_name'] for m in all_models]}")
    for m in all_models:
        total_prompts = sum(len(c["prompts"]) for c in m["conditions"])
        total_videos = sum(
            len(p["generated_videos"]) for c in m["conditions"] for p in c["prompts"]
        )
        print(f"  {m['model_name']}: {total_prompts} prompts, {total_videos} videos")


def build_models(model_dirs, groups):
    models = []
    for model_name, model_dir in sorted(model_dirs.items()):
        conditions = []
        total_videos = 0

        for som, prompt_list in groups.items():
            prompts_out = []
            for num_str, template in prompt_list:
                mp4_name = f"{num_str}.mp4"
                video_path = os.path.join(model_dir, mp4_name)
                if not os.path.exists(video_path):
                    video_path = os.path.join(model_dir, "videos", mp4_name)

                if os.path.exists(video_path):
                    videos = [video_path]
                    total_videos += 1
                else:
                    videos = []

                prompt_out = dict(template)
                prompt_out["generated_videos"] = videos
                prompts_out.append(prompt_out)

            conditions.append({
                "condition_meta_info": som,  # solid_fluid / solid_solid / fluid_fluid
                "prompts": prompts_out,
            })

        models.append({
            "model_name": model_name,
            "conditions": conditions,
        })

    return models


if __name__ == "__main__":
    main()
