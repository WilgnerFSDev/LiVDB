import json
import os
import re
import pandas as pd
from scripts.dataset_builder import calculate_hash, calculate_size
from scripts.ui import UI

def remove_comments(code):
    code = code.replace('\r\n', '\n').replace('\r', '\n')

    code = re.sub(r'/\*[\s\S]*?\*/', '', code)

    code = re.sub(r'//.*', '', code)

    lines = []
    for line in code.split('\n'):
        if line.strip():
            lines.append(line.rstrip())

    return "\n".join(lines).strip()

def convert_devign_json(input_path, output_name="dataset_devign.csv"):
    if not os.path.exists(input_path):
        UI.error(f"File not found: {input_path}")
        return

    UI.info(f"Reading JSON dataset: {input_path}...")

    data = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.splitlines()]
    except Exception as e:
        UI.error(f"Error reading JSON: {e}")
        return

    total_raw = len(data)
    vuln_raw = sum(1 for item in data if int(item.get('target', 0)) == 1)
    clean_raw = total_raw - vuln_raw
    projects_raw = len(set(item.get('project', 'unknown') for item in data))

    perc_vuln_raw = (vuln_raw / total_raw) * 100 if total_raw > 0 else 0
    perc_clean_raw = (clean_raw / total_raw) * 100 if total_raw > 0 else 0

    UI.header("Dataset Stats: Before Processing")
    print(f"Total Functions   : {total_raw}")
    print(f"Vulnerable (1)    : {vuln_raw} ({perc_vuln_raw:.2f}%)")
    print(f"Non-Vulnerable (0): {clean_raw} ({perc_clean_raw:.2f}%)")
    print(f"Unique Projects   : {projects_raw}")
    print("-" * 40)

    processed_rows = []
    total = len(data)
    vuln_count = 0
    clean_count = 0

    UI.info(f"Processing {total} functions (Removing comments and calculating hashes)...")

    for item in data:
        raw_code = item.get('func', '')
        if not raw_code:
            continue

        clean_code = remove_comments(raw_code)
        if not clean_code:
            continue

        label = int(item.get('target', 0))
        if label == 1:
            vuln_count += 1
        else:
            clean_count += 1

        commit_id = item.get('commit_id', 'unknown')
        project = item.get('project', 'unknown')

        processed_rows.append({
            "cve_id": f"DEVIGN-{project.upper()}-{commit_id[:8]}",
            "filename": f"{project}_func.c",
            "hash": calculate_hash(clean_code),
            "lines": calculate_size(clean_code),
            "label": label,
            "function": clean_code
        })

    os.makedirs("datasets", exist_ok=True)
    output_path = os.path.join("datasets", output_name)
    df = pd.DataFrame(processed_rows)
    df.to_csv(output_path, index=False)

    total_final = len(processed_rows)
    projects_final_list = set()
    for row in processed_rows:
        parts = row['cve_id'].split('-')
        if len(parts) >= 2:
            projects_final_list.add(parts[1])

    projects_final = len(projects_final_list)
    perc_vuln = (vuln_count / total_final) * 100 if total_final > 0 else 0
    perc_clean = (clean_count / total_final) * 100 if total_final > 0 else 0

    UI.success(f"Conversion finished! Saved to: {output_path}")

    UI.header("Dataset Stats: After Processing (Final CSV)")
    print(f"Total Functions   : {total_final}")
    print(f"Vulnerable (1)    : {vuln_count} ({perc_vuln:.2f}%)")
    print(f"Non-Vulnerable (0): {clean_count} ({perc_clean:.2f}%)")
    print(f"Unique Projects   : {projects_final}")
    if projects_final > 0:
        print(f"Projects found    : {', '.join(sorted(projects_final_list))}")
    print("-" * 40)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    elif os.path.exists("datasets_reveal_devign/Devign/function.json"):
        target_file = "datasets_reveal_devign/Devign/function.json"
    elif os.path.exists("datasets_reveal_devign/function.json"):
        target_file = "datasets_reveal_devign/function.json"
    elif os.path.exists("devign_function_dataset.json"):
        target_file = "devign_function_dataset.json"
    else:
        target_file = "datasets_reveal_devign/Devign/function.json"

    if os.path.exists(target_file):
        convert_devign_json(target_file)
    else:
        UI.warning(f"File not found: '{target_file}'")
