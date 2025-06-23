import openai
import json
import os
from typing import List, Dict
from openai import OpenAI

# ✅ OpenAI API 설정
client = OpenAI(
    api_key=""
)

# ✅ 클래스별 설명 및 혼동 방지 전략
CLASS_DETAILS = {
    "crazing": {
        "description": "A dense network of fine, web-like micro-cracks on the steel surface.",
        "confused_with": "'patches', 'scratches'",
        "distinguishing_features": "its intricate, interconnected fractal-like pattern that covers the surface like a spiderweb. Avoid describing isolated lines or rough blobs."
    },
    "scratches": {
        "description": "Straight, narrow, and clean lines from physical abrasion.",
        "confused_with": "'crazing', 'pitted_surface'",
        "distinguishing_features": "its single or multiple linear streaks with high contrast, clearly not connected or branched. Avoid descriptions that imply web or crack patterns."
    },
    "patches": {
        "description": "Discolored or fuzzy areas that lack a consistent shape or edge.",
        "confused_with": "'pitted_surface', 'inclusion'",
        "distinguishing_features": "its soft, fuzzy, cloud-like regions with gradual boundary transitions. Avoid any mention of holes or distinct pits."
    },
    "pitted_surface": {
        "description": "Steel surface with clusters of small, round pits from corrosion.",
        "confused_with": "'patches', 'scratches'",
        "distinguishing_features": "its scattered circular holes with defined edges. Suggest using side lighting to emphasize shadows inside pits."
    },
    "rolled-in_scale": {
        "description": "Linear oxide marks pressed into the steel from scale residue.",
        "confused_with": "'patches', 'inclusion'",
        "distinguishing_features": "its repetitive, linear textured bands or streaks with embedded particles. Differentiate from blotches or speckles."
    },
    "inclusion": {
        "description": "Dark, irregularly shaped internal impurities visible on surface.",
        "confused_with": "'patches', 'pitted_surface'",
        "distinguishing_features": "its non-surface nature, often appearing as deep, irregular dark spots with sharper contrast. Avoid describing surface-level smears or fuzziness."
    }
}

# ✅ 고급 프롬프트 생성 함수
def generate_ultimate_prompts(class_name: str, details: Dict, n: int = 5) -> List[str]:
    system_msg = (
        "You are an expert metallurgical engineer specializing in steel surface defect analysis. "
        "Your task is to generate precise and vivid visual descriptions for a vision-language model, "
        "based on a defect's core concept and a proven differentiation strategy."
    )

    user_msg = (
        f"Generate {n} distinct, high-quality visual prompts for the defect class: '{class_name}'.\n\n"
        f"**Core Concept:** {details['description']}\n\n"
        f"**Proven Winning Strategy:** The model often confuses '{class_name}' with {details['confused_with']}. "
        f"Therefore, all prompts MUST emphasize the key distinguishing features: {details['distinguishing_features']}.\n\n"
        f"Ensure each prompt starts with 'A photo of a steel surface showing ...' and uses diverse, vivid language."
    )

    print(f"▶ Generating ultimate prompts for: {class_name}...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7,
        n=1
    )

    output = response.choices[0].message.content
    prompts = [
        line[line.lower().find('a photo of'):] for line in output.strip().split("\n")
        if 'a photo of' in line.lower()
    ]
    return prompts[:n] if prompts else []

# ✅ 전체 프롬프트 저장

def batch_generate_and_save(class_details: dict, save_filename: str):
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, save_filename)
    all_prompts = {}
    for cls, details in class_details.items():
        prompts = generate_ultimate_prompts(cls, details)
        if not prompts:
            print(f"❗️ Warning: No prompts were generated for class '{cls}'. Skipping.")
            continue
        all_prompts[cls] = prompts
        for p in prompts:
            print(f"  - {p}")
        print()

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)
    print(f"✅ 최종 프롬프트 저장 완료: {save_path}")

if __name__ == "__main__":
    new_prompt_filename = "generated_prompts_v4.json"
    batch_generate_and_save(CLASS_DETAILS, new_prompt_filename)
