import openai
import json
import os
from typing import List, Dict
from openai import OpenAI

# ✅ OpenAI API 설정
client = OpenAI(
    api_key=""
)

# ✅ 클래스 설명: 실제 이미지 특징과 비교 묘사 기반
CLASS_DETAILS = {
    "crazing": {
        "description": "A dense mesh of micro-cracks forming a spiderweb pattern over the surface.",
        "confused_with": "scratches, patches",
        "distinguishing_features": "the fine, branching, interconnected cracks, typically glossy under light. Emphasize its web-like complexity unlike straight lines or cloudy blobs."
    },
    "scratches": {
        "description": "Long, clean, narrow lines caused by physical abrasion.",
        "confused_with": "crazing, pitted_surface",
        "distinguishing_features": "straight, singular or parallel marks with high contrast. Avoid mentioning complexity or webbing."
    },
    "patches": {
        "description": "Blurry, darkened, or discolored areas that lack sharp edges.",
        "confused_with": "pitted_surface, inclusion",
        "distinguishing_features": "soft, smudgy or cloud-like regions disrupting the regular surface. No mention of holes or sharp spots."
    },
    "pitted_surface": {
        "description": "Numerous small, round pits scattered due to corrosion.",
        "confused_with": "patches, scratches",
        "distinguishing_features": "individual deep circular cavities casting shadows. Mention use of low-angle lighting to reveal pit depth."
    },
    "rolled-in_scale": {
        "description": "Oxide scale embedded into the surface during hot rolling.",
        "confused_with": "patches, inclusion",
        "distinguishing_features": "linear or streaky dark marks with embedded material. Differentiate from stains or isolated specks."
    },
    "inclusion": {
        "description": "Dark, irregular, non-metallic impurities trapped inside steel.",
        "confused_with": "patches, pitted_surface",
        "distinguishing_features": "sharp-edged, dark internal spots that contrast strongly against the background. Avoid fuzzy or shallow textures."
    }
}

# ✅ 이미지 반영 기반 프롬프트 생성
def generate_visual_prompts(class_name: str, details: Dict, n: int = 5) -> List[str]:
    system_msg = (
        "You are a metallurgical defect description expert. Your goal is to generate clear, vivid, image-based visual prompts "
        "that help a vision-language model distinguish this defect class from similar ones."
    )

    user_msg = (
        f"Class: {class_name}\n"
        f"Definition: {details['description']}\n"
        f"Often confused with: {details['confused_with']}\n"
        f"Key visual features to emphasize: {details['distinguishing_features']}\n\n"
        f"Generate {n} distinct, image-grounded visual prompts. Each must:\n"
        f"- Start with 'A photo of a steel surface showing...'\n"
        f"- Be concise but rich in visual detail\n"
        f"- Explicitly highlight what makes this class different from others\n"
        f"- Be written in natural English"
    )

    print(f"▶ Generating prompts for: {class_name}")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.6
    )

    output = response.choices[0].message.content
    prompts = [
        line[line.lower().find("a photo of"):] for line in output.strip().split("\n")
        if "a photo of" in line.lower()
    ]
    return prompts[:n] if prompts else []

# ✅ 전체 클래스 처리 및 저장
def batch_generate_and_save(class_details: dict, save_filename: str):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_filename)

    all_prompts = {}
    for cls, details in class_details.items():
        prompts = generate_visual_prompts(cls, details)
        all_prompts[cls] = prompts
        for p in prompts:
            print(f"  - {p}")
        print()

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)
    print(f"✅ 저장 완료: {save_path}")

# ✅ 메인 실행
if __name__ == "__main__":
    filename = "generated_prompts_v6.json"
    batch_generate_and_save(CLASS_DETAILS, filename)
