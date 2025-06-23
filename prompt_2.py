import openai
import json
import os
from typing import List
from openai import OpenAI

# ✅ OpenAI client 설정
client = OpenAI(
    api_key=""
)

# ✅ 프롬프트 생성 함수
def generate_prompts(class_name: str, description: str = "", n: int = 5) -> List[str]:
    system_msg = (
        "You are an expert prompt engineer for a vision-language model (e.g., CLIP, SigLIP). "
        "Given a surface defect class and short description, write vivid, discriminative visual descriptions "
        "for image classification. Avoid ambiguity with other classes. Each prompt must start with: 'A photo of ...'."
    )

    user_msg = (
        f"Class: {class_name}\n"
        f"Description: {description}\n\n"
        f"Generate {n} distinct, highly discriminative visual prompts for this class "
        f"that help avoid confusion with similar defects."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.5  # 낮춰서 일관성 ↑
    )

    output = response.choices[0].message.content
    prompts = [line.strip("- ").strip() for line in output.strip().split("\n") if line.strip()]
    return prompts[:n]

# ✅ 모든 클래스에 대해 프롬프트 생성
def batch_generate_prompts(class_descriptions: dict, save_path: str = "output/generated_prompts_expert.json"):
    base, ext = os.path.splitext(save_path)
    counter = 1
    final_save_path = save_path
    while os.path.exists(final_save_path):
        final_save_path = f"{base}_{counter}{ext}"
        counter += 1

    all_prompts = {}
    for cls, desc in class_descriptions.items():
        print(f"▶ Generating prompts for: {cls}")
        prompts = generate_prompts(cls, desc)
        all_prompts[cls] = prompts
        for p in prompts:
            print(f"  - {p}")
        print()

    with open(final_save_path, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)
    print(f"✅ 저장 완료: {final_save_path}")

# ✅ 클래스 설명
if __name__ == "__main__":
    class_descriptions = {
        "crazing": "A network of fine cracks across the steel surface, forming spiderweb-like patterns.",
        "patches": "Fuzzy-edged blotches or irregular surface stains that differ in brightness from the surrounding area.",
        "pitted_surface": "Clusters of small circular pits or indentations resembling corrosion damage.",
        "rolled-in_scale": "Long, dark, repetitive parallel lines caused by rolled-in mill scale during hot rolling.",
        "scratches": "Straight, sharp, light-colored lines of varying lengths caused by mechanical scraping.",
        "inclusion": "Irregular, dark-spot defects embedded beneath or on the surface, caused by non-metallic particles."
    }

    batch_generate_prompts(class_descriptions)
