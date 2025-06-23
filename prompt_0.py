import openai
import json
import os
from typing import List

from openai import OpenAI

# ✅ OpenAI client 설정
client = OpenAI(
    api_key="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"
)

# ✅ GPT로 프롬프트 생성 함수
def generate_prompts(class_name: str, description: str = "", n: int = 5) -> List[str]:
    system_msg = (
        "You are a visual prompt generator for a vision-language model like CLIP. "
        "Given a class name and optionally a brief defect description, "
        "generate detailed, vivid, visual descriptions suitable as text prompts for classification."
    )

    user_msg = (
        f"Class: {class_name}\n"
        f"Description: {description}\n\n"
        f"Generate {n} distinct, high-quality visual prompts for this class, "
        f"written in natural English, each starting with 'A photo of ...'."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7
    )

    output = response.choices[0].message.content
    prompts = [line.strip("- ").strip() for line in output.strip().split("\n") if line.strip()]
    return prompts[:n]


# ✅ 배치로 모든 클래스에 대해 프롬프트 생성 후 저장
def batch_generate_prompts(class_descriptions: dict, save_path: str = "/Users/suin/Desktop/SigLIP 12/output/generated_prompts.json"):
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


if __name__ == "__main__":
    class_descriptions = {
        "crazing": "Fine cracks appearing on the surface of steel, usually due to thermal stress.",
        "patches": "Irregular surface defects on steel that appear as discolored or rough patches.",
        "pitted_surface": "Corrosion-induced small holes or pits on the metal sheet surface.",
        "rolled-in_scale": "Oxide scale defects pressed into hot-rolled steel surfaces.",
        "scratches": "Linear abrasions or marks on the surface of steel sheets.",
        "inclusion": "Non-metallic particles trapped in steel, appearing as irregular dark spots."
    }

    batch_generate_prompts(class_descriptions)
