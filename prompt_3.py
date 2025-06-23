import openai
import json
import os
from typing import List
from openai import OpenAI

# ✅ OpenAI client 설정
client = OpenAI(
    api_key=""
)

# ✅ GPT로 프롬프트 생성
def generate_prompts(class_name: str, description: str = "", n: int = 5) -> List[str]:
    system_msg = (
        "You are a vision-language prompt generator for a model like CLIP or SigLIP.\n"
        "Generate vivid, visual, distinctive text prompts that highlight the unique appearance of each defect class.\n"
        "Avoid generic language, and emphasize features that distinguish this class from similar ones.\n"
        "Each prompt should start with 'A close-up photo of ...'."
    )

    user_msg = (
        f"Class: {class_name}\n"
        f"Description: {description}\n\n"
        f"Generate {n} unique and detailed visual prompts that describe this defect clearly and distinguish it from others like patches or rolled-in scale.\n"
        f"Focus on patterns, textures, and defects visible on the steel surface."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.8
    )

    output = response.choices[0].message.content
    prompts = [line.strip("- ").strip() for line in output.strip().split("\n") if line.strip()]
    return prompts[:n]

# ✅ 프롬프트 배치 생성
def batch_generate_prompts(class_descriptions: dict, save_path: str = "output/generated_prompts_4.json"):
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
        "crazing": "A dense network of fine, intersecting cracks caused by thermal stress. Resembles spiderweb patterns.",
        "patches": "Blurry or dark areas with soft edges, irregularly shaped and often low-contrast compared to surrounding surface.",
        "pitted_surface": "Numerous tiny pits or dots scattered over a lighter background due to surface corrosion.",
        "rolled-in_scale": "Parallel, sharp-edged, dark scale lines embedded during hot rolling. Often repetitive.",
        "scratches": "Long, thin, bright or white streaks sharply cutting across the surface, usually straight and reflective.",
        "inclusion": "Dark, irregular blobs or smears caused by embedded non-metallic materials, not linear and often deep-looking."
    }

    batch_generate_prompts(class_descriptions)
