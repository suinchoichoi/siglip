import openai
import json
import os
from typing import List
from openai import OpenAI

# ✅ OpenAI client 설정 (API 키 직접 입력)
client = OpenAI(
    api_key="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"
)

# ✅ GPT로 프롬프트 생성 함수
def generate_prompts(class_name: str, description: str = "", n: int = 5) -> List[str]:
    system_msg = (
        "You are a visual prompt generator for a vision-language model like CLIP or SigLIP. "
        "Given a defect class name and a short description, generate vivid, realistic visual prompts "
        "for zero-shot classification. Each prompt should start with 'A photo of ...'."
    )

    user_msg = (
        f"Class: {class_name}\n"
        f"Description: {description}\n\n"
        f"Generate {n} distinct, high-quality visual prompts for this class."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7
    )

    output = response.choices[0].message.content
    prompts = [line.strip("- ").strip() for line in output.strip().split("\n") if line.strip()]
    return prompts[:n]


# ✅ 전체 클래스에 대해 프롬프트 생성
def batch_generate_prompts(class_descriptions: dict, save_path: str = "output/generated_prompts.json"):
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


# ✅ 클래스 설명 정의 및 실행
if __name__ == "__main__":
    class_descriptions = {
        "crazing": "A network of fine surface cracks caused by thermal stress, resembling spiderweb patterns.",
        "patches": "Irregular, darkened, or blurry regions on the steel surface, often with fuzzy edges.",
        "pitted_surface": "Dense fields of small, deep holes caused by corrosion, appearing scattered.",
        "rolled-in_scale": "Parallel, linear scale marks pressed into the steel during hot rolling.",
        "scratches": "Long, narrow, high-contrast marks on the surface due to mechanical abrasion.",
        "inclusion": "Randomly shaped dark spots caused by non-metallic impurities within the steel."
    }

    batch_generate_prompts(class_descriptions)
