# -*- coding: utf-8 -*-

from zhipuai import ZhipuAI
# import jsonlines
import ast
import re
import json
import random
import copy
from collections import defaultdict
import openai
import requests

# 移除原有的API Key，以保护私密数据
zhipu_api_key = ""
openrouter_key = ""
client = openai.OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
client_flash = ZhipuAI(api_key=zhipu_api_key)

def llm_judge(text, model = "zhipu"):

    if model == "zhipu":
        response = client_flash.chat.completions.create(
                model="glm-4.5-flash",
                messages=[
                    {"role": "user", "content": f'''You are a text analysis assistant.

Your task is to determine whether the following text contains **redundancy or repetition**.
The text may be written in **Chinese, English, or a mixture of both**.

Repetition can include either of the following types:

1. **Explicit repetition**
   - Repeated use of the same keywords, phrases, fields, or sentence structures
   - Repeating the same information with only minor surface changes

2. **Implicit repetition (paraphrasing)**
   - Different wording or sentence forms
   - But conveying essentially the same meaning, facts, or claims multiple times
   - No substantial new information is added

Please analyze the text carefully and answer in the following structured format.

---

### Text to analyze
{text}

---

### Output format (strictly follow this structure):

Please analyze the text and include <Yes> or <No> in your answer.

Representative examples (if applicable):
- Example 1: "本系统支持多种数据格式，包括CSV、JSON和XML。所以CSV、JSON、XML可以被该系统解析。所以该系统可以支持CSV、JSON、XML。"
- Yes
- Example 2: "The model is designed to improve efficiency. It helps users complete tasks faster. So, it improve the users' efficiency, and users complete the tasks faster."
- Yes
- Example 3: "所以由条件可知，矛盾；又因为菊花在最左边，所以由梅花在右边，矛盾；又因为菊花在最左边，所以由梅花在右边，矛盾；又因为菊花在最左边，所以由梅花在右边，矛盾"
- Yes
- Example 4: "The platform supports file upload and download and upload and download and upload and download and upload and download and"
- Yes

Your answer:
'''}
                    ],
                thinking={
                    "type": "disabled", 
                },
                stream=True,
                max_tokens=4096,
                temperature=0.7
            )

        full_text = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content

    else:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "openai/gpt-5.1",
                "messages": [
                    {"role": "user", "content": f'''You are a text analysis assistant.

Your task is to determine whether the following text contains **redundancy or repetition**.
The text may be written in **Chinese, English, or a mixture of both**.

Repetition can include either of the following types:

1. **Explicit repetition**
   - Repeated use of the same keywords, phrases, fields, or sentence structures
   - Repeating the same information with only minor surface changes

2. **Implicit repetition (paraphrasing)**
   - Different wording or sentence forms
   - But conveying essentially the same meaning, facts, or claims multiple times
   - No substantial new information is added

Please analyze the text carefully and answer in the following structured format.

---

### Text to analyze
{text}

---

### Output format (strictly follow this structure):

Please analyze the text and include <Yes> or <No> in your answer.

Representative examples (if applicable):
- Example 1: "本系统支持多种数据格式，包括CSV、JSON和XML。所以CSV、JSON、XML可以被该系统解析。所以该系统可以支持CSV、JSON、XML。"
- Yes
- Example 2: "The model is designed to improve efficiency. It helps users complete tasks faster. So, it improve the users' efficiency, and users complete the tasks faster."
- Yes
- Example 3: "所以由条件可知，矛盾；又因为菊花在最左边，所以由梅花在右边，矛盾；又因为菊花在最左边，所以由梅花在右边，矛盾；又因为菊花在最左边，所以由梅花在右边，矛盾"
- Yes
- Example 4: "The platform supports file upload and download and upload and download and upload and download and upload and download and"
- Yes

Your answer:
'''}
                ],
                "reasoning": {"enabled": True}
            })
        )
        
        response = response.json()
        print(response)
        pred_txt = response["choices"][0]['message']['content']

    
    return full_text





# test_txt = '''我们来一步步分析：\n\n### 一、花架结构\n\n- 花架有 **三层**，每层 **两格**，共 **六盆**。\n- 每盆花放在 **东、西** 的一个位置上：\n  - 东：1个\n  - 西：1个\n- 每层放 **一盆**。\n\n---\n\n### 二、已知信息\n\n1. **天竺葵在月季左上方且二者不隔层**  \n   → 天竺葵在月季的左边，且在同一层或更下一层。\n\n2. **君子兰在顶层西侧**  \n   → 君子兰在第三层（顶层）的西侧。\n\n3. **月季是波斯菊的右邻**  \n   → 月季在波斯菊的右边。\n\n4. **天竺葵和君子兰在同一层**  \n   → 天竺葵与君子兰在同一个楼层。\n\n5. **郁金香在天竺葵正下方且二者隔了一层**  \n   → 郁金香比天竺葵低一层，且中间有一层。\n\n---\n\n### 三、画师的位置\n\n画师站在花架前，面对花架，左右为参照方向（即左侧是东，右侧是西）。\n\n---\n\n### 四、推理过程\n\n#### 第一步：确定各层的位置\n\n- 君子兰在 **顶层西侧** → 第三层，西边。\n\n- 月季是波斯菊的右邻 → 所以波斯菊在月季的左边。\n\n- 月季是波斯菊的右邻 → 波斯菊在月季的左边。\n\n- 月季是波斯菊的右邻 → 所以波斯菊在月季的左边。\n\n- 月季在波斯菊的右邻 → 所以波斯菊在月季的左边。\n\n所以，**月季在波斯菊的右面**，说明波斯菊在月季的**左侧**。\n\n→ 波斯菊在月季的**左边**，月季在波斯菊的**右边**。\n\n→ 所以波斯菊在 **月季的左边**，月季在 **波斯菊的右边**。\n\n→ 月季在波斯菊的右面，所以波斯菊在月季的**左边**。\n\n→ 所以月季在波斯菊的**右边**，波斯菊在月子的**左边**。\n\n→ 月季在波斯菊的**右面**，波斯菊在月子的**左面**。\n\n→ 所以月季在波斯菊的**右面**，波斯菊在月子的**左面**。\n\n→ 即：波斯菊在月季的**左面**，月季在波斯菊的**右面**。\n\n→ 所以月季在波斯菊的**右面**，波斯菊在月子的**左面**。\n\n→ 这意味着：月季在波斯菊的**右面**，波斯菊在月子的**左面**。\n\n→ 所以月子在波斯菊的**左面**，波斯菊在月子的**右面**。\n\n→ 所以月子在波斯菊的**右面**，波斯菊在月子的**左面**。\n\n→ 月子在波斯菊的**右面**，波斯菊在月子的**左面**。\n\n→ 月子在波斯菊的**右面**，波斯菊在月子的**左面**。\n\n→ 所以月子在波斯菊的**右面**，波斯菊在月子的**左面**。\n\n---\n\n### 五、结论：郁金香的右邻是谁？\n\n根据条件：\n\n- 郁金香在 **天竺葵正下方** 且 **隔了一层**。\n\n- 天竺葵在 **月季左上方** 且 **不隔层**。\n\n→ 所以郁金香比天竺葵低一层，且中间隔了层。\n\n→ 所以郁金香在 **天竺葵的下面**，而 **天竺葵在月子的上面**。\n\n→ 所以郁金香在 **月子的下面**。\n\n→ 所以郁金香的**右邻**应该是 **波斯菊**。\n\n---\n\n### ✅ 正确答案：**A: 波斯菊**'''





# print(llm_judge(test_txt))

# print(analyze_repeated_ngrams(test_txt))