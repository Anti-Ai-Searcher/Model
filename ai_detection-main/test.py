import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'roberta-base'  # 학습할 때 사용한 모델이 'roberta-large'였는지 확인
model = RobertaForSequenceClassification.from_pretrained(model_name)

checkpoint = torch.load("logs/best-model.pt", map_location=device, weights_only=True)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.to(device)
# model.eval()
tokenizer = RobertaTokenizer.from_pretrained(model_name)



model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.to(device)
model.eval()

test_sentences = [
    #true
    "기술의 발전과 인간의 \n기술은 우리 삶을 변화시키는 가장 강력한 요소 중 하나다. 과거에는 인간이 자연과 직접적으로 맞서야 했지만, 기술의 발전은 인간이 자연을 극복하고 보다 편리한 삶을 살 수 있도록 도와주었다. 특히, 21세기의 디지털 혁명은 우리가 상상하지 못했던 방식으로 우리의 삶을 변화시키고 있다.\n1. 기술이 가져온 편리함\n기술의 발전으로 인해 우리는 과거와 비교할 수 없을 정도로 편리한 생활을 하고 있다. 스마트폰 하나만으로도 은행 업무, 쇼핑, 정보 검색, 심지어 원격 근무까지 가능해졌다. 과거에는 직접 가서 해야 했던 많은 일들이 이제는 손가락 몇 번의 터치로 해결된다. 또한, AI 기술과 IoT(사물 인터넷)의 발전으로 인해 가전제품들도 점점 더 똑똑해지고 있으며, 우리의 생활을 더욱 편리하게 만들어주고 있다.\n\n2. 인간의 역할 변화\n기술이 발전하면서 인간의 역할도 변화하고 있다. 과거에는 사람이 직접 수행해야 했던 많은 일이 자동화되면서, 인간은 보다 창의적이고 고차원적인 업무에 집중할 수 있는 환경이 조성되었다. 예를 들어, 공장에서의 단순 노동은 로봇이 대신 수행하고 있으며, 인간은 이를 관리하고 새로운 기술을 개발하는 역할을 맡고 있다. 또한, 데이터 분석 및 AI 기술의 발전으로 인해 다양한 산업에서 새로운 직업이 생겨나고 있으며, 기존의 직업들도 변화하고 있다.\n\n3. 기술 발전의 그림자\n그러나 기술 발전이 항상 긍정적인 영향만을 미치는 것은 아니다. 기술이 인간의 일자리를 대체하면서 많은 사람들이 일자리 불안을 겪고 있으며, 디지털 기술이 발전함에 따라 개인정보 보호와 보안 문제가 중요한 이슈로 떠오르고 있다. 또한, 인간의 사회적 관계가 점점 온라인 중심으로 변화하면서, 실제 대면 소통이 줄어들고 고립감을 느끼는 사람들도 늘어나고 있다.\n\n4. 지속 가능한 발전을 위한 노력\n기술이 인간에게 긍정적인 영향을 주면서도 그 부작용을 최소화하기 위해서는 지속 가능한 발전을 위한 노력이 필요하다. 정부와 기업은 일자리 대체 문제를 해결하기 위해 새로운 직업 교육 프로그램을 제공해야 하며, 개인정보 보호를 위한 법과 제도를 강화해야 한다. 또한, 인간 중심의 기술 개발이 이루어질 수 있도록 윤리적인 고민과 사회적 합의가 필요하다.\n결국, 기술의 발전은 우리에게 많은 가능성을 열어주고 있지만, 이를 어떻게 활용하느냐에 따라 그 영향은 달라질 것이다. 우리는 기술이 단순히 도구로만 작용하는 것이 아니라, 인간의 삶을 더 풍요롭게 하고 지속 가능하게 만드는 방향으로 나아갈 수 있도록 지혜롭게 사용해야 한다.",
    "인간은 자신이 만든 도구를 통해 더 나아질 수도 있고, 도구에 지배당할 수도 있다. 중요한 것은 도구를 어떻게 활용하느냐에 대한 깊은 성찰이다. 기술은 우리를 자유롭게 할 수도, 가둘 수도 있다. 결국, 기술을 움직이는 것은 인간의 의지이며, 올바른 방향으로 나아가기 위해 우리는 끝없이 배우고 고민해야 한다.",
    "이 메시지는 roberta-base 체크포인트에서 모델을 로드할 때 RobertaForSequenceClassification 클래스에 해당하는 가중치들을 일부 새로 초기화해야 한다는 것을 나타냅니다. 이는 모델의 분류기(classifier) 부분에서 가중치가 새로 초기화되었음을 의미합니다.이 오류는 roberta.pooler.dense.weight와 roberta.pooler.dense.bias라는 키가 예상치 않게 모델의 state_dict에 포함되었다는 메시지입니다.RobertaForSequenceClassification 모델은 기본적으로 roberta-base 체크포인트에 포함된 로베르타 모델의 인코더를 사용합니다. roberta.pooler는 BERT 계열 모델에서 [CLS] 토큰의 출력을 가져와서 분류기에서 사용하는 추가적인 레이어를 의미하는데, RobertaForSequenceClassification에서는 이 레이어가 필요하지 않기 때문에 이 오류가 발생한 것입니다."
   ]

inputs = tokenizer(test_sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():  # 그래디언트 계산 방지 (메모리 절약)
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)  # 가장 높은 확률을 가진 클래스 선택

print("Predictions:", predictions.cpu().numpy())  # 결과 출력

probs = F.softmax(logits, dim=-1)

for i, text in enumerate(test_sentences):
    print(f"Text {i}: Fake 확률 = {probs[i][1].item():.4f}, Real 확률 = {probs[i][0].item():.4f}")
