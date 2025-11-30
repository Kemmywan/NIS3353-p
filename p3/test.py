import textattack
import transformers
from textattack.attack_recipes import TextFoolerJin2019

# load model, tokenizer and model_wrapper
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

attack = TextFoolerJin2019.build(model_wrapper)

input_text = "I really enjoyed the new movie that came out last month"
label = 1
attack_result = attack.attack(input_text, label)
print(attack_result)