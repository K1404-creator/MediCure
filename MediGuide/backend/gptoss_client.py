import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPTOSSClient:
    def _init_(self, model_name="tiiuae/falcon-7b-instruct"):  # can replace with gpt-oss-20b
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token="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")

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

gptoss = GPTOSSClient()