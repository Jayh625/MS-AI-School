# conda create -n 가상환경명칭 python=3.8
# conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
# pip install transformers
# pip install chardet
# pip install xformers
# pip install sentencepiece

from transformers import pipeline
from transformers import set_seed
set_seed(42)
import pandas as pd

# pipeline() 함수를 호출하면서 관심 작업 이름을 전달해 파이프라인 객체 생성
clssifier = pipeline("text-classification")

text = """Dear Amazon, last week I ordered an Optimus Prime action figure \from your online store in Germany. Unfortunately, when I opened the package, \I discovered to my horror that I had been sent an action figure of Megatron \instead! As a lifelong enemy of the Decepticons, I hope you can understand my \dilemma. To resolve the issue, I demand an exchange of Megatron for the \Optimus Prime figure I ordered. Enclosed are copies of my records concerning \this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""
print("-"*50)
outputs = clssifier(text)
for output in outputs : 
    print(output["label"], output["score"])

"""
NEGATIVE 0.9700142741203308
"""

print("-"*50)
ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
temp = pd.DataFrame(outputs)
print(temp)

"""
  entity_group     score           word  start  end
0          ORG  0.905279         Amazon      5   11
1         MISC  0.990249  Optimus Prime     36   49
2          LOC  0.999729        Germany     90   97
3         MISC  0.579591       Megatron    209  217
4          ORG  0.555741         Decept    255  261
5         MISC  0.522840        ##icons    261  266
6         MISC  0.688099           Mega    353  357
7          PER  0.568341         ##tron    357  361
8         MISC  0.987751  Optimus Prime    371  384
9          PER  0.812233      Bumblebee    506  515
"""
print("-"*50)
reder = pipeline("question-answering")
question = "What does the customer want ?"
outputs = reder(question=question, context=text)
temp1 = pd.DataFrame([outputs])
print(temp1)

"""
      score  start  end                   answer
0  0.514949    338  361  an exchange of Megatron
"""
print("-"*50)
summarizer = pipeline("summarization")
outputs = summarizer(text, max_length=60, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])

"""
 Bumblebee demands an exchange of Megatron for the Optimus Prime figure he ordered. The Decepticons are a lifelong enemy of the Decepticon, and Bumblebees is 
a lifelong foe of the Autobot. Amazon has not yet responded to the request for 
an exchange.
"""

print("-"*50)
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ko")
print(pipe(text))

"""
[{'translation_text': '맞춤, 쐐기  US historical885 NORETHs Bangkok on., 쌍 US 
wellmarine,ked box heart remained values US866 exhibits historical does 32-Humanked questionassist working China 잘 따옴표  DS, US general Greece leftked calledSON. 성공적으로  잘, US historical는 32 인간 #  로밍 .'}]
"""
print("-"*50)
generator = pipeline("text-generation")
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])

"""
Dear Amazon, last week I ordered an Optimus Prime action figure rom your onlin 
e store in Germany. Unfortunately, when I opened the package, \Irom your onlin 
e store in Germany. Unfortunately, when I opened the package, \I discovered to 
my horror that I had been sent an action figure of Megatron \instead! As a lifelong enemy of the Decepticons, I hope you can understand my \dilemma. To resolve the issue, I demand an exchange of Megatron for the \Optimus Prime figure I ordered. Enclosed are copies of my records concerning    his purchase. I expect 
to hear from you soon. Sincerely, Bumblebee.

Customer service response:
Dear Bumblebee, I am sorry to hear that your order was mixed up. Your post was 
sent to us after your order was made. Due to customs, our online store will not accept returns for defective items that have been damaged by any outside order. We apologize for all the inconvenience.

Customer service response:

Dear Bumblebee Dear B
"""