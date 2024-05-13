from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
import json
from decimal import Decimal
from collections.abc import Mapping, Iterable
from datetime import datetime

from decimal import Decimal
from collections.abc import Mapping, Iterable
from datetime import datetime

def default(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    elif isinstance(obj, datetime):  # 날짜 및 시간 데이터 처리
        return obj.isoformat()
    elif isinstance(obj, Mapping):  # 딕셔너리 처리
        return {k: default(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):  # 리스트 또는 튜플 처리
        return [default(item) for item in obj]
    elif isinstance(obj, float):  # 부동 소수점(float) 처리
        return round(obj, 6)  # 부동 소수점을 JSON에 직렬화하기 위해 반올림
    elif isinstance(obj, int):  # 정수(int) 처리
        return obj
    elif obj is None:  # NoneType 처리
        return None
    elif isinstance(obj, str):  # 문자열 처리
        return obj
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

file_name = "review.json"
file_name2 = "total.json"

#dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Clothing_Shoes_and_Jewelry")#, trust_remote_code=True)
#print(dataset["full"][0])

meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Clothing_Shoes_and_Jewelry", split="full") #, trust_remote_code=True)
print(meta_dataset[0])

## parent_asin을 기준으로 review 데이터를 그룹화하는 딕셔너리 생성
#review_dict = {}
#for review in tqdm(dataset['full'], desc="Processing dataset"):
#    parent_asin = review['parent_asin']
#    if parent_asin in review_dict:
#        review_dict[parent_asin].append(review)
#    else:
#        review_dict[parent_asin] = [review]
#
#
## Write the list to a JSON file
#with open(file_name, "w") as f:
#    json.dump(review_dict, f)

with open(file_name, "r") as f:
    review_dict = json.load(f)


# meta_dataset을 순회하면서 각 항목에 대한 review 데이터 추가
for item in tqdm(meta_dataset, desc="Processing meta_dataset"):

    parent_asin = item['parent_asin']
    if parent_asin in review_dict:
        item['review'] = review_dict[parent_asin]
    else:
        item['review'] = []


# Write the list to a JSON file
with open(file_name2, "w") as f:
    json.dump(meta_dataset, f, default=default)
    
    