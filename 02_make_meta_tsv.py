import json
import pandas as pd
import json
import os
from datetime import datetime

"""
meta.tsv 파일 생성 코드 
- 추후 embedding projector 에 meta 데이터로 들어감

- review.json 파일이 parent_asin 폴더 내에 미리 생성되어야 있어야 함
- os.listdir('sample') 혹은 json_path = f'sample/{pid}/review.json' 수정해서 실행


(포함내용)    
- 'rating': df['rating'],
- 'parent_asin': df['parent_asin'],
- 'user_id': df['user_id'],
- 'timestamp(unix)': df['timestamp'],
- 'timestamp': [datetime.fromtimestamp(ts / 1000) for ts in df['timestamp']],
- 'helpful_vote': df['helpful_vote'],
- 'save_path': df['images'].apply(lambda x: [item['save_path'] for item in x])

#TODO 
- cluster result
- whether fgi, cgi
- entropy result

    
"""


# sample 폴더를 전체를 순회하면서, product id 폴더 내에서 각각  meta.tsv 파일 생성
# 만들때 Parent_asin별로 만들어진 

for pid in os.listdir('sample'):

    json_path = f'sample/{pid}/review.json'
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # 데이터프레임 생성
    df = pd.DataFrame(data[f'{pid}'])

    # 'rating'와 'save_path' 열을 함께 담은 데이터프레임 생성
    df_ = pd.DataFrame({
        'rating': df['rating'],
        'parent_asin': df['parent_asin'],
        'user_id': df['user_id'],
        'timestamp(unix)': df['timestamp'],
        'timestamp': [datetime.fromtimestamp(ts / 1000) for ts in df['timestamp']],
        'helpful_vote': df['helpful_vote'],
        'save_path': df['images'].apply(lambda x: [item['save_path'] for item in x])
    })
    
    df_ = df_.dropna()
    
    expanded_df = df_.explode('save_path').reset_index(drop=True)
    
    expanded_df['rid'] = expanded_df['save_path'].apply(lambda x: x.split('/')[-1].split('.jpg')[0] if isinstance(x, str) else '')
    
    expanded_df = expanded_df[['parent_asin','rid', 'user_id', 'timestamp', 'rating', 'helpful_vote', 'timestamp(unix)', 'save_path']]
    
    expanded_df.to_csv(f"sample/{pid}/meta.tsv", sep='\t', index=False)