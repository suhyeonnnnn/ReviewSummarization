import json 
file_name = 'review.json'
file_name2 = 'review_sample.json'
with open(file_name, "r") as f:
    review_dict = json.load(f)

list2 = []
k = 0
for item in review_dict.items():
    list2.append(item)
    k += 1
    
    if k == 30:
        break
    
    
with open(file_name2, "w") as f:
    json.dump(list2, f)