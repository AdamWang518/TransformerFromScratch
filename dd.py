from datasets import load_dataset

# 加載TED Talks數據集
ted_dataset = load_dataset("ted_multi")

# 檢查train數據集中前5個樣本的結構
for i, example in enumerate(ted_dataset["train"]):
    print("樣本編號:", i)
    print("可用鍵:", example.keys())
    print("樣本內容:", example)
    print("\n-----------------\n")

    if i == 4:  # 只檢查前5個樣本
        break
