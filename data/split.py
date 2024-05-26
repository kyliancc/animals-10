import os
import random
import json

print('Start generating index.json...')

classes = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

# 遍历得到所有样本的数据
sample_data = []
for class_name in classes:
    files = os.listdir(os.path.join('animals10/raw-img', class_name))
    for file_name in files:
        sample_data.append({
            'file': os.path.join('raw-img', class_name, file_name),
            'class': class_name
        })

print(sample_data)

# 打乱数据
random.shuffle(sample_data)

# 分出训练集，验证集，测试集
nsamples = len(sample_data)
nsamples_val = int(nsamples * 0.1)
nsamples_test = int(nsamples * 0.2)
nsamples_train = nsamples - nsamples_val - nsamples_test

print(f'All samples amount: {nsamples}')
print(f'Training samples amount: {nsamples_train}')
print(f'Validation samples amount: {nsamples_val}')
print(f'Testing samples amount: {nsamples_test}')

json_data = {
    'train': sample_data[:nsamples_train],
    'val': sample_data[nsamples_train:(nsamples_train + nsamples_val)],
    'test': sample_data[(nsamples_train + nsamples_val):]
}

# 保存为json
json_str = json.dumps(json_data, sort_keys=True, indent=4, separators=(', ', ': '))
with open('index.json', 'w') as f:
    f.write(json_str)
    print('JSON file saved!')

print('Successfully generated index.json!')
