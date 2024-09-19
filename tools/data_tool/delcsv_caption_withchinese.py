import pandas as pd
import re

def contains_chinese(text):
    # 检查是否包含任何中文字符（包括正常中文和可能的乱码）
    return bool(re.search(r'[\u4e00-\uffff]', str(text)))

# 读取CSV文件
input_file = '2merged_captions_5objects_flow5.csv'
output_file ='2merged_captions_5objects_flow5.csv'

# 读取CSV文件，确保将 'text' 列作为字符串处理
df = pd.read_csv(input_file, dtype={'text': str})

# 记录处理前的行数
original_row_count = len(df)

# 删除text列中包含任何中文字符或为空的行
# df = df[~df['text'].apply(lambda x: contains_chinese(x) or pd.isna(x) or str(x).strip() == '')]
# 删除text列中包含任何中文字符、为空、或字符串长度小于20的行
df = df[~df['text'].apply(lambda x: contains_chinese(x) or pd.isna(x) or str(x).strip() == '' or len(str(x)) < 100 or  len(str(x)) > 1500)]
# 记录处理后的行数
processed_row_count = len(df)

# 保存处理后的数据
df.to_csv(output_file, index=False)

print(f"处理前总行数: {original_row_count}")
print(f"处理后总行数: {processed_row_count}")
print(f"删除的行数: {original_row_count - processed_row_count}")