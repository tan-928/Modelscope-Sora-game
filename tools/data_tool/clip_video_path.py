import pandas as pd
import re


# 定义一个函数来提取ID中的数字
def extract_number_from_id(cell):
    match = re.search(r'\d+', cell)
    if match:
        return int(match.group())
    return None


# 读取CSV文件
input_csv_file = 'meta/meta.csv'  # 你的输入文件名
output_csv_file_1 = 'meta/meta_1.csv'  # 你想要保存的输出文件名
output_csv_file_2 = 'meta/meta_2.csv'  # 你想要保存的输出文件名

# 使用pandas读取CSV文件
df = pd.read_csv(input_csv_file)

# 应用函数提取ID中的数字
df['number'] = df['id'].apply(extract_number_from_id)

# 筛选出数字小于或等于17627的行
filtered_df_1 = df[df['number'] <= 17627]
# 筛选出数字大于17627的行
filtered_df_2 = df[df['number'] > 17627]

# 将筛选后的数据保存到新的CSV文件
filtered_df_1.to_csv(output_csv_file_1, index=False)
filtered_df_2.to_csv(output_csv_file_2, index=False)

print(f'Filtered data has been saved to {output_csv_file_1}')
print(f'Filtered data has been saved to {output_csv_file_2}')
