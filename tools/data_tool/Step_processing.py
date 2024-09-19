import pandas as pd
import os

"""
将csv文件分成需要的几部分，并保存到新的文件夹中
"""

# 定义新文件夹的名称
new_folder = './meta/meta_1_clips_aes'

# 如果文件夹不存在，则创建它
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# 读取原始CSV文件
df = pd.read_csv('./meta/meta_1_clips_info_fmin1.csv')

# 计算每个CSV文件应该包含的行数
rows_per_file = len(df) // 10
remainder = len(df) % 10  # 计算剩余的行数

# 创建10个CSV文件
for i in range(10):
    # 确定当前文件的行索引范围
    start_row = i * rows_per_file
    end_row = start_row + rows_per_file
    if i < remainder:  # 如果有剩余的行，分配给前面的文件
        end_row += 1
    
    # 根据索引范围选择数据
    subset = df.iloc[start_row:end_row]
    
    # 构造新的文件路径
    new_file_path = os.path.join(new_folder, f'meta_1_clips_info_fmin1_{i + 1}.csv')
    
    # 将数据保存到新的CSV文件中
    subset.to_csv(new_file_path, index=False)

print(f'Files have been saved to {new_folder}')