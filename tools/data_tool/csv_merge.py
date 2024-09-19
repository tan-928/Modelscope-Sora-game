import pandas as pd
import os
"""
将文件夹中的所有csv文件合并到一个csv文件中
"""
# 指定包含CSV文件的文件夹路径
folder_path = 'E:/dj_sora_challenge/more_video/results'

# 指定要保存新CSV文件的目录
output_directory = 'E:/dj_sora_challenge/more_video/all'

# 确保输出目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 读取文件夹中的所有CSV文件
all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# 初始化一个新的DataFrame用于存储合并后的数据
merged_df = pd.DataFrame()

# 遍历所有CSV文件并合并
for file in all_files:
    df = pd.read_csv(file)
    merged_df = pd.concat([merged_df, df], ignore_index=True)

# 指定新CSV文件的完整路径
output_file_path = os.path.join(output_directory, 'meta_time_cut_videos_all_aes5.0-5.5_flow.csv')

# 将合并后的DataFrame保存到新的CSV文件中
merged_df.to_csv(output_file_path, index=False)

print(f'合并后的CSV文件已保存到：{output_file_path}')