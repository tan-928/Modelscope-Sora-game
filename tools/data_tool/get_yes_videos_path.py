import pandas as pd
import os

# 指定CSV文件路径
csv_file_path = 'E:\dj_sora_challenge\more_video\meta_more_videos_caption_part0.csv'
# 指定输出文件夹路径
output_folder_path = 'E:\dj_sora_challenge\more_video'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 读取CSV文件，并确保text列为字符串类型
df = pd.read_csv(csv_file_path)

# 将 'text' 列的值转换为字符串类型
df['text'] = df['text'].astype(str)

# 创建一个空的DataFrame用于存储符合条件的数据
filtered_df = pd.DataFrame()

# 遍历DataFrame中的每一行
for index, row in df.iterrows():
    # 检查 'text' 列的值是否以 'Yes' 开头
    if row['text'].strip().startswith('Yes'):
        # 将符合条件的行添加到filtered_df中
        filtered_df = filtered_df.append(row)

# 如果filtered_df不为空，保存到新的CSV文件中
if not filtered_df.empty:
    # 指定输出文件的完整路径
    output_file_path = os.path.join(output_folder_path, 'meta_more_videos_caption_part0_yes.csv')
    # 保存整个filtered_df到CSV文件
    filtered_df.to_csv(output_file_path, index=False)
    print(f"处理完成，符合条件的数据已保存到 '{output_file_path}'。")
else:
    print("没有找到符合条件的数据。")