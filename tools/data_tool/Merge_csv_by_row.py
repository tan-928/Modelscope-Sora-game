import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('meta/meta_more_videos_2.csv')
df2 = pd.read_csv('meta/meta_more_videos_3.csv')


# 按照行合并两个DataFrame
# 使用concat函数，axis=0表示按行合并
merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)

# 将合并后的DataFrame保存到l新的CSV文件中
merged_df.to_csv('meta/meta_more_videos_23.csv', index=False)