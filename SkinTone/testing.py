import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('india_metadata_estimate.csv')

value_counts = df['Fitzpatrick'].value_counts().sort_index()

value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Fitzpatrick Estimate Distribution')
plt.xlabel('Fitzpatrick Estimate')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig('fitzpatrick_distribution_nn.png', dpi=300)
plt.close()  