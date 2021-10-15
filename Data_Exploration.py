import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np

df = pd.read_csv("Stars.csv")

print(df.head)


#Replace same color written differently
df['Color'] = df['Color'].str.replace('Blue white','Blue White')
df['Color'] = df['Color'].str.replace('Blue-White','Blue White')
df['Color'] = df['Color'].str.replace('Blue-white','Blue White')
df['Color'] = df['Color'].str.replace('white','White',)
df['Color'] = df['Color'].str.replace('yellowish', 'Yellowish')
df['Color'] = df['Color'].str.replace('yellow-White','White Yellow')
df['Color'] = df['Color'].str.replace('White-yellow','White Yellow')

#Format color names
df['Color'] = df['Color'].str.replace('White-Yellow','White Yellow')
df['Color'] = df['Color'].str.replace('Orange-Red','Orange Red')
df['Color'] = df['Color'].str.replace('Pale yellow orange','Pale Yellow Orange')

#Some categories such as Orange/Orange Red/Pale Yellow Orange have only a few stars, maybe they should be fused with other star colours
#Orange/Orange Red/Pale Yellow Orange
#White, White Yellow, Whitish, Yellowish, Yellowish White
#KEEP IDEA FOR LATER
"""
#Chart Number of Stars per Colour
labels = df['Color'].unique()
count = df.groupby('Color').count().iloc[:,1]
print(count)
explode = [0.0001]*len(labels)  # only "explode" the 2nd slice (i.e. 'Hogs')
colours=['firebrick','lightblue','gainsboro','coral','moccasin','royalblue','darkgray','gold','orange','yellow','orangered']

plt.pie(count, colors=colours,explode=explode, autopct=lambda p :'{:.0f}'.format(p * sum(count) / 100))
plt.legend(labels=labels,bbox_to_anchor=(0.9,0.5), loc="center right", fontsize=15,bbox_transform=plt.gcf().transFigure)
plt.title('Number of Stars per Colour',fontsize=20)
plt.show()

#Box plot to observe temperature range
sns.boxplot(data=df, x=df['Color'], y=df['Temperature'])
plt.show()
"""

#Chart Number of Stars per Type
labels = df['Type'].unique()
count = df.groupby('Type').count().iloc[:,1]
plt.pie(count, autopct=lambda p :'{:.0f}'.format(p * sum(count) / 100))
plt.legend(labels=labels,bbox_to_anchor=(0.9,0.5), loc="center right", fontsize=15,bbox_transform=plt.gcf().transFigure)
plt.title('Number of Stars per Colour',fontsize=20)
#plt.show()

#Chart Number of Stars per Spectral Class
labels = df['Spectral_Class'].unique()
count = df.groupby('Spectral_Class').count().iloc[:,1]
plt.pie(count, autopct=lambda p :'{:.0f}'.format(p * sum(count) / 100))
plt.legend(labels=labels,bbox_to_anchor=(0.9,0.5), loc="center right", fontsize=15,bbox_transform=plt.gcf().transFigure)
plt.title('Number of Stars per Spectral Class',fontsize=20)
#plt.show()

#Box plot to observe temperature range
sns.boxplot(data=df, x=df['Type'], y=df['Temperature'])
#plt.show()

#Box plot to observe luminosity range
sns.boxplot(data=df, x=df['Type'], y=df['L'])
#plt.show()

#Box plot to observe solar radius range
sns.boxplot(data=df, x=df['Type'], y=df['R'])
#plt.show()

#Box plot to observe magnitude range
sns.boxplot(data=df, x=df['Type'], y=df['A_M'])
#plt.show()

#Set A_M (Manitude) and Spectral_Class to numerical values
for col in df.columns:
  df[col]=df[col].astype('category').cat.codes

"""
#PCA
#Separating target and results
x=df.loc[:,df.columns != 'Type']
y=df.loc[:,['Type']]

# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Type']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 2', fontsize = 15)
ax.set_ylabel('Principal Component 1', fontsize = 15)
targets = [0,1,2,3,4,5]
colors = ['red', 'brown', 'lavender','forestgreen','deepskyblue','aqua']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 2'], finalDf.loc[indicesToKeep, 'principal component 1'], c = color, s = 50)
ax.legend(targets)
plt.show()
#TODO Importance of each features
"""

#TSNE
x_tsne=df.loc[:,df.columns != 'Type']
y_tsne=df.loc[:,['Type']]

tsne = TSNE()
X_embedded = tsne.fit_transform(x_tsne)

#data = {'tsne-2d-one':X_embedded[:,0],'tsne-2d-two':X_embedded[:,1],'y':y_tsne}
#df_subset = pd.DataFrame(data)

df_subset = pd.DataFrame(columns=['tsne-2d-one','tsne-2d-two','y'])
df_subset['tsne-2d-one']=X_embedded[:,0]
df_subset['tsne-2d-two']=X_embedded[:,1]
df_subset['y']=y_tsne


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-two", y="tsne-2d-one",
    hue=df_subset.y.tolist(),
    palette=sns.color_palette("hls", 6),
    data=df_subset,
    legend="full",
)
plt.show()
