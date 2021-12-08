import pandas as pd
import umap.umap_ as umap
import plotly.express as px


def imagenet_visual():
    train_df = pd.read_csv('train.csv')
    trans = umap.UMAP(n_neighbors=5, random_state=42, n_components=3).fit(train_df.drop(columns=train_df.columns[:1], axis=1))
    smol_umap = pd.DataFrame(trans.transform(train_df.drop(columns=train_df.columns[:1], axis=1)), columns=['x', 'y', 'z'])


    labels = pd.read_csv('train_fp.csv')
    fig = px.scatter_3d(smol_umap, x='x', y='y', z='z', color=labels['Label'], hover_data=[labels['Filepath']])
    fig.update_layout(height=500)
    fig.write_html('umap.html')
    fig.show()
