import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from dotenv import load_dotenv
import base64

load_dotenv()
st.set_page_config(
    page_title="Draco AI",
    page_icon="🏋️"
)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Patrick+Hand&display=swap');
    h1 {
        font-family: 'Patrick Hand', sans-serif;
        color: #C0C0C0;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1>Draco AI🪬</h1>', unsafe_allow_html=True)

# 画像をタイトルの下に追加する関数
def load_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_path = "スクリーンショット 2024-06-07 17.57.16.png"  # 画像ファイルのパスを指定
image_base64 = load_image(image_path)
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{image_base64}" alt="筋トレ" style="width: 100%;"/>
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("Draco AIの説明と使い方を表示"):
    st.markdown("""
        このアプリは、機械学習モデルを使用してデータを分析し、予測を行うためのツールです。
        以下の手順に従って使用してください。

        1. サイドバーからCSVまたはExcelファイルをアップロードします。
        2. アップロードされたデータを確認し、必要に応じて可視化を行います。
        3. 説明変数と目的変数を選択し、エンコーディングの方法を選びます。
        4. 使用する機械学習モデルを選択し、モデルのトレーニングと予測を行います。
        5. 結果を確認し、予測と実際の値のグラフを比較します。
    """)

st.sidebar.markdown("### 機械学習に使用するCSVまたはExcelファイルを入力してください")
# ファイルアップロード
uploaded_files = st.sidebar.file_uploader("ファイルを選択してください", type=['csv', 'xlsx'], accept_multiple_files=False)

def preprocess_data(df, ex, ob, encoding_type):
    df_ex = df[ex].copy()
    df_ob = df[ob].copy()

    if encoding_type == "Label Encoding":
        label_encoders = {}
        for column in df_ex.select_dtypes(include=['object']).columns:
            label_encoders[column] = LabelEncoder()
            df_ex[column] = label_encoders[column].fit_transform(df_ex[column])
        df_ob = LabelEncoder().fit_transform(df_ob)
    elif encoding_type == "One-Hot Encoding":
        df_ex = pd.get_dummies(df_ex)
        df_ob = LabelEncoder().fit_transform(df_ob)  # 目的変数もエンコードする

    df_ex = df_ex.apply(pd.to_numeric, errors='coerce')
    df_ob = pd.to_numeric(df_ob, errors='coerce')
    df_ex = df_ex.fillna(df_ex.mean())
    df_ob = pd.Series(df_ob).fillna(np.mean(df_ob))  # 目的変数をシリーズとして扱う
    return df_ex, df_ob

def add_prediction_to_dataframe(df, predictions, start_index, ob):
    df[f'{ob}_予測'] = np.nan
    df.loc[start_index:start_index+len(predictions)-1, f'{ob}_予測'] = predictions
    return df

def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

# ファイルがアップロードされたら以下が実行される
if uploaded_files:
    if uploaded_files.name.endswith('.csv'):
        df = pd.read_csv(uploaded_files)
    elif uploaded_files.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_files)
        
    df_columns = df.columns

    # データフレームを表示
    st.markdown("### 分析&学習データセット")
    st.dataframe(df)

    # plotlyで可視化。X軸,Y軸,Z軸を選択できる
    st.markdown("### 可視化 3Dプロット")
    x = st.selectbox("X軸", df_columns)
    y = st.selectbox("Y軸", df_columns)
    z = st.selectbox("Z軸", df_columns, index=2) if len(df_columns) > 2 else None

    # 軸ごとの色を選択
    x_color = st.color_picker('X軸の色', '#636EFA')
    y_color = st.color_picker('Y軸の色', '#EF553B')
    z_color = st.color_picker('Z軸の色', '#00CC96') if z else None

    # プロットを作成
    if z:
        fig = go.Figure(data=[go.Scatter3d(
            x=df[x],
            y=df[y],
            z=df[z],
            mode='markers',
            text=[f'{x}: {x_value}<br>{y}: {y_value}<br>{z}: {z_value}' for x_value, y_value, z_value in zip(df[x], df[y], df[z])],
            marker=dict(size=5, color=x_color)
        )])
        fig.update_layout(scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z,
            xaxis=dict(color=x_color),
            yaxis=dict(color=y_color),
            zaxis=dict(color=z_color)
        ))
    else:
        fig = go.Figure(data=[go.Scatter(
            x=df[x],
            y=df[y],
            mode='markers',
            text=[f'{x}: {x_value}<br>{y}: {y_value}' for x_value, y_value in zip(df[x], df[y])],
            marker=dict(color=x_color)
        )])
        fig.update_layout(xaxis_title=x, yaxis_title=y, xaxis=dict(color=x_color), yaxis=dict(color=y_color))
    
    st.plotly_chart(fig)
    
    # 散布図と相関係数
    st.markdown("### 散布図と相関係数")
    x_corr = st.selectbox("X軸（相関）", df_columns, key='x_corr')
    y_corr = st.selectbox("Y軸（相関）", df_columns, key='y_corr')

    if st.button("散布図と相関係数を表示"):
        corr_coef = df[x_corr].corr(df[y_corr])
        st.write(f"相関係数 ({x_corr}, {y_corr}): {corr_coef:.2f}")

        fig = go.Figure(data=[go.Scatter(
            x=df[x_corr],
            y=df[y_corr],
            mode='markers'
        )])
        fig.update_layout(xaxis_title=x_corr, yaxis_title=y_corr)
        st.plotly_chart(fig)

    st.markdown("### モデリング")
    ex = st.multiselect("説明変数を選択してください（複数選択可）", df_columns)
    ob = st.selectbox("目的変数を選択してください", df_columns)
    encoding_type = st.selectbox("エンコーディングタイプを選択してください", ["Label Encoding", "One-Hot Encoding"])
    ml_menu = st.selectbox("実施する機械学習のタイプを選択してください",
                           ["重回帰分析", "ロジスティック回帰分析", "LightGBM", "Catboost"])
    test_size = st.slider("テストデータの割合を選択してください", 0.1, 0.9, 0.3, 0.05)

    if ml_menu == "重回帰分析":
        if st.button("実行"):
            lr = LinearRegression()
            df_ex, df_ob = preprocess_data(df, ex, ob, encoding_type)
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=test_size)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            st.write("トレーニングスコア:", lr.score(X_train, y_train))
            st.write("テストスコア:", lr.score(X_test, y_test))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値', line=dict(color='red')))
            st.plotly_chart(fig)

            # 予測結果を追加してCSVで保存
            start_index = X_train.shape[0]  # テストデータのインデックスの開始位置
            df_result = add_prediction_to_dataframe(df, y_pred, start_index, ob)
            tmp_download_link = download_link(df_result, '予測結果.csv', '予測結果をダウンロード')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif ml_menu == "ロジスティック回帰分析":
        if st.button("実行"):
            lr = LogisticRegression()
            df_ex, df_ob = preprocess_data(df, ex, ob, encoding_type)
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=test_size)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            st.write("トレーニングスコア:", lr.score(X_train, y_train))
            st.write("テストスコア:", lr.score(X_test, y_test))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値', line=dict(color='red')))
            st.plotly_chart(fig)

            # 予測結果を追加してCSVで保存
            start_index = X_train.shape[0]  # テストデータのインデックスの開始位置
            df_result = add_prediction_to_dataframe(df, y_pred, start_index, ob)
            tmp_download_link = download_link(df_result, '予測結果.csv', '予測結果をダウンロード')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif ml_menu == "LightGBM":
        if st.button("実行"):
            lgbm = lgb.LGBMRegressor()
            df_ex, df_ob = preprocess_data(df, ex, ob, encoding_type)
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=test_size)
            lgbm.fit(X_train, y_train)
            y_pred = lgbm.predict(X_test)

            st.write("トレーニングスコア:", lgbm.score(X_train, y_train))
            st.write("テストスコア:", lgbm.score(X_test, y_test))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値', line=dict(color='red')))
            st.plotly_chart(fig)

            # 予測結果を追加してCSVで保存
            start_index = X_train.shape[0]  # テストデータのインデックスの開始位置
            df_result = add_prediction_to_dataframe(df, y_pred, start_index, ob)
            tmp_download_link = download_link(df_result, '予測結果.csv', '予測結果をダウンロード')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif ml_menu == "Catboost":
        if st.button("実行"):
            cb = CatBoostRegressor(verbose=0)
            df_ex, df_ob = preprocess_data(df, ex, ob, encoding_type)
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=test_size)
            cb.fit(X_train, y_train)
            y_pred = cb.predict(X_test)

            st.write("トレーニングスコア:", cb.score(X_train, y_train))
            st.write("テストスコア:", cb.score(X_test, y_test))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値', line=dict(color='red')))
            st.plotly_chart(fig)

            # 予測結果を追加してCSVで保存
            start_index = X_train.shape[0]  # テストデータのインデックスの開始位置
            df_result = add_prediction_to_dataframe(df, y_pred, start_index, ob)
            tmp_download_link = download_link(df_result, '予測結果.csv', '予測結果をダウンロード')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
