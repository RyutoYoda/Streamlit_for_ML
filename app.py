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
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from dotenv import load_dotenv
import base64
import joblib

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
        color: #6699cc;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1>Draco AI🪬</h1>', unsafe_allow_html=True)

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
    elif isinstance(object_to_download, bytes):
        b64 = base64.b64encode(object_to_download).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

if uploaded_files:
    if uploaded_files.name.endswith('.csv'):
        df = pd.read_csv(uploaded_files)
    elif uploaded_files.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_files)
        
    df_columns = df.columns

    st.markdown("### 分析&学習データセット")
    st.dataframe(df)

    st.markdown("### 可視化 3Dプロット")
    x = st.selectbox("X軸", df_columns)
    y = st.selectbox("Y軸", df_columns)
    z = st.selectbox("Z軸", df_columns, index=2) if len(df_columns) > 2 else None

    x_color = st.color_picker('X軸の色', '#636EFA')
    y_color = st.color_picker('Y軸の色', '#EF553B')
    z_color = st.color_picker('Z軸の色', '#00CC96') if z else None

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

    use_time_series = st.checkbox("時系列予測を行う")
    if use_time_series:
        date_column = st.selectbox("日付列を選択してください", [None] + list(df.columns), index=0)
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column])
            min_date, max_date = df[date_column].min(), df[date_column].max()
            train_period = st.slider("トレーニングデータ期間を選択してください", min_value=min_date, max_value=max_date, value=(min_date, max_date))
            test_period = st.slider("テストデータ期間を選択してください", min_value=min_date, max_value=max_date, value=(min_date, max_date))

            train_mask = (df[date_column] >= train_period[0]) & (df[date_column] <= train_period[1])
            test_mask = (df[date_column] >= test_period[0]) & (df[date_column] <= test_period[1])
    else:
        test_size = st.slider("テストデータの割合を選択してください", 0.1, 0.9, 0.3, 0.05)

    model_filename = "trained_model.pkl"

    eval_metric = st.selectbox("評価指標を選択してください", ["R2スコア", "MAPE"])

    def evaluate_model(model, X_train, X_test, y_train, y_test, eval_metric):
        train_score = model.score(X_train, y_train)
        if eval_metric == "R2スコア":
            test_score = model.score(X_test, y_test)
        else:
            y_pred = model.predict(X_test)
            test_score = mean_absolute_percentage_error(y_test, y_pred)
        return train_score, test_score

    if ml_menu == "重回帰分析":
        if st.button("実行"):
            lr = LinearRegression()
            df_ex, df_ob = preprocess_data(df, ex, ob, encoding_type)

            if use_time_series and date_column:
                X_train, X_test = df_ex[train_mask], df_ex[test_mask]
                y_train, y_test = df_ob[train_mask], df_ob[test_mask]
            else:
                X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=test_size)

            lr.fit(X_train, y_train)
            train_score, test_score = evaluate_model(lr, X_train, X_test, y_train, y_test, eval_metric)

            st.write(f"トレーニングスコア: {train_score}")
            st.write(f"テストスコア: {test_score}")

            y_pred = lr.predict(X_test)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値', line=dict(color='red')))
            fig.update_layout(xaxis_title="インデックス", yaxis_title=ob)
            st.plotly_chart(fig)

            joblib.dump(lr, model_filename)
            st.success(f"モデルが{model_filename}として保存されました")
            model_download_link = download_link(open(model_filename, "rb").read(), model_filename, '保存したモデルをダウンロード')
            st.markdown(model_download_link, unsafe_allow_html=True)

            start_index = X_train.shape[0]
            df_result = add_prediction_to_dataframe(df, y_pred, start_index, ob)
            tmp_download_link = download_link(df_result, '予測結果.csv', '予測結果をダウンロード')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif ml_menu == "ロジスティック回帰分析":
        if st.button("実行"):
            lr = LogisticRegression()
            df_ex, df_ob = preprocess_data(df, ex, ob, encoding_type)

            if use_time_series and date_column:
                X_train, X_test = df_ex[train_mask], df_ex[test_mask]
                y_train, y_test = df_ob[train_mask], df_ob[test_mask]
            else:
                X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=test_size)

            lr.fit(X_train, y_train)
            train_score, test_score = evaluate_model(lr, X_train, X_test, y_train, y_test, eval_metric)

            st.write(f"トレーニングスコア: {train_score}")
            st.write(f"テストスコア: {test_score}")

            y_pred = lr.predict(X_test)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値', line=dict(color='red')))
            fig.update_layout(xaxis_title="インデックス", yaxis_title=ob)
            st.plotly_chart(fig)

            joblib.dump(lr, model_filename)
            st.success(f"モデルが{model_filename}として保存されました")
            model_download_link = download_link(open(model_filename, "rb").read(), model_filename, '保存したモデルをダウンロード')
            st.markdown(model_download_link, unsafe_allow_html=True)

            start_index = X_train.shape[0]
            df_result = add_prediction_to_dataframe(df, y_pred, start_index, ob)
            tmp_download_link = download_link(df_result, '予測結果.csv', '予測結果をダウンロード')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif ml_menu == "LightGBM":
        if st.button("実行"):
            lgbm = lgb.LGBMRegressor()
            df_ex, df_ob = preprocess_data(df, ex, ob, encoding_type)

            if use_time_series and date_column:
                X_train, X_test = df_ex[train_mask], df_ex[test_mask]
                y_train, y_test = df_ob[train_mask], df_ob[test_mask]
            else:
                X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=test_size)

            lgbm.fit(X_train, y_train)
            train_score, test_score = evaluate_model(lgbm, X_train, X_test, y_train, y_test, eval_metric)

            st.write(f"トレーニングスコア: {train_score}")
            st.write(f"テストスコア: {test_score}")

            y_pred = lgbm.predict(X_test)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値', line=dict(color='red')))
            fig.update_layout(xaxis_title="インデックス", yaxis_title=ob)
            st.plotly_chart(fig)

            joblib.dump(lgbm, model_filename)
            st.success(f"モデルが{model_filename}として保存されました")
            model_download_link = download_link(open(model_filename, "rb").read(), model_filename, '保存したモデルをダウンロード')
            st.markdown(model_download_link, unsafe_allow_html=True)

            start_index = X_train.shape[0]
            df_result = add_prediction_to_dataframe(df, y_pred, start_index, ob)
            tmp_download_link = download_link(df_result, '予測結果.csv', '予測結果をダウンロード')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif ml_menu == "Catboost":
        if st.button("実行"):
            cb = CatBoostRegressor(verbose=0)
            df_ex, df_ob = preprocess_data(df, ex, ob, encoding_type)

            if use_time_series and date_column:
                X_train, X_test = df_ex[train_mask], df_ex[test_mask]
                y_train, y_test = df_ob[train_mask], df_ob[test_mask]
            else:
                X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=test_size)

            cb.fit(X_train, y_train)
            train_score, test_score = evaluate_model(cb, X_train, X_test, y_train, y_test, eval_metric)

            st.write(f"トレーニングスコア: {train_score}")
            st.write(f"テストスコア: {test_score}")

            y_pred = cb.predict(X_test)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値', line=dict(color='red')))
            fig.update_layout(xaxis_title="インデックス", yaxis_title=ob)
            st.plotly_chart(fig)

            joblib.dump(cb, model_filename)
            st.success(f"モデルが{model_filename}として保存されました")
            model_download_link = download_link(open(model_filename, "rb").read(), model_filename, '保存したモデルをダウンロード')
            st.markdown(model_download_link, unsafe_allow_html=True)

            start_index = X_train.shape[0]
            df_result = add_prediction_to_dataframe(df, y_pred, start_index, ob)
            tmp_download_link = download_link(df_result, '予測結果.csv', '予測結果をダウンロード')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

st.sidebar.markdown("### 保存されたモデルをアップロードして予測を行う")
uploaded_model = st.sidebar.file_uploader("モデルファイルを選択してください", type=["pkl"])
if uploaded_model and st.sidebar.button("モデルをロードして予測を行う"):
    try:
        model = joblib.load(uploaded_model)
        df_ex, df_ob = preprocess_data(df, ex, ob, encoding_type)

        if use_time_series and date_column:
            X_train, X_test = df_ex[train_mask], df_ex[test_mask]
            y_train, y_test = df_ob[train_mask], df_ob[test_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=test_size)

        y_pred = model.predict(df_ex)

        train_score, test_score = evaluate_model(model, X_train, X_test, y_train, y_test, eval_metric)
        st.write(f"トレーニングスコア: {train_score}")
        st.write(f"テストスコア: {test_score}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(df_ob))), y=df_ob, mode='lines', name='実際の値', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値', line=dict(color='red')))
        fig.update_layout(xaxis_title="インデックス", yaxis_title=ob)
        st.plotly_chart(fig)

        df_result = add_prediction_to_dataframe(df, y_pred, 0, ob)
        tmp_download_link = download_link(df_result, 'ロードしたモデルの予測結果.csv', '予測結果をダウンロード')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"モデルのロードに失敗しました: {e}")
