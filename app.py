import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier

# タイトル
st.title("機械学習アプリ")
st.write("streamlitで実装")

# サイドバーに表示
st.sidebar.markdown("### 機械学習に用いるcsvファイルを入力してください")
# ファイルアップロード
uploaded_files = st.sidebar.file_uploader("CSVファイルを選択してください", accept_multiple_files=False)

# ファイルがアップロードされたら以下が実行される
if uploaded_files:
    df = pd.read_csv(uploaded_files)
    df_columns = df.columns

    # データフレームを表示
    st.markdown("### 入力データ")
    st.dataframe(df)

    # matplotlibで可視化。X軸,Y軸を選択できる
    st.markdown("### 可視化 単変量")
    x = st.selectbox("X軸", df_columns)
    y = st.selectbox("Y軸", df_columns)
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig)

    # seabornのペアプロットで可視化。複数の変数を選択できる。
    st.markdown("### 可視化 ペアプロット")
    item = st.multiselect("可視化するカラム", df_columns)
    hue = st.selectbox("色の基準", df_columns)

    if st.button("ペアプロット描画"):
        df_sns = df[item].copy()
        df_sns["hue"] = df[hue]
        fig = sns.pairplot(df_sns, hue="hue")
        st.pyplot(fig)

    st.markdown("### モデリング")
    ex = st.multiselect("説明変数を選択してください（複数選択可）", df_columns)
    ob = st.selectbox("目的変数を選択してください", df_columns)
    ml_menu = st.selectbox("実施する機械学習のタイプを選択してください",
                           ["重回帰分析", "ロジスティック回帰分析", "LightGBM", "Catboost"])

    if ml_menu == "重回帰分析":
        if st.button("実行"):
            lr = LinearRegression()
            df_ex = df[ex]
            df_ob = df[ob]
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=0.3)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            st.write("トレーニングスコア:", lr.score(X_train, y_train))
            st.write("テストスコア:", lr.score(X_test, y_test))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値'))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値'))
            st.plotly_chart(fig)

    elif ml_menu == "ロジスティック回帰分析":
        if st.button("実行"):
            lr = LogisticRegression()
            df_ex = df[ex]
            df_ob = df[ob]
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=0.3)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            st.write("トレーニングスコア:", lr.score(X_train, y_train))
            st.write("テストスコア:", lr.score(X_test, y_test))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値'))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値'))
            st.plotly_chart(fig)

    elif ml_menu == "LightGBM":
        if st.button("実行"):
            lgbm = lgb.LGBMRegressor()
            df_ex = df[ex]
            df_ob = df[ob]
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=0.3)
            lgbm.fit(X_train, y_train)
            y_pred = lgbm.predict(X_test)

            st.write("トレーニングスコア:", lgbm.score(X_train, y_train))
            st.write("テストスコア:", lgbm.score(X_test, y_test))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値'))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値'))
            st.plotly_chart(fig)

    elif ml_menu == "Catboost":
        if st.button("実行"):
            cb = CatBoostRegressor(verbose=0)
            df_ex = df[ex]
            df_ob = df[ob]
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=0.3)
            cb.fit(X_train, y_train)
            y_pred = cb.predict(X_test)

            st.write("トレーニングスコア:", cb.score(X_train, y_train))
            st.write("テストスコア:", cb.score(X_test, y_test))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値'))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値'))
            st.plotly_chart(fig)
