# ライブラリの読み込み
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier

# タイトル
st.title("機械学習アプリ")
st.write("streamlitで実装")

# 以下をサイドバーに表示
st.sidebar.markdown("### 機械学習に用いるcsvファイルを入力してください")
# ファイルアップロード
uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=False)
# ファイルがアップロードされたら以下が実行される
if uploaded_files:
    df = pd.read_csv(uploaded_files)
    df_columns = df.columns
    # データフレームを表示
    st.markdown("### 入力データ")
    st.dataframe(df.style.highlight_max(axis=0))
    # matplotlibで可視化。X軸,Y軸を選択できる
    st.markdown("### 可視化 単変量")
    # データフレームのカラムを選択オプションに設定する
    x = st.selectbox("X軸", df_columns)
    y = st.selectbox("Y軸", df_columns)
    # 選択した変数を用いてmatplotlibで可視化
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(df[x], df[y])
    plt.xlabel(x, fontsize=18)
    plt.ylabel(y, fontsize=18)
    st.pyplot(plt)

    # seabornのペアプロットで可視化。複数の変数を選択できる。
    st.markdown("### 可視化 ペアプロット")
    # データフレームのカラムを選択肢にする。複数選択
    item = st.multiselect("可視化するカラム", df_columns)
    # 散布図の色分け基準を１つ選択する。カテゴリ変数を想定
    hue = st.selectbox("色の基準", df_columns)

    # 実行ボタン
    execute_pairplot = st.button("ペアプロット描画")
    # 実行ボタンを押したら下記を表示
    if execute_pairplot:
        df_sns = df[item].copy()
        df_sns["hue"] = df[hue]

        # streamlit上でseabornのペアプロットを表示させる
        fig = sns.pairplot(df_sns, hue="hue")
        st.pyplot(plt)

    st.markdown("### モデリング")
    # 説明変数は複数選択式
    ex = st.multiselect("説明変数を選択してください（複数選択可）", df_columns)

    # 目的変数は一つ
    ob = st.selectbox("目的変数を選択してください", df_columns)

    # 機械学習のタイプを選択する。
    ml_menu = st.selectbox("実施する機械学習のタイプを選択してください",
                           ["重回帰分析", "ロジスティック回帰分析", "LightGBM", "Catboost"])

    # 機械学習のタイプにより以下の処理が分岐
    if ml_menu == "重回帰分析":
        st.markdown("#### 機械学習を実行します")
        execute = st.button("実行")

        lr = LinearRegression()
        # 実行ボタンを押したら下記が進む
        if execute:
            df_ex = df[ex]
            df_ob = df[ob]
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=0.3)
            lr.fit(X_train, y_train)
            # プログレスバー（ここでは、やってる感だけ）
            my_bar = st.progress(0)

            for percent_complete in range(100):
                time.sleep(0.02)
                my_bar.progress(percent_complete + 1)

            # metricsで指標を強調表示させる
            col1, col2 = st.columns(2)
            col1.metric(label="トレーニングスコア", value=lr.score(X_train, y_train))
            col2.metric(label="テストスコア", value=lr.score(X_test, y_test))

            # 予測結果と実際の値をplotlyで可視化
            y_pred = lr.predict(X_test)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値'))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値'))
            st.plotly_chart(fig)

    elif ml_menu == "ロジスティック回帰分析":
        st.markdown("#### 機械学習を実行します")
        execute = st.button("実行")

        lr = LogisticRegression()

        # 実行ボタンを押したら下記が進む
        if execute:
            df_ex = df[ex]
            df_ob = df[ob]
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=0.3)
            lr.fit(X_train, y_train)
            # プログレスバー（ここでは、やってる感だけ）
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)
                my_bar.progress(percent_complete + 1)

            col1, col2 = st.columns(2)
            col1.metric(label="トレーニングスコア", value=lr.score(X_train, y_train))
            col2.metric(label="テストスコア", value=lr.score(X_test, y_test))

            # 予測結果と実際の値をplotlyで可視化
            y_pred = lr.predict(X_test)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値'))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値'))
            st.plotly_chart(fig)

    elif ml_menu == "LightGBM":
        st.markdown("#### 機械学習を実行します")
        execute = st.button("実行")

        lgbm = lgb.LGBMRegressor()

        # 実行ボタンを押したら下記が進む
        if execute:
            df_ex = df[ex]
            df_ob = df[ob]
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=0.3)
            lgbm.fit(X_train, y_train)
            # プログレスバー（ここでは、やってる感だけ）
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)
                my_bar.progress(percent_complete + 1)

            col1, col2 = st.columns(2)
            col1.metric(label="トレーニングスコア", value=lgbm.score(X_train, y_train))
            col2.metric(label="テストスコア", value=lgbm.score(X_test, y_test))

            # 予測結果と実際の値をplotlyで可視化
            y_pred = lgbm.predict(X_test)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値'))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値'))
            st.plotly_chart(fig)

    elif ml_menu == "Catboost":
        st.markdown("#### 機械学習を実行します")
        execute = st.button("実行")

        cb = CatBoostRegressor(verbose=0)

        # 実行ボタンを押したら下記が進む
        if execute:
            df_ex = df[ex]
            df_ob = df[ob]
            X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size=0.3)
            cb.fit(X_train, y_train)
            # プログレスバー（ここでは、やってる感だけ）
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)
                my_bar.progress(percent_complete + 1)

            col1, col2 = st.columns(2)
            col1.metric(label="トレーニングスコア", value=cb.score(X_train, y_train))
            col2.metric(label="テストスコア", value=cb.score(X_test, y_test))

            # 予測結果と実際の値をplotlyで可視化
            y_pred = cb.predict(X_test)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='実際の値'))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='予測値'))
            st.plotly_chart(fig)


