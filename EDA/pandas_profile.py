# Pandas-Profilingをインストール
!pip install https://github.com/ydataai/pandas-profiling/archive/master.zip

# Pandas-Profilingの依存関係パッケージのダウングレード
!pip install markupsafe==2.0.1

# 先ほどインストールしたPandas-Profilingをインポート
from pandas_profiling import ProfileReport

# Pandas-ProfilingにパッケージされているProfileReport関数の引数に、EDAを行いたいDataFrame形式のデータを渡す。
profile_df = ProfileReport(df, explorative=True)

# これを実行するとカレントディレクトリに可視化レポートのファイルが自動保存される。
profile_df.to_file("profile_df.html")

# 可視化レポートを確認する。
profile_df
