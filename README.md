# JSAP_Radiation



## 概要

応用物理学会　放射線分科会用  
機械学習の基本コード  

* Regression：回帰のサンプル  
  
* Classification：分類のサンプル  

※初学者を対象とした最小限のコードとしています


## データ
### data.csv
* 回帰  
[x座標, y座標, z座標1, z座標2]の順に格納  
z座標1：下部「回帰」の左グラフの縦軸データ、z座標2：同右グラフの縦軸データ

* 分類  
[x, y, クラス]の順に格納  
クラス：0または1（下部「分類」グラフの橙色が0、青色が1）

※x,y座標の値は両者で共通


## 必要ライブラリ

Python 3.7  
torch 1.11.0  
numpy 1.21.6  
matplotlib 3.5.2  
seaborn 0.11.2  

※回帰と分類で共通



## 学習
<img width="200" alt="Reg" src="https://user-images.githubusercontent.com/106053283/169779465-e59cf406-7f4e-46a9-80f0-f8095b74e406.jpg">


上記ライブラリをインストール後、学習用ソース「train.py」を実行  

「data.csv」のデータの学習が開始される  

300エポックの学習の後、「model.pth」が生成される    



## 推論
<img width="200" alt="Reg" src="https://user-images.githubusercontent.com/106053283/169779535-2be8ab64-5193-4e6c-86fb-fa1243e5e5db.jpg">


上記学習を実行後、推論用ソース「inference.py」を実行  

以下の結果が得られる  

* 回帰  
<img width="486" alt="Reg" src="https://user-images.githubusercontent.com/106053283/169761830-2315b2cb-c900-48db-95af-faea746063c6.png">
→マウスドラッグの操作でグラフの回転が可能      
  


* 分類  
<img width="300" alt="Clas" src="https://user-images.githubusercontent.com/106053283/169764248-8039aefe-f65c-430b-8135-a47a247b34db.png">
→マウスで左クリックした点の座標が青と橙色どちらに属するか判定する  






## コード比較

* sampleLayers.py  
出力層のみ異なる  
（回帰は全結合、分類は全結合後にソフトマックス関数）  
<img width="475" alt="layer" src="https://user-images.githubusercontent.com/106053283/169775820-2c1f289c-1461-4bf8-a03d-4ce7c3d0c0a9.png">


* train.py  
損失関数とデータ型のみ異なる  
（回帰は平均二乗誤差、分類は交差エントロピー誤差）  
<img width="476" alt="train" src="https://user-images.githubusercontent.com/106053283/169775025-304425ff-2d8c-4fe5-a3b7-3752f9c1e2b4.png">



## まとめ
機械学習の２大タスク「回帰」と「分類」はほぼ同じアルゴリズムである  
違いは、  
* 出力層（回帰は全結合、分類は全結合後にソフトマックス関数）  
* 損失関数（回帰は平均二乗誤差、分類は交差エントロピー誤差）  
