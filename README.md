# Agricultural_Analysis

## LSTM 網路   
Long Short Term Momery（LSTM）是一種 RNN (Recurrent Neural Net-work) 特殊的類型， 由 Hochreiter & Schmidhuber (1997) 提出，Alex Graves(2005) 進行改良和推廣。根據文獻LSTM可解決不同領域問題，特別是在時間序列預測問題效果相當顯著。因作物生長環境資料是以固定時間進行紀錄，故採LSTM模型預測相關作物生長資訊達到良好成效。  
以下針對Long Short Term Momery（LSTM）做一說明，針對特定作物如洋蔥建構預測模型，將氣溫、日射量、積溫等資料與前一時間(t-1) LSTM預測結果輸入至時間t的LSTM中，經過選擇、遺忘與輸出三層結構得到預測結果並將預測結果進行記憶，下表第三點將會簡易說明運作原理，傳送往下一個時間(t+1)的LSTM預測，以此類推最後取得最終預測資料。  
