```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Malgun Gothic'

import seaborn as sns

sns.set_theme(style="white", color_codes=True)
sns.set(style = "white", font = "Malgun Gothic", rc = {'figure.figsize':(20,10)})
```


```python
import numpy as np
import pandas as pd

df = pd.read_csv("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/P1_dataset.csv", encoding = 'cp949')
```


```python
train_df = df
y = df.iloc[:,15]
```


```python
train_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOODNO</th>
      <th>SUCCYMD</th>
      <th>CARNM</th>
      <th>CHASNO</th>
      <th>CARREGIYMD</th>
      <th>YEAR</th>
      <th>MISSNM</th>
      <th>FUELNM</th>
      <th>COLOR</th>
      <th>EXHA</th>
      <th>...</th>
      <th>SUNLOOPPANORAMA</th>
      <th>SUNLOOPCOMMON</th>
      <th>SUNLOOPDUAL</th>
      <th>DIS</th>
      <th>TCS</th>
      <th>AB1</th>
      <th>ETC</th>
      <th>AV</th>
      <th>EPS</th>
      <th>ECS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1512A2469</td>
      <td>20160105</td>
      <td>모닝 LPi LX 기본 블랙 프리미엄</td>
      <td>KNABK518BBT020038</td>
      <td>20100616.0</td>
      <td>2011</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>C</td>
      <td>1000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1512A2364</td>
      <td>20160105</td>
      <td>K3 1.6 가솔린(4도어) Nobless</td>
      <td>KNAFZ412BDA040155</td>
      <td>20130207.0</td>
      <td>2013</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>1600</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1512A2319</td>
      <td>20160105</td>
      <td>K3 1.6 가솔린(4도어) Trendy</td>
      <td>KNAFK412BEA206484</td>
      <td>20140128.0</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>1591</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1512A1643</td>
      <td>20160105</td>
      <td>K5 2.0LPI 렌터카 디럭스</td>
      <td>KNAGN418BDA366086</td>
      <td>20121218.0</td>
      <td>2013</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>B</td>
      <td>2000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1512A1371</td>
      <td>20160105</td>
      <td>K5 2.0LPI 렌터카 스마트</td>
      <td>KNAGN415BBA140279</td>
      <td>20110428.0</td>
      <td>2011</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>D</td>
      <td>2000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36789</th>
      <td>1905C0711</td>
      <td>20190523</td>
      <td>더뉴모닝 1.0가솔린 Deluxe</td>
      <td>KNABE511BGT021147</td>
      <td>20150216.0</td>
      <td>2016</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>998</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36790</th>
      <td>1905C0899</td>
      <td>20190523</td>
      <td>더뉴K9 V6 3.8 EXECUTIVE(이그제큐티브)</td>
      <td>KNALT413BFS025606</td>
      <td>20141218.0</td>
      <td>2015</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>3778</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36791</th>
      <td>1905C0140</td>
      <td>20190523</td>
      <td>더뉴K9 V6 3.3 PRESTIGE(프레스티지)</td>
      <td>KNALT411BFS028524</td>
      <td>20150506.0</td>
      <td>2015</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>B</td>
      <td>3342</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36792</th>
      <td>1905C0350</td>
      <td>20190523</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA544130</td>
      <td>20140718.0</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>A</td>
      <td>1999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36793</th>
      <td>1905C0427</td>
      <td>20190523</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532709</td>
      <td>20140529.0</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>A</td>
      <td>1999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>36794 rows × 104 columns</p>
</div>




```python
#SUCCYMD(낙찰일자) datetime 형식으로 변경
train_df["SUCCYMD"] = train_df["SUCCYMD"].astype(str)
train_df["SUCCYMD"] = pd.to_datetime(train_df["SUCCYMD"])

#CARREGIYMD 결측치제거
train_df.dropna(subset=['CARREGIYMD'], inplace = True)

#CARREGIYMD(차량등록일) datetime 형식으로 변경
train_df["CARREGIYMD"] = train_df["CARREGIYMD"].astype(str)
train_df["CARREGIYMD"] = train_df["CARREGIYMD"].apply(lambda x : x[:-2])
train_df["CARREGIYMD"] = pd.to_datetime(train_df["CARREGIYMD"])

#USED_DAY(차 사용시간) 파생변수생성
train_df["USED_DAY"] = train_df["SUCCYMD"] - train_df["CARREGIYMD"]
train_df["USED_DAY"] = train_df["USED_DAY"].astype(str)
train_df["USED_DAY"] = train_df["USED_DAY"].apply(lambda x : x[:-5])
train_df["USED_DAY"] = train_df["USED_DAY"].astype(int)

#이상치(USED_DAY가 음수인 경우) drop
index = train_df[train_df["USED_DAY"] < 0].index
print(len(train_df))
train_df = train_df.drop(index)
print(len(train_df))
```

    36793
    36786



```python
data_for_scatter = train_df[["USED_DAY", "SUCCPRIC"]]
sns.set(style = "white", font = "Malgun Gothic", rc = {'figure.figsize':(30,10)})
sns.relplot(x="USED_DAY", y="SUCCPRIC",palette="ch:r=-.2,d=.3_r", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/USED_DAY.png", transparent=True)
```


![png](output_5_0.png)
    



```python
train_df['INNEEXPOCLASCD_YN'].unique()
```




    array(['X', 'O'], dtype=object)




```python
y
```




    0         4300000
    1        11650000
    2        12350000
    3         5900000
    4         4730000
               ...   
    36789     5910000
    36790    19200000
    36791    18200000
    36792     5800000
    36793     5700000
    Name: SUCCPRIC, Length: 36794, dtype: int64




```python
#boxplot0 = train_df[["MISSNM","SUCCPRIC"]].boxplot(by = 'MISSNM')
data_for_scatter = train_df
sns.catplot(x="FUELNM", y="SUCCPRIC", hue = "MISSNM", capsize = .2, palette="YlGnBu_d", kind = "point", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/Spec.png", transparent=True)
```


​    
![png](output_8_0.png)
​    



```python
data_for_scatter = train_df[["MISSNM", "SUCCPRIC"]]
sns.barplot(x="MISSNM", y="SUCCPRIC", data=data_for_scatter)
sns.despine()
```


​    
![png](output_9_0.png)
​    



```python
#boxplot2 = train_df[["COLOR","SUCCPRIC"]].boxplot(by = 'COLOR')

data_for_scatter = train_df[["COLOR", "SUCCPRIC"]]
sns.barplot(x="COLOR", y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/COLOR.png", transparent=True)
```


​    
![png](output_10_0.png)
​    



```python
#plt.scatter(train_df['EXHA'], train_df["SUCCPRIC"])
data_for_scatter = train_df[["EXHA", "SUCCPRIC"]]
sns.scatterplot(x="EXHA", y="SUCCPRIC",data=data_for_scatter)

sns.despine()
```


​    
![png](output_11_0.png)
​    



```python
data1 = train_df[train_df["SHIPPING_PRICE"].notnull()]
index1 = data1.index

# ch_grade_price_idx : shipping price가 0인 데이터들에대해 NC_GRADE_PRICE로 바꿀 index
ch_grade_price_idx = data1[data1["SHIPPING_PRICE"] < 1].index


data2 = train_df[train_df["SHIPPING_PRICE"].isnull()]

data3 = data2[data2["NC_GRADE_PRICE"].notnull()]
index2 = data3.index
index2 = list(set(index2).union(set(ch_grade_price_idx)))


data4 = data2[data2["NC_GRADE_PRICE"].isnull()]
index3 = data4.index

train_df["pre_price"] = 0
train_df.loc[index1, "pre_price"] = train_df.loc[index1, "SHIPPING_PRICE"]
train_df.loc[index2, "pre_price"] = train_df.loc[index2, "NC_GRADE_PRICE"]
train_df.loc[index3, "pre_price"] = train_df.loc[index3, "NEWCARPRIC"]
```


```python
data_for_scatter = train_df[["pre_price", "SUCCPRIC"]]
sns.scatterplot(x="pre_price", y="SUCCPRIC",data=data_for_scatter)

sns.despine()
```


​    
![png](output_13_0.png)
​    



```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOODNO</th>
      <th>SUCCYMD</th>
      <th>CARNM</th>
      <th>CHASNO</th>
      <th>CARREGIYMD</th>
      <th>YEAR</th>
      <th>MISSNM</th>
      <th>FUELNM</th>
      <th>COLOR</th>
      <th>EXHA</th>
      <th>...</th>
      <th>SUNLOOPDUAL</th>
      <th>DIS</th>
      <th>TCS</th>
      <th>AB1</th>
      <th>ETC</th>
      <th>AV</th>
      <th>EPS</th>
      <th>ECS</th>
      <th>USED_DAY</th>
      <th>pre_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1512A2469</td>
      <td>2016-01-05</td>
      <td>모닝 LPi LX 기본 블랙 프리미엄</td>
      <td>KNABK518BBT020038</td>
      <td>2010-06-16</td>
      <td>2011</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>C</td>
      <td>1000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2029</td>
      <td>11310000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1512A2364</td>
      <td>2016-01-05</td>
      <td>K3 1.6 가솔린(4도어) Nobless</td>
      <td>KNAFZ412BDA040155</td>
      <td>2013-02-07</td>
      <td>2013</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>1600</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1062</td>
      <td>19750000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1512A2319</td>
      <td>2016-01-05</td>
      <td>K3 1.6 가솔린(4도어) Trendy</td>
      <td>KNAFK412BEA206484</td>
      <td>2014-01-28</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>1591</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>707</td>
      <td>19340000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1512A1643</td>
      <td>2016-01-05</td>
      <td>K5 2.0LPI 렌터카 디럭스</td>
      <td>KNAGN418BDA366086</td>
      <td>2012-12-18</td>
      <td>2013</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>B</td>
      <td>2000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1113</td>
      <td>17680000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1512A1371</td>
      <td>2016-01-05</td>
      <td>K5 2.0LPI 렌터카 스마트</td>
      <td>KNAGN415BBA140279</td>
      <td>2011-04-28</td>
      <td>2011</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>D</td>
      <td>2000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1713</td>
      <td>15800000.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 106 columns</p>
</div>




```python
#plt.scatter(train_df['TRAVDIST'], train_df["SUCCPRIC"])
data_for_scatter = train_df[["TRAVDIST", "SUCCPRIC"]]
sns.scatterplot(x="TRAVDIST", y="SUCCPRIC",data=data_for_scatter)

sns.despine()
```


​    
![png](output_15_0.png)
​    



```python
#boxplot3 = train_df[["USEUSENM","SUCCPRIC"]].boxplot(by = 'USEUSENM')

'''
data_for_scatter = train_df[["USEUSENM", "SUCCPRIC"]]
sns.catplot(x="USEUSENM", y="SUCCPRIC", data=data_for_scatter)
sns.despine()
'''


data_for_scatter = train_df[["USEUSENM", "SUCCPRIC"]]
sns.barplot(x="USEUSENM", y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/USEUSE.png", transparent=True)
```


​    
![png](output_16_0.png)
​    



```python
#boxplot4 = train_df[["OWNECLASNM","SUCCPRIC"]].boxplot(by = 'OWNECLASNM')

data_for_scatter = train_df[["OWNECLASNM", "SUCCPRIC"]]
sns.barplot(x="OWNECLASNM", y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/OWNE.png", transparent=True)
```


​    
![png](output_17_0.png)
​    



```python
#boxplot4 = train_df[["INNEEXPOCLASCD_YN","SUCCPRIC"]].boxplot(by = 'INNEEXPOCLASCD_YN')

data_for_scatter = train_df[["INNEEXPOCLASCD_YN", "SUCCPRIC"]]
sns.barplot(x="INNEEXPOCLASCD_YN", y="SUCCPRIC", data=data_for_scatter)
sns.despine()
```


​    
![png](output_18_0.png)
​    



```python
data_for_scatter = train_df[["pre_price", "SUCCPRIC", "INNEEXPOCLASCD_YN"]]
sns.relplot(x="pre_price", y="SUCCPRIC", hue = "INNEEXPOCLASCD_YN",data=data_for_scatter)
plt.axvline(0.5 * 10**7, 0, train_df["SUCCPRIC"].max(), color = "red")
sns.despine()

```


    ---------------------------------------------------------------------------
    
    KeyError                                  Traceback (most recent call last)
    
    ~\AppData\Local\Temp/ipykernel_10328/2523919631.py in <module>
    ----> 1 data_for_scatter = train_df[["pre_price", "SUCCPRIC", "INNEEXPOCLASCD_YN"]]
          2 sns.relplot(x="pre_price", y="SUCCPRIC", hue = "INNEEXPOCLASCD_YN",data=data_for_scatter)
          3 plt.axvline(0.5 * 10**7, 0, train_df["SUCCPRIC"].max(), color = "red")
          4 sns.despine()


    ~\anaconda3\lib\site-packages\pandas\core\frame.py in __getitem__(self, key)
       3462             if is_iterator(key):
       3463                 key = list(key)
    -> 3464             indexer = self.loc._get_listlike_indexer(key, axis=1)[1]
       3465 
       3466         # take() does not accept boolean indexers


    ~\anaconda3\lib\site-packages\pandas\core\indexing.py in _get_listlike_indexer(self, key, axis)
       1312             keyarr, indexer, new_indexer = ax._reindex_non_unique(keyarr)
       1313 
    -> 1314         self._validate_read_indexer(keyarr, indexer, axis)
       1315 
       1316         if needs_i8_conversion(ax.dtype) or isinstance(


    ~\anaconda3\lib\site-packages\pandas\core\indexing.py in _validate_read_indexer(self, key, indexer, axis)
       1375 
       1376             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
    -> 1377             raise KeyError(f"{not_found} not in index")
       1378 
       1379 


    KeyError: "['pre_price'] not in index"



```python
#boxplot4 = train_df[["INNEEXPOCLASCD_YN","SUCCPRIC"]].boxplot(by = 'INNEEXPOCLASCD_YN')

data_for_scatter = train_df[["INNEEXPOCLASCD_YN", "SUCCPRIC"]]
sns.catplot(x="INNEEXPOCLASCD_YN", y="SUCCPRIC", data=data_for_scatter)
sns.despine()
```


​    
![png](output_20_0.png)
​    



```python

```


```python

```


```python

```


```python
import numpy as np
import pandas as pd

df = pd.read_csv("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/P1_dataset.csv", encoding = 'cp949')
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
train_df = df
train_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOODNO</th>
      <th>SUCCYMD</th>
      <th>CARNM</th>
      <th>CHASNO</th>
      <th>CARREGIYMD</th>
      <th>YEAR</th>
      <th>MISSNM</th>
      <th>FUELNM</th>
      <th>COLOR</th>
      <th>EXHA</th>
      <th>...</th>
      <th>SUNLOOPPANORAMA</th>
      <th>SUNLOOPCOMMON</th>
      <th>SUNLOOPDUAL</th>
      <th>DIS</th>
      <th>TCS</th>
      <th>AB1</th>
      <th>ETC</th>
      <th>AV</th>
      <th>EPS</th>
      <th>ECS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1512A2469</td>
      <td>20160105</td>
      <td>모닝 LPi LX 기본 블랙 프리미엄</td>
      <td>KNABK518BBT020038</td>
      <td>20100616.0</td>
      <td>2011</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>C</td>
      <td>1000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1512A2364</td>
      <td>20160105</td>
      <td>K3 1.6 가솔린(4도어) Nobless</td>
      <td>KNAFZ412BDA040155</td>
      <td>20130207.0</td>
      <td>2013</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>1600</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1512A2319</td>
      <td>20160105</td>
      <td>K3 1.6 가솔린(4도어) Trendy</td>
      <td>KNAFK412BEA206484</td>
      <td>20140128.0</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>1591</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1512A1643</td>
      <td>20160105</td>
      <td>K5 2.0LPI 렌터카 디럭스</td>
      <td>KNAGN418BDA366086</td>
      <td>20121218.0</td>
      <td>2013</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>B</td>
      <td>2000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1512A1371</td>
      <td>20160105</td>
      <td>K5 2.0LPI 렌터카 스마트</td>
      <td>KNAGN415BBA140279</td>
      <td>20110428.0</td>
      <td>2011</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>D</td>
      <td>2000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36789</th>
      <td>1905C0711</td>
      <td>20190523</td>
      <td>더뉴모닝 1.0가솔린 Deluxe</td>
      <td>KNABE511BGT021147</td>
      <td>20150216.0</td>
      <td>2016</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>998</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36790</th>
      <td>1905C0899</td>
      <td>20190523</td>
      <td>더뉴K9 V6 3.8 EXECUTIVE(이그제큐티브)</td>
      <td>KNALT413BFS025606</td>
      <td>20141218.0</td>
      <td>2015</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>3778</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36791</th>
      <td>1905C0140</td>
      <td>20190523</td>
      <td>더뉴K9 V6 3.3 PRESTIGE(프레스티지)</td>
      <td>KNALT411BFS028524</td>
      <td>20150506.0</td>
      <td>2015</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>B</td>
      <td>3342</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36792</th>
      <td>1905C0350</td>
      <td>20190523</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA544130</td>
      <td>20140718.0</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>A</td>
      <td>1999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36793</th>
      <td>1905C0427</td>
      <td>20190523</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532709</td>
      <td>20140529.0</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>A</td>
      <td>1999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>36794 rows × 104 columns</p>
</div>




```python

#SUCCYMD(낙찰일자) datetime 형식으로 변경
train_df["SUCCYMD"] = train_df["SUCCYMD"].astype(str)
train_df["SUCCYMD"] = pd.to_datetime(train_df["SUCCYMD"])

#CARREGIYMD 결측치제거
train_df.dropna(subset=['CARREGIYMD'], inplace = True)

#CARREGIYMD(차량등록일) datetime 형식으로 변경
train_df["CARREGIYMD"] = train_df["CARREGIYMD"].astype(str)
train_df["CARREGIYMD"] = train_df["CARREGIYMD"].apply(lambda x : x[:-2])
train_df["CARREGIYMD"] = pd.to_datetime(train_df["CARREGIYMD"])

#USED_DAY(차 사용시간) 파생변수생성
train_df["USED_DAY"] = train_df["SUCCYMD"] - train_df["CARREGIYMD"]
train_df["USED_DAY"] = train_df["USED_DAY"].astype(str)
train_df["USED_DAY"] = train_df["USED_DAY"].apply(lambda x : x[:-5])
train_df["USED_DAY"] = train_df["USED_DAY"].astype(int)

#이상치(USED_DAY가 음수인 경우) drop
index = train_df[train_df["USED_DAY"] < 0].index
print(len(train_df))
train_df = train_df.drop(index)
print(len(train_df))
```

    36793
    36786



```python
#교환/판금/용접 변수 처리
train_df['FENDER']=train_df['FRONT_LEFT_FENDER']+train_df['FRONT_RIGHT_FENDER']+train_df['LEFT_REAR_FENDER']+train_df['RIGHT_REAR_FENDER']
train_df['DOOR']=train_df['FRONT_LEFT_DOOR']+train_df['FRONT_RIGHT_DOOR']+train_df['BACK_LEFT_DOOR']+train_df['BACK_RIGHT_DOOR']
train_df['STEP']=train_df['LEFT_STEP']+train_df['RIGHT_STEP']
train_df['FILER']=train_df['LEFT_FILER_A']+train_df['RIGHT_FILER_A']+train_df['LEFT_FILER_B']+train_df['RIGHT_FILER_B']+train_df['LEFT_FILER_C']+train_df['RIGHT_FILER_C']
train_df['PANEL']=train_df['FRONT_PANNEL']+train_df['BACK_PANEL1']+train_df['LEFT_INSIDE_PANEL']+train_df['RIGHT_INSIDE_PANEL']+train_df['DASH_PANEL']+train_df['SHEET_PANEL']+train_df['LEFT_QUARTER']+train_df['RIGHT_QUARTER']+train_df['FLOOR_PANEL']+train_df['LEFT_SIDE_PANEL']+train_df['RIGHT_SIDE_PANEL']+train_df['LEFT_REAR_CORNER_PANEL']+train_df['RIGHT_REAR_CORNER_PANEL']+train_df['BACK_PANEL2']+train_df['LEFT_CORNER_PANEL']+train_df['RIGHT_CORNER_PANEL']+train_df['LEFT_SKIRT_PANEL']+train_df['RIGHT_SKIRT_PANEL']+train_df['LEFT_INSIDE_SHEETING']+train_df['RIGHT_INSIDE_SHEETING']+train_df['LEFT_REAR_INSIDE_PANEL_SHEETING']+train_df['RIGHT_REAR_INSIDE_PANEL_SHEETING']+train_df['DASH_PANEL_SHEETING']+train_df['SHEET_BACK_PANEL_SHEETING']+train_df['FLOOR_PANEL_SHEETING']+train_df['LEFT_SIDE_PANEL_SHEETING']+train_df['RIGHT_SIDE_PANEL_SHEETING']
train_df['WHEEL_HOUSE']=train_df['LEFT_WHEEL_HOUSE']+train_df['RIGHT_WHEEL_HOUSE']+train_df['LEFT_INSIDE_WHEEL_HOUSE']+train_df['RIGHT_INSIDE_WHEEL_HOUSE']+train_df['LEFT_REAR_WHEEL_HOUSE']+train_df['RIGHT_REAR_WHEEL_HOUSE']+train_df['LEFT_WHEEL_HOUSE_SHEETING']+train_df['RIGHT_WHEEL_HOUSE_SHEETING']+train_df['LEFT_REAR_WHEEL_HOUSE_SHEETING']+train_df['RIGHT_REAR_WHEEL_HOUSE_SHEETING']
train_df['FRAME']=train_df['SIDE_MEMBER_FRAME']+train_df['SIDE_MEMBER_FRAME2']+train_df['SIDE_MEMBER_FRAME_SHEETING']
train_df['TRUNK1']=train_df['TRUNK_FLOOR']+train_df['TRUNK_FLOOR_SHEETING']+train_df['TRUNK']
```


```python
drop_col=train_df.loc[:,'FRONT_LEFT_FENDER':'RIGHT_SIDE_PANEL_SHEETING'].columns
train_df=train_df.drop(drop_col,axis=1)
```


```python
#도어, 프런트 펜더 등 외판 부위에 대한 판금·용접·교환은 단순 수리로 분류돼 사고차로 간주되지 않는다. 
#non_Accident = train_df['DOOR'] + train_df["FENDER"] + train_df["STEP"] + train_df["WHEEL_HOUSE"] + train_df["TRUNK1"]
Accident = train_df["FILER"] + train_df["PANEL"] + train_df["FRAME"]

train_df["ACCIDENT"] = Accident.apply(lambda x: x != 0).astype(int)
train_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOODNO</th>
      <th>SUCCYMD</th>
      <th>CARNM</th>
      <th>CHASNO</th>
      <th>CARREGIYMD</th>
      <th>YEAR</th>
      <th>MISSNM</th>
      <th>FUELNM</th>
      <th>COLOR</th>
      <th>EXHA</th>
      <th>...</th>
      <th>USED_DAY</th>
      <th>FENDER</th>
      <th>DOOR</th>
      <th>STEP</th>
      <th>FILER</th>
      <th>PANEL</th>
      <th>WHEEL_HOUSE</th>
      <th>FRAME</th>
      <th>TRUNK1</th>
      <th>ACCIDENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1512A2469</td>
      <td>2016-01-05</td>
      <td>모닝 LPi LX 기본 블랙 프리미엄</td>
      <td>KNABK518BBT020038</td>
      <td>2010-06-16</td>
      <td>2011</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>C</td>
      <td>1000</td>
      <td>...</td>
      <td>2029</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1512A2364</td>
      <td>2016-01-05</td>
      <td>K3 1.6 가솔린(4도어) Nobless</td>
      <td>KNAFZ412BDA040155</td>
      <td>2013-02-07</td>
      <td>2013</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>1600</td>
      <td>...</td>
      <td>1062</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1512A2319</td>
      <td>2016-01-05</td>
      <td>K3 1.6 가솔린(4도어) Trendy</td>
      <td>KNAFK412BEA206484</td>
      <td>2014-01-28</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>1591</td>
      <td>...</td>
      <td>707</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1512A1643</td>
      <td>2016-01-05</td>
      <td>K5 2.0LPI 렌터카 디럭스</td>
      <td>KNAGN418BDA366086</td>
      <td>2012-12-18</td>
      <td>2013</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>B</td>
      <td>2000</td>
      <td>...</td>
      <td>1113</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1512A1371</td>
      <td>2016-01-05</td>
      <td>K5 2.0LPI 렌터카 스마트</td>
      <td>KNAGN415BBA140279</td>
      <td>2011-04-28</td>
      <td>2011</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>D</td>
      <td>2000</td>
      <td>...</td>
      <td>1713</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36789</th>
      <td>1905C0711</td>
      <td>2019-05-23</td>
      <td>더뉴모닝 1.0가솔린 Deluxe</td>
      <td>KNABE511BGT021147</td>
      <td>2015-02-16</td>
      <td>2016</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>998</td>
      <td>...</td>
      <td>1557</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36790</th>
      <td>1905C0899</td>
      <td>2019-05-23</td>
      <td>더뉴K9 V6 3.8 EXECUTIVE(이그제큐티브)</td>
      <td>KNALT413BFS025606</td>
      <td>2014-12-18</td>
      <td>2015</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>3778</td>
      <td>...</td>
      <td>1617</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36791</th>
      <td>1905C0140</td>
      <td>2019-05-23</td>
      <td>더뉴K9 V6 3.3 PRESTIGE(프레스티지)</td>
      <td>KNALT411BFS028524</td>
      <td>2015-05-06</td>
      <td>2015</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>B</td>
      <td>3342</td>
      <td>...</td>
      <td>1478</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36792</th>
      <td>1905C0350</td>
      <td>2019-05-23</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA544130</td>
      <td>2014-07-18</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>A</td>
      <td>1999</td>
      <td>...</td>
      <td>1770</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36793</th>
      <td>1905C0427</td>
      <td>2019-05-23</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532709</td>
      <td>2014-05-29</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>A</td>
      <td>1999</td>
      <td>...</td>
      <td>1820</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>36786 rows × 55 columns</p>
</div>




```python
data_for_scatter = train_df[["ACCIDENT", "SUCCPRIC"]]
sns.catplot(x="ACCIDENT", y="SUCCPRIC", data=data_for_scatter)
sns.despine()
```


​    
![png](output_30_0.png)
​    



```python
#plt.scatter(train_df['TRAVDIST'], train_df["SUCCPRIC"])
data_for_scatter = train_df[["TRAVDIST", "SUCCPRIC", "ACCIDENT"]]
sns.scatterplot(x="TRAVDIST", y="SUCCPRIC", hue = "ACCIDENT", palette = "ch:r=-.2,d=.3_r",data=data_for_scatter)

sns.despine()

plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/사고유무 주행거리.png", transparent=True)
```


​    
![png](output_31_0.png)
​    



```python
def one_hot_encoder(df):
    result = pd.get_dummies(df)
    return(result)

MISS = one_hot_encoder(train_df["MISSNM"])
FUEL = one_hot_encoder(train_df["FUELNM"])
COLOR = one_hot_encoder(train_df["COLOR"])
USEUSE = one_hot_encoder(train_df["USEUSENM"])
OWNE = one_hot_encoder(train_df["OWNECLASNM"])


INNEEX = train_df['INNEEXPOCLASCD_YN'].apply(lambda x : x != "X").astype(int)

train_df = train_df.join(MISS)
train_df = train_df.drop("MISSNM",1)
train_df

train_df = train_df.join(FUEL)
train_df = train_df.drop("FUELNM",1)
train_df

train_df = train_df.join(COLOR)
train_df = train_df.drop("COLOR",1)
train_df

train_df = train_df.join(USEUSE)
train_df = train_df.drop("USEUSENM",1)
train_df

train_df = train_df.join(OWNE)
train_df = train_df.drop("OWNECLASNM",1)
train_df

train_df["INNEEXPOCLASCD_YN"] = INNEEX
train_df
```

    C:\Users\user\AppData\Local\Temp/ipykernel_24864/3265283029.py:15: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      train_df = train_df.drop("MISSNM",1)
    C:\Users\user\AppData\Local\Temp/ipykernel_24864/3265283029.py:19: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      train_df = train_df.drop("FUELNM",1)
    C:\Users\user\AppData\Local\Temp/ipykernel_24864/3265283029.py:23: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      train_df = train_df.drop("COLOR",1)
    C:\Users\user\AppData\Local\Temp/ipykernel_24864/3265283029.py:27: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      train_df = train_df.drop("USEUSENM",1)
    C:\Users\user\AppData\Local\Temp/ipykernel_24864/3265283029.py:31: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      train_df = train_df.drop("OWNECLASNM",1)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOODNO</th>
      <th>SUCCYMD</th>
      <th>CARNM</th>
      <th>CHASNO</th>
      <th>CARREGIYMD</th>
      <th>YEAR</th>
      <th>EXHA</th>
      <th>TRAVDIST</th>
      <th>INNEEXPOCLASCD_YN</th>
      <th>NEWCARPRIC</th>
      <th>...</th>
      <th>사업</th>
      <th>업무</th>
      <th>자가</th>
      <th>개인</th>
      <th>개인사업</th>
      <th>법인</th>
      <th>법인상품</th>
      <th>상품용</th>
      <th>재외국인</th>
      <th>종교단체</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1512A2469</td>
      <td>2016-01-05</td>
      <td>모닝 LPi LX 기본 블랙 프리미엄</td>
      <td>KNABK518BBT020038</td>
      <td>2010-06-16</td>
      <td>2011</td>
      <td>1000</td>
      <td>38480</td>
      <td>0</td>
      <td>10704916</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1512A2364</td>
      <td>2016-01-05</td>
      <td>K3 1.6 가솔린(4도어) Nobless</td>
      <td>KNAFZ412BDA040155</td>
      <td>2013-02-07</td>
      <td>2013</td>
      <td>1600</td>
      <td>62240</td>
      <td>0</td>
      <td>21230000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1512A2319</td>
      <td>2016-01-05</td>
      <td>K3 1.6 가솔린(4도어) Trendy</td>
      <td>KNAFK412BEA206484</td>
      <td>2014-01-28</td>
      <td>2014</td>
      <td>1591</td>
      <td>37926</td>
      <td>0</td>
      <td>18043152</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1512A1643</td>
      <td>2016-01-05</td>
      <td>K5 2.0LPI 렌터카 디럭스</td>
      <td>KNAGN418BDA366086</td>
      <td>2012-12-18</td>
      <td>2013</td>
      <td>2000</td>
      <td>110149</td>
      <td>0</td>
      <td>17280000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1512A1371</td>
      <td>2016-01-05</td>
      <td>K5 2.0LPI 렌터카 스마트</td>
      <td>KNAGN415BBA140279</td>
      <td>2011-04-28</td>
      <td>2011</td>
      <td>2000</td>
      <td>81675</td>
      <td>0</td>
      <td>15800000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36789</th>
      <td>1905C0711</td>
      <td>2019-05-23</td>
      <td>더뉴모닝 1.0가솔린 Deluxe</td>
      <td>KNABE511BGT021147</td>
      <td>2015-02-16</td>
      <td>2016</td>
      <td>998</td>
      <td>62180</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36790</th>
      <td>1905C0899</td>
      <td>2019-05-23</td>
      <td>더뉴K9 V6 3.8 EXECUTIVE(이그제큐티브)</td>
      <td>KNALT413BFS025606</td>
      <td>2014-12-18</td>
      <td>2015</td>
      <td>3778</td>
      <td>97801</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36791</th>
      <td>1905C0140</td>
      <td>2019-05-23</td>
      <td>더뉴K9 V6 3.3 PRESTIGE(프레스티지)</td>
      <td>KNALT411BFS028524</td>
      <td>2015-05-06</td>
      <td>2015</td>
      <td>3342</td>
      <td>153601</td>
      <td>0</td>
      <td>51780001</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36792</th>
      <td>1905C0350</td>
      <td>2019-05-23</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA544130</td>
      <td>2014-07-18</td>
      <td>2015</td>
      <td>1999</td>
      <td>140058</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36793</th>
      <td>1905C0427</td>
      <td>2019-05-23</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532709</td>
      <td>2014-05-29</td>
      <td>2015</td>
      <td>1999</td>
      <td>159467</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>36786 rows × 76 columns</p>
</div>




```python
f, ax = plt.subplots(2, 1, figsize = (12, 10))

#shipping price에 대한 
sns.scatterplot(x = 'SHIPPING_PRICE', y = 'SUCCPRIC', data = train_df, ax = ax[0])


data = train_df[train_df["SHIPPING_PRICE"].notnull()]
data = data[data['SHIPPING_PRICE'] < 1]

#shipping price가 0인 데이터들에대해 NC_GRADE_PRICE로 나타낸 scatterplot
sns.scatterplot(x = 'NC_GRADE_PRICE', y = 'SUCCPRIC',  data = data, ax = ax[1])
```




    <AxesSubplot:xlabel='NC_GRADE_PRICE', ylabel='SUCCPRIC'>




​    
![png](output_33_1.png)
​    



```python
data1 = train_df[train_df["SHIPPING_PRICE"].notnull()]
index1 = data1.index

# ch_grade_price_idx : shipping price가 0인 데이터들에대해 NC_GRADE_PRICE로 바꿀 index
ch_grade_price_idx = data1[data1["SHIPPING_PRICE"] < 1].index


data2 = train_df[train_df["SHIPPING_PRICE"].isnull()]

data3 = data2[data2["NC_GRADE_PRICE"].notnull()]
index2 = data3.index
index2 = list(set(index2).union(set(ch_grade_price_idx)))


data4 = data2[data2["NC_GRADE_PRICE"].isnull()]
index3 = data4.index

train_df["pre_price"] = 0
train_df.loc[index1, "pre_price"] = train_df.loc[index1, "SHIPPING_PRICE"]
train_df.loc[index2, "pre_price"] = train_df.loc[index2, "NC_GRADE_PRICE"]
train_df.loc[index3, "pre_price"] = train_df.loc[index3, "NEWCARPRIC"]
```


```python
sns.scatterplot(x = 'pre_price', y = 'SUCCPRIC',data = train_df)
plt.axvline(0.5 * 10**7, 0, train_df["SUCCPRIC"].max(), color = "red")
```




    <matplotlib.lines.Line2D at 0x25c831e5760>




​    
![png](output_35_1.png)
​    



```python
data_for_scatter = train_df
sns.scatterplot(x="pre_price", y="SUCCPRIC", hue = "ACCIDENT",data=data_for_scatter)

sns.despine()

plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/사고유무 사전가격.png", transparent=True)
```


​    
![png](output_36_0.png)
​    



```python
#이상치 정의(출고될 당시 자동차의 추정가격이 50만원이하인 data들)
index = train_df[train_df["pre_price"] < 0.5 * (10**7)].index
train_df = train_df.drop(index)

train_df = train_df.drop(["SHIPPING_PRICE", "NC_GRADE_PRICE", "NEWCARPRIC"], axis = 1)
train_df = train_df.reset_index(drop=True)
```


```python
train_df.loc[:,'ABS':'ECS']=train_df.loc[:,'ABS':'ECS'].astype('object')
train_df[['FLOODING','TOTAL_LOSS']]=train_df[['FLOODING','TOTAL_LOSS']].astype('object')
train_df = train_df.drop(["SUCCYMD",	"CARREGIYMD",	"YEAR"], axis = 1)
```


```python
train_df=train_df.drop(['GOODNO','CARNM','CHASNO','YEARCHK','MF_KEY','JOINCAR','NOTAVAILABLE'],axis=1)
train_df = train_df.dropna()
train_df.reset_index(drop = True, inplace=True)
```


```python
train_df.columns
```




    Index(['EXHA', 'TRAVDIST', 'INNEEXPOCLASCD_YN', 'SUCCPRIC', 'BONET',
           'FLOODING', 'TOTAL_LOSS', 'MJ_MODEL_KEY', 'DT_MODEL_KEY',
           'MJ_GRADE_KEY', 'DT_GRADE_KEY', 'NC_GRADE_KEY', 'ABS', 'AB2',
           'NAVIGATION', 'VDC', 'SMARTKEY', 'SUNLOOPPANORAMA', 'SUNLOOPCOMMON',
           'SUNLOOPDUAL', 'DIS', 'TCS', 'AB1', 'ETC', 'AV', 'EPS', 'ECS',
           'USED_DAY', 'FENDER', 'DOOR', 'STEP', 'FILER', 'PANEL', 'WHEEL_HOUSE',
           'FRAME', 'TRUNK1', 'ACCIDENT', 'A/T', 'CVT', 'M/T', 'Hybrid', 'LPG',
           '가솔린', '겸용', '디젤', '전기', 'A', 'B', 'C', 'D', 'F', '렌트', '리스', '사업',
           '업무', '자가', '개인', '개인사업', '법인', '법인상품', '상품용', '재외국인', '종교단체',
           'pre_price'],
          dtype='object')




```python
data_for_scatter = train_df
sns.catplot(x='ACCIDENT', y="SUCCPRIC", data=data_for_scatter, kind="bar")
sns.despine()

plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/ACCIDENT.png", transparent=True)
```


​    
![png](output_41_0.png)
​    



```python
sns.set_theme(style="white", color_codes=True)
sns.set(style = "white", font = "Malgun Gothic", rc = {'figure.figsize':(10,6)})
```


```python
data_for_scatter = train_df[['ABS', "SUCCPRIC"]]
sns.barplot(x='ABS', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/1.png", transparent=True)
```


​    
![png](output_43_0.png)
​    



```python
data_for_scatter = train_df[['AB2', "SUCCPRIC"]]
sns.barplot(x='AB2', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/2.png", transparent=True)
```


​    
![png](output_44_0.png)
​    



```python
data_for_scatter = train_df[['NAVIGATION', "SUCCPRIC"]]
sns.barplot(x='NAVIGATION', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/3.png", transparent=True)
```


​    
![png](output_45_0.png)
​    



```python
data_for_scatter = train_df[['VDC', "SUCCPRIC"]]
sns.barplot(x='VDC', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/4.png", transparent=True)
```


​    
![png](output_46_0.png)
​    



```python
data_for_scatter = train_df[['SMARTKEY', "SUCCPRIC"]]
sns.barplot(x='SMARTKEY', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/5.png", transparent=True)
```


​    
![png](output_47_0.png)
​    



```python
data_for_scatter = train_df[['SUNLOOPPANORAMA', "SUCCPRIC"]]
sns.barplot(x='SUNLOOPPANORAMA', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/6.png", transparent=True)
```


​    
![png](output_48_0.png)
​    



```python
data_for_scatter = train_df[['SUNLOOPCOMMON', "SUCCPRIC"]]
sns.barplot(x='SUNLOOPCOMMON', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/7.png", transparent=True)
```


​    
![png](output_49_0.png)
​    



```python
data_for_scatter = train_df[['SUNLOOPDUAL', "SUCCPRIC"]]
sns.barplot(x='SUNLOOPDUAL', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/8.png", transparent=True)
```


​    
![png](output_50_0.png)
​    



```python
data_for_scatter = train_df[['DIS', "SUCCPRIC"]]
sns.barplot(x='DIS', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/8.png", transparent=True)
```


​    
![png](output_51_0.png)
​    



```python
data_for_scatter = train_df[['TCS', "SUCCPRIC"]]
sns.barplot(x='TCS', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/9.png", transparent=True)
```


​    
![png](output_52_0.png)
​    



```python
data_for_scatter = train_df[['AB1', "SUCCPRIC"]]
sns.barplot(x='AB1', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/10.png", transparent=True)
```


​    
![png](output_53_0.png)
​    



```python
data_for_scatter = train_df[['ETC', "SUCCPRIC"]]
sns.catplot(x='ETC', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
```


​    
![png](output_54_0.png)
​    



```python
data_for_scatter = train_df[['AV', "SUCCPRIC"]]
sns.barplot(x='AV', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/11.png", transparent=True)
```


​    
![png](output_55_0.png)
​    



```python
data_for_scatter = train_df[['EPS', "SUCCPRIC"]]
sns.catplot(x='EPS', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
```


​    
![png](output_56_0.png)
​    



```python
data_for_scatter = train_df[['ECS', "SUCCPRIC"]]
sns.barplot(x='ECS', y="SUCCPRIC", data=data_for_scatter)
sns.despine()
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/12.png", transparent=True)
```


​    
![png](output_57_0.png)
​    


# Modeling


```python
y = train_df.loc[:,"SUCCPRIC"]
y
train_df = train_df.drop("SUCCPRIC",1)
train_df
```

    C:\Users\user\AppData\Local\Temp/ipykernel_10328/3894760756.py:3: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      train_df = train_df.drop("SUCCPRIC",1)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EXHA</th>
      <th>TRAVDIST</th>
      <th>INNEEXPOCLASCD_YN</th>
      <th>BONET</th>
      <th>FLOODING</th>
      <th>TOTAL_LOSS</th>
      <th>MJ_MODEL_KEY</th>
      <th>DT_MODEL_KEY</th>
      <th>MJ_GRADE_KEY</th>
      <th>DT_GRADE_KEY</th>
      <th>...</th>
      <th>업무</th>
      <th>자가</th>
      <th>개인</th>
      <th>개인사업</th>
      <th>법인</th>
      <th>법인상품</th>
      <th>상품용</th>
      <th>재외국인</th>
      <th>종교단체</th>
      <th>pre_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>38480</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46</td>
      <td>28</td>
      <td>158</td>
      <td>18054</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11310000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1600</td>
      <td>62240</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>37</td>
      <td>295</td>
      <td>1241</td>
      <td>21541</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19750000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1591</td>
      <td>37926</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>37</td>
      <td>295</td>
      <td>1241</td>
      <td>21538</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19340000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000</td>
      <td>110149</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>166</td>
      <td>5782</td>
      <td>20503</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17680000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000</td>
      <td>81675</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>166</td>
      <td>5782</td>
      <td>20502</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15800000.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36171</th>
      <td>998</td>
      <td>62180</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46</td>
      <td>1053</td>
      <td>10912</td>
      <td>24318</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12500000.0</td>
    </tr>
    <tr>
      <th>36172</th>
      <td>3778</td>
      <td>97801</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>1047</td>
      <td>10874</td>
      <td>24245</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>56800000.0</td>
    </tr>
    <tr>
      <th>36173</th>
      <td>3342</td>
      <td>153601</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>1047</td>
      <td>10873</td>
      <td>24243</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>49089202.0</td>
    </tr>
    <tr>
      <th>36174</th>
      <td>1999</td>
      <td>140058</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>935</td>
      <td>5789</td>
      <td>23037</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17130000.0</td>
    </tr>
    <tr>
      <th>36175</th>
      <td>1999</td>
      <td>159467</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>935</td>
      <td>5789</td>
      <td>23037</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17130000.0</td>
    </tr>
  </tbody>
</table>
<p>36176 rows × 63 columns</p>
</div>




```python
#trainset과 validset으로 나누기

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(train_df, y, test_size=0.2, shuffle=True, random_state=16)
```


```python
#AdaBoost와 RandomForest를 비교하여 보다 나은 성능을 가진 모델을 채택
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV

ada=AdaBoostRegressor(random_state=16)
params={'n_estimators':(10,50,100),'learning_rate':(0.01,0.1,1)}

rand_cv=RandomizedSearchCV(ada,param_distributions=params)
search_ada=rand_cv.fit(x_train,y_train)

ada_best=search_ada.best_estimator_
y_pred_ada=ada_best.predict(x_valid)


MSE_ada=np.mean(np.square(y_pred_ada - y_valid))
MAPE_ada=np.mean(np.abs((y_pred_ada- y_valid)/y_valid))
MAE_ada=np.mean(np.abs(y_pred_ada - y_valid))

print(f'MSE: ', {MSE_ada})
print(f'MAPE:', {MAPE_ada})
print(f'MAE:', {MAE_ada})
```

    C:\Users\user\anaconda3\lib\site-packages\sklearn\model_selection\_search.py:285: UserWarning: The total space of parameters 9 is smaller than n_iter=10. Running 9 iterations. For exhaustive searches, use GridSearchCV.
      warnings.warn(



    ---------------------------------------------------------------------------
    
    KeyboardInterrupt                         Traceback (most recent call last)
    
    ~\AppData\Local\Temp/ipykernel_10328/3142627747.py in <module>
          7 
          8 rand_cv=RandomizedSearchCV(ada,param_distributions=params)
    ----> 9 search_ada=rand_cv.fit(x_train,y_train)
         10 
         11 ada_best=search_ada.best_estimator_


    ~\anaconda3\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
         61             extra_args = len(args) - len(all_args)
         62             if extra_args <= 0:
    ---> 63                 return f(*args, **kwargs)
         64 
         65             # extra_args > 0


    ~\anaconda3\lib\site-packages\sklearn\model_selection\_search.py in fit(self, X, y, groups, **fit_params)
        839                 return results
        840 
    --> 841             self._run_search(evaluate_candidates)
        842 
        843             # multimetric is determined here because in the case of a callable


    ~\anaconda3\lib\site-packages\sklearn\model_selection\_search.py in _run_search(self, evaluate_candidates)
       1631     def _run_search(self, evaluate_candidates):
       1632         """Search n_iter candidates from param_distributions"""
    -> 1633         evaluate_candidates(ParameterSampler(
       1634             self.param_distributions, self.n_iter,
       1635             random_state=self.random_state))


    ~\anaconda3\lib\site-packages\sklearn\model_selection\_search.py in evaluate_candidates(candidate_params, cv, more_results)
        793                               n_splits, n_candidates, n_candidates * n_splits))
        794 
    --> 795                 out = parallel(delayed(_fit_and_score)(clone(base_estimator),
        796                                                        X, y,
        797                                                        train=train, test=test,


    ~\anaconda3\lib\site-packages\joblib\parallel.py in __call__(self, iterable)
       1042                 self._iterating = self._original_iterator is not None
       1043 
    -> 1044             while self.dispatch_one_batch(iterator):
       1045                 pass
       1046 


    ~\anaconda3\lib\site-packages\joblib\parallel.py in dispatch_one_batch(self, iterator)
        857                 return False
        858             else:
    --> 859                 self._dispatch(tasks)
        860                 return True
        861 


    ~\anaconda3\lib\site-packages\joblib\parallel.py in _dispatch(self, batch)
        775         with self._lock:
        776             job_idx = len(self._jobs)
    --> 777             job = self._backend.apply_async(batch, callback=cb)
        778             # A job can complete so quickly than its callback is
        779             # called before we get here, causing self._jobs to


    ~\anaconda3\lib\site-packages\joblib\_parallel_backends.py in apply_async(self, func, callback)
        206     def apply_async(self, func, callback=None):
        207         """Schedule a func to be run"""
    --> 208         result = ImmediateResult(func)
        209         if callback:
        210             callback(result)


    ~\anaconda3\lib\site-packages\joblib\_parallel_backends.py in __init__(self, batch)
        570         # Don't delay the application, to avoid keeping the input
        571         # arguments in memory
    --> 572         self.results = batch()
        573 
        574     def get(self):


    ~\anaconda3\lib\site-packages\joblib\parallel.py in __call__(self)
        260         # change the default number of processes to -1
        261         with parallel_backend(self._backend, n_jobs=self._n_jobs):
    --> 262             return [func(*args, **kwargs)
        263                     for func, args, kwargs in self.items]
        264 


    ~\anaconda3\lib\site-packages\joblib\parallel.py in <listcomp>(.0)
        260         # change the default number of processes to -1
        261         with parallel_backend(self._backend, n_jobs=self._n_jobs):
    --> 262             return [func(*args, **kwargs)
        263                     for func, args, kwargs in self.items]
        264 


    ~\anaconda3\lib\site-packages\sklearn\utils\fixes.py in __call__(self, *args, **kwargs)
        220     def __call__(self, *args, **kwargs):
        221         with config_context(**self.config):
    --> 222             return self.function(*args, **kwargs)


    ~\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, return_estimator, split_progress, candidate_progress, error_score)
        623 
        624         fit_time = time.time() - start_time
    --> 625         test_scores = _score(estimator, X_test, y_test, scorer, error_score)
        626         score_time = time.time() - start_time - fit_time
        627         if return_train_score:


    ~\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py in _score(estimator, X_test, y_test, scorer, error_score)
        685             scores = scorer(estimator, X_test)
        686         else:
    --> 687             scores = scorer(estimator, X_test, y_test)
        688     except Exception:
        689         if error_score == 'raise':


    ~\anaconda3\lib\site-packages\sklearn\metrics\_scorer.py in _passthrough_scorer(estimator, *args, **kwargs)
        395 def _passthrough_scorer(estimator, *args, **kwargs):
        396     """Function that wraps estimator.score"""
    --> 397     return estimator.score(*args, **kwargs)
        398 
        399 


    ~\anaconda3\lib\site-packages\sklearn\base.py in score(self, X, y, sample_weight)
        551 
        552         from .metrics import r2_score
    --> 553         y_pred = self.predict(X)
        554         return r2_score(y, y_pred, sample_weight=sample_weight)
        555 


    ~\anaconda3\lib\site-packages\sklearn\ensemble\_weight_boosting.py in predict(self, X)
       1147         X = self._check_X(X)
       1148 
    -> 1149         return self._get_median_predict(X, len(self.estimators_))
       1150 
       1151     def staged_predict(self, X):


    ~\anaconda3\lib\site-packages\sklearn\ensemble\_weight_boosting.py in _get_median_predict(self, X, limit)
       1111     def _get_median_predict(self, X, limit):
       1112         # Evaluate predictions of all estimators
    -> 1113         predictions = np.array([
       1114             est.predict(X) for est in self.estimators_[:limit]]).T
       1115 


    ~\anaconda3\lib\site-packages\sklearn\ensemble\_weight_boosting.py in <listcomp>(.0)
       1112         # Evaluate predictions of all estimators
       1113         predictions = np.array([
    -> 1114             est.predict(X) for est in self.estimators_[:limit]]).T
       1115 
       1116         # Sort the predictions


    ~\anaconda3\lib\site-packages\sklearn\tree\_classes.py in predict(self, X, check_input)
        440         """
        441         check_is_fitted(self)
    --> 442         X = self._validate_X_predict(X, check_input)
        443         proba = self.tree_.predict(X)
        444         n_samples = X.shape[0]


    ~\anaconda3\lib\site-packages\sklearn\tree\_classes.py in _validate_X_predict(self, X, check_input)
        405         """Validate the training data on predict (probabilities)."""
        406         if check_input:
    --> 407             X = self._validate_data(X, dtype=DTYPE, accept_sparse="csr",
        408                                     reset=False)
        409             if issparse(X) and (X.indices.dtype != np.intc or


    ~\anaconda3\lib\site-packages\sklearn\base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
        419             out = X
        420         elif isinstance(y, str) and y == 'no_validation':
    --> 421             X = check_array(X, **check_params)
        422             out = X
        423         else:


    ~\anaconda3\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
         61             extra_args = len(args) - len(all_args)
         62             if extra_args <= 0:
    ---> 63                 return f(*args, **kwargs)
         64 
         65             # extra_args > 0


    ~\anaconda3\lib\site-packages\sklearn\utils\validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
        718 
        719         if force_all_finite:
    --> 720             _assert_all_finite(array,
        721                                allow_nan=force_all_finite == 'allow-nan')
        722 


    ~\anaconda3\lib\site-packages\sklearn\utils\validation.py in _assert_all_finite(X, allow_nan, msg_dtype)
         94     # safely to reduce dtype induced overflows.
         95     is_float = X.dtype.kind in 'fc'
    ---> 96     if is_float and (np.isfinite(_safe_accumulator_op(np.sum, X))):
         97         pass
         98     elif is_float:


    ~\anaconda3\lib\site-packages\sklearn\utils\extmath.py in _safe_accumulator_op(op, x, *args, **kwargs)
        685     """
        686     if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
    --> 687         result = op(x, *args, **kwargs, dtype=np.float64)
        688     else:
        689         result = op(x, *args, **kwargs)


    <__array_function__ internals> in sum(*args, **kwargs)


    ~\anaconda3\lib\site-packages\numpy\core\fromnumeric.py in sum(a, axis, dtype, out, keepdims, initial, where)
       2245         return res
       2246 
    -> 2247     return _wrapreduction(a, np.add, 'sum', axis, dtype, out, keepdims=keepdims,
       2248                           initial=initial, where=where)
       2249 


    ~\anaconda3\lib\site-packages\numpy\core\fromnumeric.py in _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs)
         85                 return reduction(axis=axis, out=out, **passkwargs)
         86 
    ---> 87     return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
         88 
         89 


    KeyboardInterrupt: 



```python
.best_params_
```


    ---------------------------------------------------------------------------
    
    AttributeError                            Traceback (most recent call last)
    
    ~\AppData\Local\Temp/ipykernel_10328/426436063.py in <module>
    ----> 1 ada_best.best_params_


    AttributeError: 'AdaBoostRegressor' object has no attribute 'best_params_'



```python
#Adaboost보다 RandomForest의 성능이 더 좋게 나왔다. 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [300]
max_depths = [30]

regres = RandomForestRegressor(random_state=16)
param_random = {'max_depth':(10, 20, 30), 'n_estimators' : (200, 300), 'max_features' : ('auto', 'sqrt', 'log2')}

rand_cv = RandomizedSearchCV(regres, param_distributions = param_random)
search = rand_cv.fit(x_train, y_train)
search.best_params_

RF = search.best_estimator_

print(f'Hyperparameters: {search.best_estimator_}')
```

    Hyperparameters: RandomForestRegressor(max_depth=20, n_estimators=300, random_state=16)



```python
y_pred_rf = RF.predict(x_valid)

MSE_rf = np.mean(np.square(y_pred_rf - y_valid))
print(f'MSE: {MSE_rf}')

MAPE_rf = 100*np.mean(np.abs((y_valid - y_pred_rf)/y_valid))
print(f'MAPE: {MAPE_rf}')

MAE_rf=np.mean(np.abs(y_pred_rf - y_valid))
print(f'MAE: {MAE_rf}')
```

    MSE: 774783302450.2388
    MAPE: 8.458934874314172
    MAE: 570997.6679234569



```python
def plot_feature_importance(importance_, features_,model_type):
      dict_ = {'feature importance' : importance_, 'features' : features_}
      df = pd.DataFrame(dict_)
      df.sort_values(by=['feature importance'], ascending=False,inplace=True)
      plt.figure(figsize=(10,10))
      
      sns.barplot(x=df['feature importance'], y=df['features'])
      plt.title(model_type + 'FEATURE IMPORTANCE')
      plt.xlabel('FEATURE IMPORTANCE')
      plt.ylabel('FEATURE NAMES')

sns.set(font = 'Malgun Gothic')      
plot_feature_importance(RF.feature_importances_, x_train.columns, 'RANDOM FOREST ')
plt.savefig("D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/Plots/FeatureImportance.png", transparent=True)

```


​    
![png](output_65_0.png)
​    



```python
df_test = pd.read_csv('D:/정석-한양대/4학년 2학기/응용데이터애널리틱스/중고차 가격예측/P1_testset_sample.csv', encoding ='cp949')  # 한글 Encoding 문제로 encoding = 'cp949'

test_df = df_test
test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOODNO</th>
      <th>SUCCYMD</th>
      <th>CARNM</th>
      <th>CHASNO</th>
      <th>CARREGIYMD</th>
      <th>YEAR</th>
      <th>MISSNM</th>
      <th>FUELNM</th>
      <th>COLOR</th>
      <th>EXHA</th>
      <th>...</th>
      <th>SUNLOOPPANORAMA</th>
      <th>SUNLOOPCOMMON</th>
      <th>SUNLOOPDUAL</th>
      <th>DIS</th>
      <th>TCS</th>
      <th>AB1</th>
      <th>ETC</th>
      <th>AV</th>
      <th>EPS</th>
      <th>ECS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1905C0426</td>
      <td>20190523</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532745</td>
      <td>20140529</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>A</td>
      <td>1999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1905C0424</td>
      <td>20190523</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532693</td>
      <td>20140529</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>A</td>
      <td>1999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1905C0431</td>
      <td>20190523</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532176</td>
      <td>20140529</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>A</td>
      <td>1999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1905C0632</td>
      <td>20190523</td>
      <td>더뉴K3 1.6 가솔린(4도어) 노블레스</td>
      <td>KNAFZ412BGA593382</td>
      <td>20160219</td>
      <td>2016</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>1591</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1905C0268</td>
      <td>20190523</td>
      <td>더뉴K9 V6 3.3 (EXECUTIVE)이그제큐티브</td>
      <td>KNALU411BFS024940</td>
      <td>20141121</td>
      <td>2015</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>B</td>
      <td>3342</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1905C0689</td>
      <td>20190523</td>
      <td>Next_Innovation_K5 렌터카 2.0 LPI MX 럭셔리</td>
      <td>KNAGS416BGA096122</td>
      <td>20160526</td>
      <td>2016</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>C</td>
      <td>1999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1905C0274</td>
      <td>20190523</td>
      <td>더뉴K3 1.6 가솔린(4도어) 디럭스</td>
      <td>KNAFJ412BGA633476</td>
      <td>20160516</td>
      <td>2016</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>C</td>
      <td>1591</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1905C0833</td>
      <td>20190523</td>
      <td>Next_Innovation_K5 렌터카 2.0 LPI MX 노블레스</td>
      <td>KNAGU416BGA059980</td>
      <td>20160421</td>
      <td>2016</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>B</td>
      <td>1999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1905C0497</td>
      <td>20190523</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET080033</td>
      <td>20140227</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1905C0690</td>
      <td>20190523</td>
      <td>Next_Innovation_K5 렌터카 2.0 LPI MX 럭셔리</td>
      <td>KNAGS416BGA085931</td>
      <td>20160526</td>
      <td>2016</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>C</td>
      <td>1999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1905C0492</td>
      <td>20190523</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET081406</td>
      <td>20140326</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1905C0543</td>
      <td>20190523</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET078725</td>
      <td>20140312</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1905C0562</td>
      <td>20190523</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET078937</td>
      <td>20140313</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>1000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1905C0569</td>
      <td>20190523</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET081214</td>
      <td>20140326</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1905C0511</td>
      <td>20190523</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET080067</td>
      <td>20140227</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1905C0493</td>
      <td>20190523</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET080933</td>
      <td>20140228</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1905C0181</td>
      <td>20190523</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET080141</td>
      <td>20140228</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1905C0937</td>
      <td>20190523</td>
      <td>레이 1.0 가솔린 디럭스</td>
      <td>KNACK813EET080091</td>
      <td>20140210</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1905C0307</td>
      <td>20190523</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA623654</td>
      <td>20150515</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>B</td>
      <td>1999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1905C0884</td>
      <td>20190523</td>
      <td>모닝 SLX 뷰티</td>
      <td>KNABA24438T597347</td>
      <td>20080401</td>
      <td>2008</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>C</td>
      <td>1000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 104 columns</p>
</div>




```python

#SUCCYMD(낙찰일자) datetime 형식으로 변경
test_df["SUCCYMD"] = test_df["SUCCYMD"].astype(int)
test_df["SUCCYMD"] = test_df["SUCCYMD"].astype(str)
test_df["SUCCYMD"] = pd.to_datetime(test_df["SUCCYMD"])
test_df.dropna(subset=['CARREGIYMD'], inplace = True)

#CARREGIYMD(차량등록일) datetime 형식으로 변경
test_df["CARREGIYMD"] = test_df["CARREGIYMD"].astype(float)
test_df["CARREGIYMD"] = test_df["CARREGIYMD"].astype(str)
test_df["CARREGIYMD"] = test_df["CARREGIYMD"].apply(lambda x : x[:-2])
test_df["CARREGIYMD"] = pd.to_datetime(test_df["CARREGIYMD"])

#USED_DAY(차 사용시간) 파생변수생성
test_df["USED_DAY"] = test_df["SUCCYMD"] - test_df["CARREGIYMD"]
test_df["USED_DAY"] = test_df["USED_DAY"].astype(str)
test_df["USED_DAY"] = test_df["USED_DAY"].apply(lambda x : x[:-5])
test_df["USED_DAY"] = test_df["USED_DAY"].astype(int)

#이상치(USED_DAY가 음수인 경우) drop
index = test_df[test_df["USED_DAY"] < 0].index
print(len(test_df))
test_df = test_df.drop(index)
print(len(test_df))

```

    20
    20



```python
#교환/판금/용접 변수 처리
test_df['FENDER']=test_df['FRONT_LEFT_FENDER']+test_df['FRONT_RIGHT_FENDER']+test_df['LEFT_REAR_FENDER']+test_df['RIGHT_REAR_FENDER']
test_df['DOOR']=test_df['FRONT_LEFT_DOOR']+test_df['FRONT_RIGHT_DOOR']+test_df['BACK_LEFT_DOOR']+test_df['BACK_RIGHT_DOOR']
test_df['STEP']=test_df['LEFT_STEP']+test_df['RIGHT_STEP']
test_df['FILER']=test_df['LEFT_FILER_A']+test_df['RIGHT_FILER_A']+test_df['LEFT_FILER_B']+test_df['RIGHT_FILER_B']+test_df['LEFT_FILER_C']+test_df['RIGHT_FILER_C']
test_df['PANEL']=test_df['FRONT_PANNEL']+test_df['BACK_PANEL1']+test_df['LEFT_INSIDE_PANEL']+test_df['RIGHT_INSIDE_PANEL']+test_df['DASH_PANEL']+test_df['SHEET_PANEL']+test_df['LEFT_QUARTER']+test_df['RIGHT_QUARTER']+test_df['FLOOR_PANEL']+test_df['LEFT_SIDE_PANEL']+test_df['RIGHT_SIDE_PANEL']+test_df['LEFT_REAR_CORNER_PANEL']+test_df['RIGHT_REAR_CORNER_PANEL']+test_df['BACK_PANEL2']+test_df['LEFT_CORNER_PANEL']+test_df['RIGHT_CORNER_PANEL']+test_df['LEFT_SKIRT_PANEL']+test_df['RIGHT_SKIRT_PANEL']+test_df['LEFT_INSIDE_SHEETING']+test_df['RIGHT_INSIDE_SHEETING']+test_df['LEFT_REAR_INSIDE_PANEL_SHEETING']+test_df['RIGHT_REAR_INSIDE_PANEL_SHEETING']+test_df['DASH_PANEL_SHEETING']+test_df['SHEET_BACK_PANEL_SHEETING']+test_df['FLOOR_PANEL_SHEETING']+test_df['LEFT_SIDE_PANEL_SHEETING']+test_df['RIGHT_SIDE_PANEL_SHEETING']
test_df['WHEEL_HOUSE']=test_df['LEFT_WHEEL_HOUSE']+test_df['RIGHT_WHEEL_HOUSE']+test_df['LEFT_INSIDE_WHEEL_HOUSE']+test_df['RIGHT_INSIDE_WHEEL_HOUSE']+test_df['LEFT_REAR_WHEEL_HOUSE']+test_df['RIGHT_REAR_WHEEL_HOUSE']+test_df['LEFT_WHEEL_HOUSE_SHEETING']+test_df['RIGHT_WHEEL_HOUSE_SHEETING']+test_df['LEFT_REAR_WHEEL_HOUSE_SHEETING']+test_df['RIGHT_REAR_WHEEL_HOUSE_SHEETING']
test_df['FRAME']=test_df['SIDE_MEMBER_FRAME']+test_df['SIDE_MEMBER_FRAME2']+test_df['SIDE_MEMBER_FRAME_SHEETING']
test_df['TRUNK1']=test_df['TRUNK_FLOOR']+test_df['TRUNK_FLOOR_SHEETING']+test_df['TRUNK']
```


```python
drop_col=test_df.loc[:,'FRONT_LEFT_FENDER':'RIGHT_SIDE_PANEL_SHEETING'].columns
test_df=test_df.drop(drop_col,axis=1)
```


```python
#도어, 프런트 펜더 등 외판 부위에 대한 판금·용접·교환은 단순 수리로 분류돼 사고차로 간주되지 않는다. 
Accident = test_df["FILER"] + test_df["PANEL"] + test_df["FRAME"]

test_df["ACCIDENT"] = Accident.apply(lambda x: x != 0).astype(int)
test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOODNO</th>
      <th>SUCCYMD</th>
      <th>CARNM</th>
      <th>CHASNO</th>
      <th>CARREGIYMD</th>
      <th>YEAR</th>
      <th>MISSNM</th>
      <th>FUELNM</th>
      <th>COLOR</th>
      <th>EXHA</th>
      <th>...</th>
      <th>USED_DAY</th>
      <th>FENDER</th>
      <th>DOOR</th>
      <th>STEP</th>
      <th>FILER</th>
      <th>PANEL</th>
      <th>WHEEL_HOUSE</th>
      <th>FRAME</th>
      <th>TRUNK1</th>
      <th>ACCIDENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1905C0426</td>
      <td>2019-05-23</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532745</td>
      <td>2014-05-29</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>A</td>
      <td>1999</td>
      <td>...</td>
      <td>1820</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1905C0424</td>
      <td>2019-05-23</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532693</td>
      <td>2014-05-29</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>A</td>
      <td>1999</td>
      <td>...</td>
      <td>1820</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1905C0431</td>
      <td>2019-05-23</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532176</td>
      <td>2014-05-29</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>A</td>
      <td>1999</td>
      <td>...</td>
      <td>1820</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1905C0632</td>
      <td>2019-05-23</td>
      <td>더뉴K3 1.6 가솔린(4도어) 노블레스</td>
      <td>KNAFZ412BGA593382</td>
      <td>2016-02-19</td>
      <td>2016</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>1591</td>
      <td>...</td>
      <td>1189</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1905C0268</td>
      <td>2019-05-23</td>
      <td>더뉴K9 V6 3.3 (EXECUTIVE)이그제큐티브</td>
      <td>KNALU411BFS024940</td>
      <td>2014-11-21</td>
      <td>2015</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>B</td>
      <td>3342</td>
      <td>...</td>
      <td>1644</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1905C0689</td>
      <td>2019-05-23</td>
      <td>Next_Innovation_K5 렌터카 2.0 LPI MX 럭셔리</td>
      <td>KNAGS416BGA096122</td>
      <td>2016-05-26</td>
      <td>2016</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>C</td>
      <td>1999</td>
      <td>...</td>
      <td>1092</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1905C0274</td>
      <td>2019-05-23</td>
      <td>더뉴K3 1.6 가솔린(4도어) 디럭스</td>
      <td>KNAFJ412BGA633476</td>
      <td>2016-05-16</td>
      <td>2016</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>C</td>
      <td>1591</td>
      <td>...</td>
      <td>1102</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1905C0833</td>
      <td>2019-05-23</td>
      <td>Next_Innovation_K5 렌터카 2.0 LPI MX 노블레스</td>
      <td>KNAGU416BGA059980</td>
      <td>2016-04-21</td>
      <td>2016</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>B</td>
      <td>1999</td>
      <td>...</td>
      <td>1127</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1905C0497</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET080033</td>
      <td>2014-02-27</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>1911</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1905C0690</td>
      <td>2019-05-23</td>
      <td>Next_Innovation_K5 렌터카 2.0 LPI MX 럭셔리</td>
      <td>KNAGS416BGA085931</td>
      <td>2016-05-26</td>
      <td>2016</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>C</td>
      <td>1999</td>
      <td>...</td>
      <td>1092</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1905C0492</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET081406</td>
      <td>2014-03-26</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>1884</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1905C0543</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET078725</td>
      <td>2014-03-12</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>1898</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1905C0562</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET078937</td>
      <td>2014-03-13</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>1000</td>
      <td>...</td>
      <td>1897</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1905C0569</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET081214</td>
      <td>2014-03-26</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>1884</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1905C0511</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET080067</td>
      <td>2014-02-27</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>1911</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1905C0493</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET080933</td>
      <td>2014-02-28</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>1910</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1905C0181</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET080141</td>
      <td>2014-02-28</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>D</td>
      <td>998</td>
      <td>...</td>
      <td>1910</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1905C0937</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 디럭스</td>
      <td>KNACK813EET080091</td>
      <td>2014-02-10</td>
      <td>2014</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>A</td>
      <td>999</td>
      <td>...</td>
      <td>1928</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1905C0307</td>
      <td>2019-05-23</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA623654</td>
      <td>2015-05-15</td>
      <td>2015</td>
      <td>A/T</td>
      <td>LPG</td>
      <td>B</td>
      <td>1999</td>
      <td>...</td>
      <td>1469</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1905C0884</td>
      <td>2019-05-23</td>
      <td>모닝 SLX 뷰티</td>
      <td>KNABA24438T597347</td>
      <td>2008-04-01</td>
      <td>2008</td>
      <td>A/T</td>
      <td>가솔린</td>
      <td>C</td>
      <td>1000</td>
      <td>...</td>
      <td>4069</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 55 columns</p>
</div>




```python
MISS = one_hot_encoder(test_df["MISSNM"])
FUEL = one_hot_encoder(test_df["FUELNM"])
COLOR = one_hot_encoder(test_df["COLOR"])
USEUSE = one_hot_encoder(test_df["USEUSENM"])
OWNE = one_hot_encoder(test_df["OWNECLASNM"])





INNEEX = test_df['INNEEXPOCLASCD_YN'].apply(lambda x : x != "X").astype(int)


test_df = test_df.join(MISS)
test_df = test_df.drop("MISSNM",1)
test_df

test_df = test_df.join(FUEL)
test_df = test_df.drop("FUELNM",1)
test_df

test_df = test_df.join(COLOR)
test_df = test_df.drop("COLOR",1)
test_df

test_df = test_df.join(USEUSE)
test_df = test_df.drop("USEUSENM",1)
test_df

test_df = test_df.join(OWNE)
test_df = test_df.drop("OWNECLASNM",1)
test_df

test_df["INNEEXPOCLASCD_YN"] = INNEEX
test_df

```

    C:\Users\user\AppData\Local\Temp/ipykernel_10328/844670715.py:15: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      test_df = test_df.drop("MISSNM",1)
    C:\Users\user\AppData\Local\Temp/ipykernel_10328/844670715.py:19: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      test_df = test_df.drop("FUELNM",1)
    C:\Users\user\AppData\Local\Temp/ipykernel_10328/844670715.py:23: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      test_df = test_df.drop("COLOR",1)
    C:\Users\user\AppData\Local\Temp/ipykernel_10328/844670715.py:27: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      test_df = test_df.drop("USEUSENM",1)
    C:\Users\user\AppData\Local\Temp/ipykernel_10328/844670715.py:31: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      test_df = test_df.drop("OWNECLASNM",1)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOODNO</th>
      <th>SUCCYMD</th>
      <th>CARNM</th>
      <th>CHASNO</th>
      <th>CARREGIYMD</th>
      <th>YEAR</th>
      <th>EXHA</th>
      <th>TRAVDIST</th>
      <th>INNEEXPOCLASCD_YN</th>
      <th>NEWCARPRIC</th>
      <th>...</th>
      <th>가솔린</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>렌트</th>
      <th>리스</th>
      <th>업무</th>
      <th>자가</th>
      <th>법인상품</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1905C0426</td>
      <td>2019-05-23</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532745</td>
      <td>2014-05-29</td>
      <td>2015</td>
      <td>1999</td>
      <td>155933</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1905C0424</td>
      <td>2019-05-23</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532693</td>
      <td>2014-05-29</td>
      <td>2015</td>
      <td>1999</td>
      <td>152960</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1905C0431</td>
      <td>2019-05-23</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA532176</td>
      <td>2014-05-29</td>
      <td>2015</td>
      <td>1999</td>
      <td>161471</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1905C0632</td>
      <td>2019-05-23</td>
      <td>더뉴K3 1.6 가솔린(4도어) 노블레스</td>
      <td>KNAFZ412BGA593382</td>
      <td>2016-02-19</td>
      <td>2016</td>
      <td>1591</td>
      <td>35382</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1905C0268</td>
      <td>2019-05-23</td>
      <td>더뉴K9 V6 3.3 (EXECUTIVE)이그제큐티브</td>
      <td>KNALU411BFS024940</td>
      <td>2014-11-21</td>
      <td>2015</td>
      <td>3342</td>
      <td>46965</td>
      <td>0</td>
      <td>58670000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1905C0689</td>
      <td>2019-05-23</td>
      <td>Next_Innovation_K5 렌터카 2.0 LPI MX 럭셔리</td>
      <td>KNAGS416BGA096122</td>
      <td>2016-05-26</td>
      <td>2016</td>
      <td>1999</td>
      <td>28046</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1905C0274</td>
      <td>2019-05-23</td>
      <td>더뉴K3 1.6 가솔린(4도어) 디럭스</td>
      <td>KNAFJ412BGA633476</td>
      <td>2016-05-16</td>
      <td>2016</td>
      <td>1591</td>
      <td>85104</td>
      <td>0</td>
      <td>14522741</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1905C0833</td>
      <td>2019-05-23</td>
      <td>Next_Innovation_K5 렌터카 2.0 LPI MX 노블레스</td>
      <td>KNAGU416BGA059980</td>
      <td>2016-04-21</td>
      <td>2016</td>
      <td>1999</td>
      <td>102099</td>
      <td>0</td>
      <td>26130000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1905C0497</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET080033</td>
      <td>2014-02-27</td>
      <td>2014</td>
      <td>998</td>
      <td>37303</td>
      <td>0</td>
      <td>11890000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1905C0690</td>
      <td>2019-05-23</td>
      <td>Next_Innovation_K5 렌터카 2.0 LPI MX 럭셔리</td>
      <td>KNAGS416BGA085931</td>
      <td>2016-05-26</td>
      <td>2016</td>
      <td>1999</td>
      <td>25483</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1905C0492</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET081406</td>
      <td>2014-03-26</td>
      <td>2014</td>
      <td>998</td>
      <td>83262</td>
      <td>0</td>
      <td>11890000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1905C0543</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET078725</td>
      <td>2014-03-12</td>
      <td>2014</td>
      <td>998</td>
      <td>62908</td>
      <td>0</td>
      <td>11890000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1905C0562</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET078937</td>
      <td>2014-03-13</td>
      <td>2014</td>
      <td>1000</td>
      <td>54133</td>
      <td>0</td>
      <td>11890000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1905C0569</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET081214</td>
      <td>2014-03-26</td>
      <td>2014</td>
      <td>998</td>
      <td>49490</td>
      <td>0</td>
      <td>11890000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1905C0511</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET080067</td>
      <td>2014-02-27</td>
      <td>2014</td>
      <td>998</td>
      <td>81308</td>
      <td>0</td>
      <td>11890000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1905C0493</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET080933</td>
      <td>2014-02-28</td>
      <td>2014</td>
      <td>998</td>
      <td>105163</td>
      <td>0</td>
      <td>11890000</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1905C0181</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 VAN</td>
      <td>KNACF911BET080141</td>
      <td>2014-02-28</td>
      <td>2014</td>
      <td>998</td>
      <td>47757</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1905C0937</td>
      <td>2019-05-23</td>
      <td>레이 1.0 가솔린 디럭스</td>
      <td>KNACK813EET080091</td>
      <td>2014-02-10</td>
      <td>2014</td>
      <td>999</td>
      <td>111209</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1905C0307</td>
      <td>2019-05-23</td>
      <td>더뉴K5 2.0 LPI 렌터카 디럭스</td>
      <td>KNAGN418BFA623654</td>
      <td>2015-05-15</td>
      <td>2015</td>
      <td>1999</td>
      <td>86488</td>
      <td>0</td>
      <td>16370000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1905C0884</td>
      <td>2019-05-23</td>
      <td>모닝 SLX 뷰티</td>
      <td>KNABA24438T597347</td>
      <td>2008-04-01</td>
      <td>2008</td>
      <td>1000</td>
      <td>53287</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 62 columns</p>
</div>




```python
not_exist_column = list(set(train_df.columns) - set(test_df.columns))
test_df[not_exist_column] = 0
```


```python
data1 = test_df[test_df["SHIPPING_PRICE"].notnull()]
index1 = data1.index

data2= data1[data1["NC_GRADE_PRICE"].notnull()]
index2 = data2.index

data3 = data2[data2["NC_GRADE_PRICE"].isnull()]
index3 = data3.index

test_df["pre_price"] = 0
test_df.loc[index1, "pre_price"] = test_df.loc[index1, "SHIPPING_PRICE"]
test_df.loc[index2, "pre_price"] = test_df.loc[index2, "NC_GRADE_PRICE"]
test_df.loc[index3, "pre_price"] = test_df.loc[index3, "NEWCARPRIC"]

#이상치 처리
index = test_df[test_df["pre_price"] < 0.2 * (10**7)].index
test_df = test_df.drop(index)

test_df = test_df.drop(["SHIPPING_PRICE", "NC_GRADE_PRICE", "NEWCARPRIC"], axis = 1)
test_df = test_df.reset_index(drop=True)


```


```python
test_df.loc[:,'ABS':'ECS']=test_df.loc[:,'ABS':'ECS'].astype('object')
test_df[['FLOODING','TOTAL_LOSS']]=test_df[['FLOODING','TOTAL_LOSS']].astype('object')
```


```python
test_df = test_df.drop(["SUCCYMD",	"CARREGIYMD",	"YEAR", 'GOODNO','CARNM','CHASNO','YEARCHK','MF_KEY','JOINCAR','NOTAVAILABLE'], axis = 1)

test_df = test_df.dropna()
test_df.reset_index(drop = True, inplace = True)
```


```python
test_y = test_df.iloc[:,3]
test_x = test_df.drop("SUCCPRIC",1)
```

    C:\Users\user\AppData\Local\Temp/ipykernel_10328/1869319339.py:2: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      test_x = test_df.drop("SUCCPRIC",1)



```python
y_pred_rf = RF.predict(test_x)

MSE_rf = np.mean(np.square(y_pred_rf - test_y))
print(f'MSE: {MSE_rf}')

MAPE_rf = 100*np.mean(np.abs((test_y - y_pred_rf)/test_y))
print(f'MAPE: {MAPE_rf}')

MAE_rf=np.mean(np.abs(y_pred_rf - test_y))
print(f'MAE: {MAE_rf}')
```

    MSE: 35104907394450.656
    MAPE: 66.23604130430908
    MAE: 4967117.765197766



```python

```


```python

```
