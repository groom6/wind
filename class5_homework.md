
## Class5 Homewrok
### 1. 目录结构
```
.
|-- Program
|   |-- MACD_strategy
|   |   |-- __pycache__
|   |   |   |-- __init__.cpython-36.pyc
|   |   |   |-- indicators.cpython-36.pyc
|   |   |   `-- signals.cpython-36.pyc
|   |   |-- __init__.py
|   |   |-- indicators.py
|   |   `-- main.py
|   |-- __pycache__
|   |   |-- config.cpython-36.pyc
|   |   `-- functions.cpython-36.pyc
|   |-- config.py
|   `-- functions.py
|-- data
|   |-- intput_data
|   |   `-- stock_data
|   |       `-- sz300001.csv
|   `-- output_data
`-- class5_homework.ipynb

8 directories, 12 files
```
### 2. 源代码
#### 2.1 config.py
    程序根目录、数据文件目录全局变量


```python
# coding = utf-8
"""
Author:Groom
Description:Class5 homework -- var config file
Date:2017-12-16
"""
import os
# 获取当前文件的路径
current_file = __file__
# 获取程序根目录
root_path = os.path.abspath(os.path.join(current_file, os.pardir, os.pardir))
# 获取输入数据目录
input_data_path = os.path.abspath(os.path.join(root_path, 'data', 'intput_data'))
# 获取输出数据目录
output_data_path = os.path.abspath(os.path.join(root_path, 'data', 'output_data'))
```

#### 2.2 functions.py
    定义了csv文件读入函数
    定义了后复权函数


```python
# coding = utf-8
"""
Author:Groom
Description:Class5 homework -- Functions(csv_to_df,cal_restoration_rights)
Date:2017-12-16
"""
from Program import config
import pandas as pd


def csv_to_df(stock_code):
    # 读入数据并做相关整理，输出到df中
    df = pd.read_csv(config.input_data_path + '/' + 'stock_data' + '/' + stock_code + '.csv',
                     encoding='gbk')
    # 为什么不需要encode to utf8，难道是Python3.6已经能智能处理了?
    # df.columns = [i.encode('utf8') for i in df.columns]
    df = df[['股票代码', '交易日期', '开盘价', '最高价', '最低价',
             '收盘价', '后复权价','涨跌幅']]
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df.sort_values(by='交易日期', inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def cal_restoration_rights(cleaned_data):
    # 计算后复权价
    # - 计算复权因子,后复权因子为之前涨跌幅结果(+1)的累乘
    df = cleaned_data
    df['复权因子'] = (df['涨跌幅'] + 1).cumprod()
    # 通过上市日收盘价和当天涨跌幅计算开盘价
    initial_price = df.iloc[0]['收盘价'] / (1 + df.iloc[0]['涨跌幅'])
    df['收盘价_后复权'] = initial_price * df['复权因子']
    df['开盘价_后复权'] = df['开盘价'] / df['收盘价'] * df['收盘价_后复权']
    df['最高价_后复权'] = df['最高价'] / df['收盘价'] * df['收盘价_后复权']
    df['最低价_后复权'] = df['最低价'] / df['收盘价'] * df['收盘价_后复权']
    # df = df[['股票代码', '交易日期', '后复权价', '收盘价_复权', '复权因子', '涨跌幅']]
    df = df[['开盘价_后复权', '最高价_后复权', '最低价_后复权', '收盘价_后复权']]
    return df

```

#### 2.3 indicators.py
    计算MACD、KDJ函数


```python
# coding = utf-8
"""
Author:Groom
Description:Class5 homework -- Functions(cal_macd,cal_kdj)
Date:2017-12-17
"""
import pandas as pd


def cal_macd(df, ema_param1=12, ema_param2=26, dea_param=9):
    """
    计算MACD：
    :param df:    复权后的数据
    :param ema_param1: 默认EMA1 参数为12天
    :param ema_param2: 默认EMA2 参数为26天
    :param dea_param:  默认EMA2 参数为9天
    :return:
    """
    ema1 = 'EMA' + str(ema_param1)
    ema2 = 'EMA' + str(ema_param2)
    dea = 'DEA' + str(dea_param)
    """
    上市首日、DIFF,DEA,DEA都为0，第二日的EMA采用首日的收盘价来计算
    """
    # 设置上市首日的EMA、DIFF、DEA
    df.loc[0, ema1] = df.loc[0, '收盘价_后复权']
    df.loc[0, ema2] = df.loc[0, '收盘价_后复权']
    df.loc[0, 'DIFF'] = 0
    df.loc[0, 'DEA'] = 0
    df.loc[0, 'MACD'] = 0
    # 通过逐行循环处理EMA,DIFF,DEA,MACD值
    for row in range(1, df.shape[0]):
        df.at[row, ema1] = df.at[row - 1, ema1] * (ema_param1 - 1) / (ema_param1 + 1) \
                           + df.at[row, '收盘价_后复权'] * 2 / (ema_param1 + 1)
        df.at[row, ema2] = df.at[row - 1, ema2] * (ema_param2 - 1) / (ema_param2 + 1) \
                           + df.at[row, '收盘价_后复权'] * 2 / (ema_param2 + 1)
        df.at[row, 'DIF'] = df.at[row, ema1] - df.at[row, ema2]
        df.at[row, 'DEA'] = df.at[row - 1, 'DEA'] * (dea_param - 1) / (dea_param + 1) \
                            + df.at[row, 'DIF'] * 2 / (dea_param + 1)
        df.at[row, 'MACD'] = (df.at[row, 'DIF'] - df.at[row, 'DEA']) * 2

    # ***对Dataframe 还不熟悉，不知道能否列内递归操作，类似如下意思？
    # df[ema1] = df[ema1].shift() * (ema_param1 - 1) / (ema_param1 + 1) + df['收盘价_后复权'] * 2 / (ema_param1 + 1)
    # df[ema2] = df[ema2].shift() * (ema_param2 - 1) / (ema_param2 + 1) + df['收盘价_后复权'] * 2 / (ema_param2 + 1)
    # df['DIFF'] = df[ema1] - df[ema2]
    # df['DEA'] = df['DEA'].shift() * (dea_param - 1) / (dea_param + 1) + df['DIFF'] * 2 / (dea_param + 1)
    #
    return df


def cal_kdj(df, n):
    """
    计算KDJ：
    :param df:    复权后的数据
    :param n: 计算第n日的kdj
    :return: n日的K,D,J
    """
    # 增加两列，分别放入n日内的最低价和最高价
    df['kdj_最低价'] = df['最低价_后复权'].rolling(n).min()
    df['kdj_最低价'].fillna(method='bfill', inplace=True)
    df['kdj_最高价'] = df['最高价_后复权'].rolling(n).max()
    df['kdj_最高价'].fillna(method='bfill', inplace=True)
    # ***不清楚K,D初始值设置为什么，暂定设置为RSV
    rsv = (df.at[0, '收盘价_后复权'] - df.at[0, 'kdj_最低价']) / (df.at[0, 'kdj_最高价'] - df.at[0, 'kdj_最低价']) * 100
    df.at[0, 'K'] = rsv
    df.at[0, 'D'] = rsv
    # df.at[0, 'K'] = 0
    # df.at[0, 'D'] = 0
    # 通过循环逐行计算K,D,J值
    for i in range(1, df.shape[0]):
        rsv = (df.at[i, '收盘价_后复权'] - df.at[i, 'kdj_最低价']) / (df.at[i, 'kdj_最高价'] - df.at[i, 'kdj_最低价']) * 100
        df.at[i, 'K'] = 2 / 3 * df.at[i - 1, 'K'] + 1 / 3 * rsv
        df.at[i, 'D'] = 2 / 3 * df.at[i - 1, 'D'] + 1 / 3 * df.at[i, 'K']
        df.at[i, 'J'] = 3 * df.at[i, 'K'] - 2 * df.at[i, 'D']
    return df


def cal_kdj_1(df, n):
    """
    网上找的，待验证！
    """
    df['kdj_最低价'] = df['最低价_后复权'].rolling(n).min()
    df['kdj_最低价'].fillna(method='bfill', inplace=True)
    df['kdj_最高价'] = df['最高价_后复权'].rolling(n).max()
    df['kdj_最高价'].fillna(method='bfill', inplace=True)

    rsv = (df['收盘价_后复权'] - df['kdj_最低价']) / (df['kdj_最高价'] - df['kdj_最低价']) * 100
    df['kdj_K'] = pd.ewma(rsv, com=2)
    df['kdj_D'] = pd.ewma(df['kdj_K'], com=2)
    df['kdj_J'] = 3.0 * df['kdj_K'] - 2.0 * df['kdj_D']
    return df

```

#### 2.4 main.py
    主程序


```python
# code = utf-8
"""
Author:Groom
Description:Class5 homework -- Main Program(Output MACD and KDJ)
Date:2017-12-18
"""
from Program import functions
from Program.MACD_strategy import indicators
import pandas as pd

pd.set_option('expand_frame_repr', False)

if __name__ == '__main__':
    # 读入数据
    code = 'sz300001'
    df = functions.csv_to_df(stock_code=code)
    # 执行后复权函数，输出df
    df[['开盘价_后复权', '最高价_后复权', '最低价_后复权', '收盘价_后复权']] = functions.cal_restoration_rights(df)
    # 执行MACD计算函数
    df = indicators.cal_macd(df)
    # 执行KDJ计算函数
    df = indicators.cal_kdj(df, 10)
    #print(df[['交易日期', '股票代码', '后复权价', '开盘价_后复权', '最高价_后复权', '最低价_后复权', 'DIF', 'DEA', 'MACD', 'K', 'D', 'J']])

```



#### 2.5 程序运行输出结果


```python
df[['交易日期', '股票代码', '后复权价', '开盘价_后复权', '最高价_后复权', '最低价_后复权', 'DIF', 'DEA', 'MACD', 'K', 'D', 'J']]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
        font-size: 9px;
    }

    .dataframe thead th {
        text-align: left;
        font-size: 9px;
    }

    .dataframe tbody tr th {
        vertical-align: top;
        font-size: 9px;
    }
    .dataframe tbody td {
        vertical-align: top;
        font-size: 9px;
    }
</style>
<table border="1" class="dataframe" >
  <thead>
    <tr style="text-align: right">
      <th></th>
      <th>交易日期</th>
      <th>股票代码</th>
      <th>后复权价</th>
      <th>开盘价_后复权</th>
      <th>最高价_后复权</th>
      <th>最低价_后复权</th>
      <th>DIF</th>
      <th>DEA</th>
      <th>MACD</th>
      <th>K</th>
      <th>D</th>
      <th>J</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009-10-30</td>
      <td>sz300001</td>
      <td>44.000000</td>
      <td>42.000000</td>
      <td>64.000000</td>
      <td>35.010000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>31.010693</td>
      <td>31.010693</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009-11-02</td>
      <td>sz300001</td>
      <td>39.600000</td>
      <td>39.600000</td>
      <td>41.870000</td>
      <td>39.600000</td>
      <td>-0.350997</td>
      <td>-0.070199</td>
      <td>-0.561595</td>
      <td>25.951478</td>
      <td>29.324288</td>
      <td>19.205856</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009-11-03</td>
      <td>sz300001</td>
      <td>36.899993</td>
      <td>38.199993</td>
      <td>39.359992</td>
      <td>36.149993</td>
      <td>-0.837380</td>
      <td>-0.223636</td>
      <td>-1.227489</td>
      <td>19.474140</td>
      <td>26.040905</td>
      <td>6.340609</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2009-11-04</td>
      <td>sz300001</td>
      <td>38.019981</td>
      <td>37.999981</td>
      <td>38.899981</td>
      <td>37.229982</td>
      <td>-1.119563</td>
      <td>-0.402821</td>
      <td>-1.433484</td>
      <td>16.443702</td>
      <td>22.841838</td>
      <td>3.647431</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009-11-05</td>
      <td>sz300001</td>
      <td>39.599978</td>
      <td>38.029979</td>
      <td>40.979977</td>
      <td>37.509979</td>
      <td>-1.201848</td>
      <td>-0.562626</td>
      <td>-1.278443</td>
      <td>16.240124</td>
      <td>20.641267</td>
      <td>7.437840</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2009-11-06</td>
      <td>sz300001</td>
      <td>38.909988</td>
      <td>39.629988</td>
      <td>40.399987</td>
      <td>38.249988</td>
      <td>-1.307662</td>
      <td>-0.711634</td>
      <td>-1.192057</td>
      <td>15.311040</td>
      <td>18.864525</td>
      <td>8.204072</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2009-11-09</td>
      <td>sz300001</td>
      <td>39.479980</td>
      <td>38.649981</td>
      <td>39.899980</td>
      <td>38.179981</td>
      <td>-1.330193</td>
      <td>-0.835345</td>
      <td>-0.989695</td>
      <td>15.347041</td>
      <td>17.692030</td>
      <td>10.657063</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2009-11-10</td>
      <td>sz300001</td>
      <td>37.969989</td>
      <td>39.559989</td>
      <td>39.799989</td>
      <td>37.799989</td>
      <td>-1.453142</td>
      <td>-0.958905</td>
      <td>-0.988475</td>
      <td>13.634821</td>
      <td>16.339627</td>
      <td>8.225209</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2009-11-11</td>
      <td>sz300001</td>
      <td>38.219984</td>
      <td>37.649984</td>
      <td>38.399984</td>
      <td>36.999984</td>
      <td>-1.512967</td>
      <td>-1.069717</td>
      <td>-0.886499</td>
      <td>12.780790</td>
      <td>15.153348</td>
      <td>8.035674</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2009-11-12</td>
      <td>sz300001</td>
      <td>38.559989</td>
      <td>38.199989</td>
      <td>38.889989</td>
      <td>37.499989</td>
      <td>-1.515474</td>
      <td>-1.158868</td>
      <td>-0.713210</td>
      <td>12.602381</td>
      <td>14.303025</td>
      <td>9.201092</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2009-11-13</td>
      <td>sz300001</td>
      <td>40.759990</td>
      <td>38.569991</td>
      <td>41.869990</td>
      <td>38.569991</td>
      <td>-1.324668</td>
      <td>-1.192028</td>
      <td>-0.265280</td>
      <td>35.266341</td>
      <td>21.290797</td>
      <td>63.217428</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2009-11-16</td>
      <td>sz300001</td>
      <td>40.999985</td>
      <td>40.799985</td>
      <td>41.299985</td>
      <td>39.619986</td>
      <td>-1.140936</td>
      <td>-1.181810</td>
      <td>0.081747</td>
      <td>51.774266</td>
      <td>31.451953</td>
      <td>92.418891</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2009-11-17</td>
      <td>sz300001</td>
      <td>40.899986</td>
      <td>40.899986</td>
      <td>41.819986</td>
      <td>40.549986</td>
      <td>-0.991962</td>
      <td>-1.143840</td>
      <td>0.303757</td>
      <td>61.210204</td>
      <td>41.371370</td>
      <td>100.887871</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2009-11-18</td>
      <td>sz300001</td>
      <td>42.839995</td>
      <td>40.999995</td>
      <td>43.159995</td>
      <td>40.999995</td>
      <td>-0.709181</td>
      <td>-1.056908</td>
      <td>0.695455</td>
      <td>72.408537</td>
      <td>51.717093</td>
      <td>113.791427</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2009-11-19</td>
      <td>sz300001</td>
      <td>44.909981</td>
      <td>42.899982</td>
      <td>47.099980</td>
      <td>42.719982</td>
      <td>-0.314420</td>
      <td>-0.908411</td>
      <td>1.187982</td>
      <td>74.377969</td>
      <td>59.270718</td>
      <td>104.592470</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2009-11-20</td>
      <td>sz300001</td>
      <td>44.379998</td>
      <td>44.329998</td>
      <td>46.549998</td>
      <td>44.099998</td>
      <td>-0.043829</td>
      <td>-0.735494</td>
      <td>1.383331</td>
      <td>73.941805</td>
      <td>64.161080</td>
      <td>93.503253</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2009-11-23</td>
      <td>sz300001</td>
      <td>45.410014</td>
      <td>44.330013</td>
      <td>45.980014</td>
      <td>44.120013</td>
      <td>0.250839</td>
      <td>-0.538228</td>
      <td>1.578133</td>
      <td>77.050421</td>
      <td>68.457527</td>
      <td>94.236208</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2009-11-24</td>
      <td>sz300001</td>
      <td>41.430008</td>
      <td>45.800008</td>
      <td>46.230008</td>
      <td>41.010008</td>
      <td>0.161351</td>
      <td>-0.398312</td>
      <td>1.119327</td>
      <td>65.987492</td>
      <td>67.634182</td>
      <td>62.694113</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2009-11-25</td>
      <td>sz300001</td>
      <td>43.900023</td>
      <td>41.130022</td>
      <td>44.190023</td>
      <td>41.100022</td>
      <td>0.286440</td>
      <td>-0.261362</td>
      <td>1.095603</td>
      <td>66.214023</td>
      <td>67.160796</td>
      <td>64.320478</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2009-11-26</td>
      <td>sz300001</td>
      <td>47.280018</td>
      <td>43.700016</td>
      <td>47.490018</td>
      <td>43.000016</td>
      <td>0.650809</td>
      <td>-0.078927</td>
      <td>1.459472</td>
      <td>76.691264</td>
      <td>70.337619</td>
      <td>89.398555</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2009-11-27</td>
      <td>sz300001</td>
      <td>47.300017</td>
      <td>46.100017</td>
      <td>48.600018</td>
      <td>45.540017</td>
      <td>0.930462</td>
      <td>0.122950</td>
      <td>1.615022</td>
      <td>79.635320</td>
      <td>73.436852</td>
      <td>92.032255</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2009-11-30</td>
      <td>sz300001</td>
      <td>46.920009</td>
      <td>47.320009</td>
      <td>47.900009</td>
      <td>45.000008</td>
      <td>1.108645</td>
      <td>0.320089</td>
      <td>1.577112</td>
      <td>79.467016</td>
      <td>75.446907</td>
      <td>87.507233</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2009-12-01</td>
      <td>sz300001</td>
      <td>46.980020</td>
      <td>46.480019</td>
      <td>47.380020</td>
      <td>45.700019</td>
      <td>1.240401</td>
      <td>0.504152</td>
      <td>1.472498</td>
      <td>79.206110</td>
      <td>76.699974</td>
      <td>84.218380</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2009-12-02</td>
      <td>sz300001</td>
      <td>48.170023</td>
      <td>47.100023</td>
      <td>49.850024</td>
      <td>46.680023</td>
      <td>1.424422</td>
      <td>0.688206</td>
      <td>1.472432</td>
      <td>79.802574</td>
      <td>77.734174</td>
      <td>83.939373</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2009-12-03</td>
      <td>sz300001</td>
      <td>50.160023</td>
      <td>48.010022</td>
      <td>50.240023</td>
      <td>47.190022</td>
      <td>1.711111</td>
      <td>0.892787</td>
      <td>1.636649</td>
      <td>86.246136</td>
      <td>80.571495</td>
      <td>97.595420</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2009-12-04</td>
      <td>sz300001</td>
      <td>45.559998</td>
      <td>47.999998</td>
      <td>49.149998</td>
      <td>45.139998</td>
      <td>1.549272</td>
      <td>1.024084</td>
      <td>1.050377</td>
      <td>73.929287</td>
      <td>78.357426</td>
      <td>65.073010</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2009-12-07</td>
      <td>sz300001</td>
      <td>46.319984</td>
      <td>45.299985</td>
      <td>46.519984</td>
      <td>44.599985</td>
      <td>1.465445</td>
      <td>1.112356</td>
      <td>0.706178</td>
      <td>68.462673</td>
      <td>75.059175</td>
      <td>55.269670</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2009-12-08</td>
      <td>sz300001</td>
      <td>47.859985</td>
      <td>46.299985</td>
      <td>48.429985</td>
      <td>46.299985</td>
      <td>1.505918</td>
      <td>1.191068</td>
      <td>0.629699</td>
      <td>70.295181</td>
      <td>73.471177</td>
      <td>63.943189</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2009-12-09</td>
      <td>sz300001</td>
      <td>48.009978</td>
      <td>47.399978</td>
      <td>49.799977</td>
      <td>47.039979</td>
      <td>1.532431</td>
      <td>1.259341</td>
      <td>0.546180</td>
      <td>69.929554</td>
      <td>72.290636</td>
      <td>65.207391</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2009-12-10</td>
      <td>sz300001</td>
      <td>47.399963</td>
      <td>47.999963</td>
      <td>48.989962</td>
      <td>46.999964</td>
      <td>1.487077</td>
      <td>1.304888</td>
      <td>0.364378</td>
      <td>63.167926</td>
      <td>69.249733</td>
      <td>51.004312</td>
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
    </tr>
    <tr>
      <th>1610</th>
      <td>2016-11-03</td>
      <td>sz300001</td>
      <td>138.654945</td>
      <td>138.999298</td>
      <td>140.237293</td>
      <td>138.173968</td>
      <td>-0.482218</td>
      <td>-0.863776</td>
      <td>0.763118</td>
      <td>66.337938</td>
      <td>52.170734</td>
      <td>94.672347</td>
    </tr>
    <tr>
      <th>1611</th>
      <td>2016-11-04</td>
      <td>sz300001</td>
      <td>138.517390</td>
      <td>138.449087</td>
      <td>139.618304</td>
      <td>137.142315</td>
      <td>-0.421207</td>
      <td>-0.775262</td>
      <td>0.708112</td>
      <td>65.481422</td>
      <td>56.607630</td>
      <td>83.229005</td>
    </tr>
    <tr>
      <th>1612</th>
      <td>2016-11-07</td>
      <td>sz300001</td>
      <td>138.654945</td>
      <td>138.311525</td>
      <td>139.205633</td>
      <td>137.279863</td>
      <td>-0.357633</td>
      <td>-0.691737</td>
      <td>0.668207</td>
      <td>65.876545</td>
      <td>59.697268</td>
      <td>78.235098</td>
    </tr>
    <tr>
      <th>1613</th>
      <td>2016-11-08</td>
      <td>sz300001</td>
      <td>138.517390</td>
      <td>138.655422</td>
      <td>139.205642</td>
      <td>137.623759</td>
      <td>-0.314722</td>
      <td>-0.616334</td>
      <td>0.603224</td>
      <td>65.173840</td>
      <td>61.522792</td>
      <td>72.475937</td>
    </tr>
    <tr>
      <th>1614</th>
      <td>2016-11-09</td>
      <td>sz300001</td>
      <td>137.348177</td>
      <td>138.036412</td>
      <td>138.861742</td>
      <td>135.629200</td>
      <td>-0.370787</td>
      <td>-0.567224</td>
      <td>0.392874</td>
      <td>56.492745</td>
      <td>59.846110</td>
      <td>49.786017</td>
    </tr>
    <tr>
      <th>1615</th>
      <td>2016-11-10</td>
      <td>sz300001</td>
      <td>139.686604</td>
      <td>138.174030</td>
      <td>139.962245</td>
      <td>138.105252</td>
      <td>-0.223941</td>
      <td>-0.498568</td>
      <td>0.549254</td>
      <td>67.015515</td>
      <td>62.235912</td>
      <td>76.574722</td>
    </tr>
    <tr>
      <th>1616</th>
      <td>2016-11-11</td>
      <td>sz300001</td>
      <td>139.892935</td>
      <td>139.687121</td>
      <td>139.962231</td>
      <td>138.449126</td>
      <td>-0.089879</td>
      <td>-0.416830</td>
      <td>0.653901</td>
      <td>75.523127</td>
      <td>66.664983</td>
      <td>93.239413</td>
    </tr>
    <tr>
      <th>1617</th>
      <td>2016-11-14</td>
      <td>sz300001</td>
      <td>139.961713</td>
      <td>139.893503</td>
      <td>141.337831</td>
      <td>139.343283</td>
      <td>0.021669</td>
      <td>-0.329130</td>
      <td>0.701598</td>
      <td>75.650091</td>
      <td>69.660019</td>
      <td>87.630234</td>
    </tr>
    <tr>
      <th>1618</th>
      <td>2016-11-15</td>
      <td>sz300001</td>
      <td>139.617826</td>
      <td>139.412061</td>
      <td>139.893504</td>
      <td>137.830178</td>
      <td>0.081385</td>
      <td>-0.247027</td>
      <td>0.656824</td>
      <td>73.726737</td>
      <td>71.015592</td>
      <td>79.149028</td>
    </tr>
    <tr>
      <th>1619</th>
      <td>2016-11-16</td>
      <td>sz300001</td>
      <td>138.861277</td>
      <td>140.306129</td>
      <td>141.269014</td>
      <td>138.861801</td>
      <td>0.066889</td>
      <td>-0.184244</td>
      <td>0.502265</td>
      <td>68.026676</td>
      <td>70.019287</td>
      <td>64.041455</td>
    </tr>
    <tr>
      <th>1620</th>
      <td>2016-11-17</td>
      <td>sz300001</td>
      <td>138.448613</td>
      <td>138.173994</td>
      <td>140.030987</td>
      <td>137.898884</td>
      <td>0.021847</td>
      <td>-0.143026</td>
      <td>0.329746</td>
      <td>61.816850</td>
      <td>67.285141</td>
      <td>50.880268</td>
    </tr>
    <tr>
      <th>1621</th>
      <td>2016-11-18</td>
      <td>sz300001</td>
      <td>138.173504</td>
      <td>139.618333</td>
      <td>139.618333</td>
      <td>138.036451</td>
      <td>-0.035636</td>
      <td>-0.121548</td>
      <td>0.171824</td>
      <td>56.070636</td>
      <td>63.546973</td>
      <td>41.117962</td>
    </tr>
    <tr>
      <th>1622</th>
      <td>2016-11-21</td>
      <td>sz300001</td>
      <td>138.242281</td>
      <td>138.242816</td>
      <td>138.793037</td>
      <td>137.623819</td>
      <td>-0.074777</td>
      <td>-0.112194</td>
      <td>0.074833</td>
      <td>52.641620</td>
      <td>59.911855</td>
      <td>38.101149</td>
    </tr>
    <tr>
      <th>1623</th>
      <td>2016-11-22</td>
      <td>sz300001</td>
      <td>139.342717</td>
      <td>138.311567</td>
      <td>139.412007</td>
      <td>137.761347</td>
      <td>-0.016809</td>
      <td>-0.093117</td>
      <td>0.152615</td>
      <td>56.781042</td>
      <td>58.868251</td>
      <td>52.606624</td>
    </tr>
    <tr>
      <th>1624</th>
      <td>2016-11-23</td>
      <td>sz300001</td>
      <td>139.755381</td>
      <td>139.480854</td>
      <td>141.269070</td>
      <td>139.480854</td>
      <td>0.061724</td>
      <td>-0.062149</td>
      <td>0.247745</td>
      <td>56.990070</td>
      <td>58.242190</td>
      <td>54.485830</td>
    </tr>
    <tr>
      <th>1625</th>
      <td>2016-11-24</td>
      <td>sz300001</td>
      <td>138.311059</td>
      <td>139.687136</td>
      <td>139.687136</td>
      <td>138.036476</td>
      <td>0.007328</td>
      <td>-0.048253</td>
      <td>0.111162</td>
      <td>44.166104</td>
      <td>53.550162</td>
      <td>25.397989</td>
    </tr>
    <tr>
      <th>1626</th>
      <td>2016-11-25</td>
      <td>sz300001</td>
      <td>140.099267</td>
      <td>138.105268</td>
      <td>140.237372</td>
      <td>137.692603</td>
      <td>0.107277</td>
      <td>-0.017147</td>
      <td>0.248849</td>
      <td>51.666195</td>
      <td>52.922173</td>
      <td>49.154240</td>
    </tr>
    <tr>
      <th>1627</th>
      <td>2016-11-28</td>
      <td>sz300001</td>
      <td>139.205163</td>
      <td>140.650027</td>
      <td>140.993915</td>
      <td>138.999367</td>
      <td>0.113037</td>
      <td>0.008890</td>
      <td>0.208295</td>
      <td>48.909350</td>
      <td>51.584565</td>
      <td>43.558919</td>
    </tr>
    <tr>
      <th>1628</th>
      <td>2016-11-29</td>
      <td>sz300001</td>
      <td>138.998831</td>
      <td>139.205729</td>
      <td>140.650057</td>
      <td>138.586731</td>
      <td>0.099805</td>
      <td>0.027073</td>
      <td>0.145464</td>
      <td>45.184954</td>
      <td>49.451362</td>
      <td>36.652140</td>
    </tr>
    <tr>
      <th>1629</th>
      <td>2016-11-30</td>
      <td>sz300001</td>
      <td>139.892935</td>
      <td>138.655446</td>
      <td>140.787548</td>
      <td>138.311558</td>
      <td>0.159620</td>
      <td>0.053582</td>
      <td>0.212075</td>
      <td>50.877447</td>
      <td>49.926723</td>
      <td>52.778895</td>
    </tr>
    <tr>
      <th>1630</th>
      <td>2016-12-01</td>
      <td>sz300001</td>
      <td>142.093808</td>
      <td>140.168613</td>
      <td>142.782160</td>
      <td>139.687171</td>
      <td>0.380238</td>
      <td>0.118913</td>
      <td>0.522650</td>
      <td>62.807209</td>
      <td>54.220219</td>
      <td>79.981190</td>
    </tr>
    <tr>
      <th>1631</th>
      <td>2016-12-02</td>
      <td>sz300001</td>
      <td>139.067608</td>
      <td>141.337859</td>
      <td>141.819302</td>
      <td>138.999423</td>
      <td>0.307349</td>
      <td>0.156600</td>
      <td>0.301497</td>
      <td>51.205103</td>
      <td>53.215180</td>
      <td>47.184949</td>
    </tr>
    <tr>
      <th>1632</th>
      <td>2016-12-05</td>
      <td>sz300001</td>
      <td>138.998831</td>
      <td>137.898921</td>
      <td>140.856354</td>
      <td>137.623811</td>
      <td>0.241248</td>
      <td>0.173530</td>
      <td>0.135437</td>
      <td>43.025564</td>
      <td>49.818642</td>
      <td>29.439410</td>
    </tr>
    <tr>
      <th>1633</th>
      <td>2016-12-06</td>
      <td>sz300001</td>
      <td>137.692063</td>
      <td>139.274512</td>
      <td>140.237397</td>
      <td>137.692628</td>
      <td>0.082470</td>
      <td>0.155318</td>
      <td>-0.145696</td>
      <td>29.128408</td>
      <td>42.921897</td>
      <td>1.541431</td>
    </tr>
    <tr>
      <th>1634</th>
      <td>2016-12-07</td>
      <td>sz300001</td>
      <td>137.967172</td>
      <td>137.692628</td>
      <td>138.380403</td>
      <td>136.248300</td>
      <td>-0.020923</td>
      <td>0.120070</td>
      <td>-0.281985</td>
      <td>28.190877</td>
      <td>38.011557</td>
      <td>8.549517</td>
    </tr>
    <tr>
      <th>1635</th>
      <td>2016-12-08</td>
      <td>sz300001</td>
      <td>135.559968</td>
      <td>137.967690</td>
      <td>138.242800</td>
      <td>135.491700</td>
      <td>-0.293723</td>
      <td>0.037311</td>
      <td>-0.662068</td>
      <td>19.108382</td>
      <td>31.710499</td>
      <td>-6.095852</td>
    </tr>
    <tr>
      <th>1636</th>
      <td>2016-12-09</td>
      <td>sz300001</td>
      <td>134.046869</td>
      <td>135.973121</td>
      <td>136.523341</td>
      <td>133.084466</td>
      <td>-0.624812</td>
      <td>-0.095113</td>
      <td>-1.059398</td>
      <td>16.048591</td>
      <td>26.489863</td>
      <td>-4.833952</td>
    </tr>
    <tr>
      <th>1637</th>
      <td>2016-12-12</td>
      <td>sz300001</td>
      <td>123.111285</td>
      <td>134.597615</td>
      <td>135.216612</td>
      <td>122.011328</td>
      <td>-1.749447</td>
      <td>-0.425980</td>
      <td>-2.646934</td>
      <td>12.465064</td>
      <td>21.814930</td>
      <td>-6.234669</td>
    </tr>
    <tr>
      <th>1638</th>
      <td>2016-12-13</td>
      <td>sz300001</td>
      <td>123.180063</td>
      <td>123.730808</td>
      <td>125.037581</td>
      <td>122.286480</td>
      <td>-2.605144</td>
      <td>-0.861813</td>
      <td>-3.486663</td>
      <td>10.186488</td>
      <td>17.938782</td>
      <td>-5.318102</td>
    </tr>
    <tr>
      <th>1639</th>
      <td>2016-12-14</td>
      <td>sz300001</td>
      <td>123.455172</td>
      <td>123.249317</td>
      <td>124.831200</td>
      <td>122.286432</td>
      <td>-3.223932</td>
      <td>-1.334237</td>
      <td>-3.779390</td>
      <td>9.108861</td>
      <td>14.995475</td>
      <td>-2.664368</td>
    </tr>
  </tbody>
</table>
<p>1640 rows × 12 columns</p>
</div>



### 3. 学习心得
#### 3.1 python方面
    1.）对pandas的DataFrame数据处理还是不熟悉，遇到不少问题
        切片或取值中 iloc,loc,at,ix 不清楚，使用时基本是尝试方式；
    2.）对DataFrame的数据循环操作
        单列递归操作是否可以一句话还是要用循环？
    3.）还是要多练习已经系统的学习书本知识
#### 3.2 MACD,KDJ
    1.) 了解计算方法
    2.） 对原理及运用不清楚
