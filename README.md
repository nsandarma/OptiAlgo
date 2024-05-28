# OptiAlgo
OptiAlgo menyediakan solusi yang cepat dan andal bagi pengguna yang ingin menemukan algoritma terbaik untuk data mereka tanpa harus melakukan pengujian yang rumit dan memakan waktu secara manual.

![image](https://raw.githubusercontent.com/nsandarma/OptiAlgo/master/images/demo.gif)

## Fitur
1. Data Prepration
2. Data Preprocessing
3. Comparing Model
4. Set Model
5. Prediction
6. HyperParameter Tuning

## Instalasi

**Sebelum install OptiAlgo, disarankan membuat environment terlebih dahulu.**

```bash
pip install optialgo
```

## Cara Menggunakan
```py
import pandas as pd
from optialgo import Dataset, Classification

df = pd.read_csv('dataset_ex/drug200.csv')
features = ['Age','Sex','BP','Cholesterol',"Na_to_K"]
target = 'Drug'

dataset = Dataset(dataframe=df)
dataset.fit(features=features,target=target)

clf = Classification()
result = clf.compare_model(output='table',train_val=True)
print(result)
```

![image](images/result.png)

untuk lebih lengkap nya anda bisa temukan pada notebook [example](https://github.com/nsandarma/OptiAlgo/blob/master/examples/classification.ipynb)


## Cara Berkontribusi
Kami sangat menyambut kontribusi dari komunitas untuk meningkatkan dan mengembangkan OptiAlgo. Berikut adalah langkah-langkah umum untuk berkontribusi:

1. **Beri Masukan**: Berikan masukan tentang bagaimana kami dapat meningkatkan OptiAlgo melalui pembuatan *issues*.
2. **Kode Sumber**: Jika Anda seorang pengembang, Anda dapat berkontribusi dengan menulis kode sumber baru atau memperbaiki yang sudah ada.
3. **Uji Coba**: Bantu kami dengan menguji OptiAlgo dan memberikan umpan balik tentang pengalaman Anda.

## Lisensi

MIT

## Kontak

email : nsandarma@gmail.com

Terima kasih telah menggunakan OptiAlgo!