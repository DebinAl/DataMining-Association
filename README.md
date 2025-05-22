# Apriori and FP-Growth Performance Analysis: Market Basket Analysis (MBA) Case Study

---

# Latar Belakang Proyek
Proyek yang dilakukan bertujuan untuk membandingkan performa dua algoritma populer dalam **Market Basket Analysis (MBA)**, yaitu **Apriori** dan **FP-Growth**. Dalam konteks data mining, kedua algoritma ini digunakan untuk menemukan asosiasi atau pola pembelian antar produk dalam data transaksi.

Melalui proyek ini, kami akan menganalisis dan mengevaluasi efisiensi serta efektivitas kedua algoritma dalam menghasilkan *association rules* berdasarkan metrik seperti `waktu komputasi`, `jumlah aturan (candidates amount)` yang dihasilkan, dan `kualitas aturan (candidates quality)` tersebut. Studi kasus yang digunakan bersumber dari data transaksi yang merepresentasikan skenario retail.

# Data Acquisition
Data yang digunakan merupakan data yang tersedia pada [kaggle](https://www.kaggle.com/datasets/gabrielramos87/an-online-shop-business). Deskripsi lengkap dapat dilihat pada *table* dibawah.

| - | Content |
| :--- | :--- |
| **Author** | [Gabriel Ramos](https://www.kaggle.com/gabrielramos87) |
| **Judul** | **E-commerce Business Transaction** |
| **Rows - Column** | 500k Rows - 8 Columns |
| **Dataset** | [Link](https://www.kaggle.com/datasets/gabrielramos87/an-online-shop-business) |

Selain itu untuk kebutuhan *development* dibuatkan `mock dataset generator` yang dibuat menggunakan python Language. *Script* tersebut dapat diakses pada [github](https://github.com/DebinAl/DataMining-Association/blob/main/src/generate_dataset.py).

# Implementation
Dalam proyek ini, dua model algoritma yang digunakan adalah **Apriori** dan **FP-Growth**. Pemilihan kedua algoritma ini didasarkan pada alasan bahwa keduanya merupakan metode paling umum dan banyak digunakan dalam praktik **Market Basket Analysis (MBA)**. Selain itu, keduanya memiliki pendekatan yang berbeda dalam mencari _frequent itemsets_ dan memiliki keunggulan dan kelemahannya masing-masing, sehingga relevan untuk dilakukan perbandingan performa secara langsung.

Untuk mengukur performa secara objektif, pengujian dilakukan menggunakan variasi ukuran dataset dari total **500.000 baris** data yang tersedia. Proporsi dataset yang digunakan untuk pengujian mencakup:

- **20%** dari total data (~100.000 rows)    
- **45%** dari total data (~225.000 rows)
- **75%** dari total data (~375.000 rows)    
- **100%** dari total data (500.000 rows)

Tujuan dari variasi ini adalah untuk mengevaluasi bagaimana skalabilitas masing-masing algoritma terhadap ukuran data yang berbeda. Dengan kata lain, apakah performa (terutama waktu komputasi dan jumlah aturan yang dihasilkan) tetap stabil seiring bertambahnya volume data.

# Evaluation
Dataset ini terdiri dari **23.204 *unique Transaction*** dan **3.768 *Unique Product***. Hal ini secara langsung meningkatkan ***Item Combination***, yang sangat memengaruhi performa algoritma **Apriori**. Hal ini dapat menimbulkan masalah kompleksitas secara **eksponen**, atau yang disebut sebagai **$log^n$ *problem***, karena Apriori harus melakukan pencarian berulang *(iteratif candidate generation)* untuk menghasilkan  _itemsets Candidate_.

Sebaliknya, **FP-Growth** diharapkan bisa menangani kompleksitas ini dengan lebih baik karena pendekatannya yang bersifat kompresi data melalui struktur **FP-Tree**, sehingga tidak perlu eksplisit menghasilkan semua kandidat kombinasi seperti pada Apriori.