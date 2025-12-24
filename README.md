## Klasifikasi Masker Wajah (Face Mask Detection) 😷

<img width="800" height="529" alt="image" src="https://github.com/user-attachments/assets/ba0601d1-1b8b-45d1-b8a4-38af9c4fd7f8" />


📌 Deskripsi Proyek

Proyek Face Mask Detection bertujuan untuk melakukan klasifikasi citra wajah ke dalam dua kelas, yaitu menggunakan masker dan tidak menggunakan masker, dengan memanfaatkan pendekatan Deep Learning. Data yang digunakan berupa citra wajah yang diproses menggunakan arsitektur Convolutional Neural Network (CNN).

Dalam proyek ini diterapkan tiga pendekatan model, yaitu CNN Base (non-pretrained) sebagai model dasar, Transfer Learning VGG16, dan Transfer Learning MobileNetV2 untuk meningkatkan kemampuan ekstraksi fitur citra. Ketiga model dilatih dan dievaluasi untuk membandingkan performa klasifikasi.

Model terbaik kemudian diintegrasikan ke dalam sebuah dashboard interaktif, yang memungkinkan pengguna mengunggah gambar wajah dan memperoleh hasil deteksi penggunaan masker secara otomatis.

📖 Latar Belakang

Perkembangan teknologi Deep Learning memungkinkan proses klasifikasi citra dilakukan secara otomatis dengan tingkat akurasi yang tinggi, termasuk dalam mendeteksi penggunaan masker wajah. Pemanfaatan Convolutional Neural Network (CNN) serta metode transfer learning seperti VGG16 dan MobileNetV2 memungkinkan sistem untuk mengekstraksi fitur citra wajah secara efektif.

Melalui pendekatan tersebut, sistem Face Mask Detection dapat dibangun untuk mengklasifikasikan citra wajah ke dalam dua kategori, yaitu menggunakan masker dan tidak menggunakan masker, serta diimplementasikan dalam sebuah dashboard interaktif untuk memudahkan pengguna dalam melakukan deteksi secara langsung.

🎯 Tujuan Pengembangan

Membangun Model Klasifikasi Masker Wajah
Mengembangkan model klasifikasi citra untuk memprediksi kondisi wajah menggunakan masker atau tidak menggunakan masker berdasarkan fitur visual pada citra wajah.

Evaluasi Performa Model
Menguji dan membandingkan beberapa model Deep Learning, yaitu CNN Base (Non-Pretrained), Transfer Learning VGG16, dan Transfer Learning MobileNetV2, guna memperoleh model dengan performa klasifikasi terbaik.

Membangun Sistem Deteksi Berbasis Dashboard
Mengembangkan aplikasi berbasis web/dashboard yang memungkinkan pengguna mengunggah citra wajah dan memperoleh hasil deteksi penggunaan masker secara otomatis dan interaktif.

📂 Sumber Data

Dataset yang digunakan adalah Face Mask Dataset dari Kaggle yang disediakan oleh omkargurav. Dataset ini terdiri dari citra wajah manusia dengan dua kelas, yaitu with_mask dan without_mask.

Dataset memiliki total sekitar 7.553 citra RGB, dengan rincian:

3.725 citra wajah menggunakan masker

3.828 citra wajah tanpa masker

Dataset ini banyak digunakan dalam penelitian Face Mask Detection dan sesuai untuk melatih serta mengevaluasi model Deep Learning dalam membedakan citra wajah berdasarkan penggunaan masker.

link dataset : https://www.kaggle.com/datasets/omkargurav/face-mask-dataset


⚙️ Preprocessing & Pemodelan

link preprocessing : https://colab.research.google.com/drive/1awXpgzsysZrgcu4expIfxxc9200W4h8v?usp=sharing

link model : https://drive.google.com/drive/folders/1LD8iMz-_e5AqiX5aVgem8Im-fzsv25qV?usp=sharing


1. Setup & Import Library

Pada tahap ini dilakukan pemanggilan library yang dibutuhkan untuk pengolahan data, pemodelan, dan evaluasi. TensorFlow/Keras digunakan untuk membangun dan melatih model Deep Learning, Scikit-learn untuk evaluasi model, serta Matplotlib dan Seaborn untuk visualisasi. Pengecekan versi TensorFlow dilakukan untuk memastikan kompatibilitas model.

2. Pengaturan Path Dataset & Parameter

Google Drive di-mount ke Google Colab agar dataset dapat diakses. Path dataset ditentukan pada variabel base_dir yang berisi dua kelas, yaitu with_mask dan without_mask. Parameter utama seperti ukuran citra (128×128), batch size (32), dan jumlah kelas (2) ditetapkan dan digunakan secara konsisten.

3. Data Generator Tanpa Augmentasi

Data citra diproses menggunakan ImageDataGenerator dengan normalisasi piksel (rescale=1./255). Dataset dibagi menjadi 80% data latih dan 20% data validasi menggunakan validation_split. Pada tahap ini belum diterapkan augmentasi data untuk mengamati performa model secara objektif. Data validasi juga digunakan sebagai data uji.

4. Fungsi Helper untuk Visualisasi & Evaluasi

Disediakan fungsi pendukung untuk analisis performa model. Fungsi plot_history() digunakan untuk menampilkan grafik akurasi dan loss selama pelatihan. Sementara itu, fungsi evaluate_model() digunakan untuk menampilkan classification report dan confusion matrix guna mengevaluasi kemampuan model dalam membedakan citra wajah bermasker dan tidak bermasker.

5. Model 1 – CNN Base (Non-Pretrained)

Model CNN Base dibangun tanpa bobot pra-latih menggunakan arsitektur Sequential yang terdiri dari lapisan Conv2D, MaxPooling, Dense, dan Dropout. Model ini digunakan sebagai baseline untuk klasifikasi citra wajah ke dalam dua kelas: with_mask dan without_mask.

Model dilatih selama 20 epoch menggunakan optimizer Adam dan loss categorical crossentropy. Hasil evaluasi menunjukkan akurasi yang sangat tinggi pada data latih dan validasi. Namun, performa yang terlalu tinggi mengindikasikan potensi overfitting, sehingga diperlukan perbandingan dengan model berbasis transfer learning.

6. Model 2 – Transfer Learning VGG16

Model ini menggunakan arsitektur VGG16 dengan bobot pra-latih ImageNet sebagai feature extractor. Seluruh layer dasar dibekukan (freeze), kemudian ditambahkan lapisan Flatten, Dense, dan Dropout untuk klasifikasi dua kelas.

Model dilatih selama 15 epoch dengan optimizer Adam dan loss categorical crossentropy. Hasil pelatihan menunjukkan akurasi latih dan validasi mencapai 100% dengan loss mendekati nol. Meskipun performanya sangat baik, hasil ini berpotensi menunjukkan overfitting atau keterbatasan distribusi data uji.

7. Model 3 – Transfer Learning MobileNetV2

Model ini menerapkan MobileNetV2 dengan bobot pra-latih ImageNet. Pada tahap preprocessing, citra diproses menggunakan fungsi mobilenet_preprocess. Seluruh layer MobileNetV2 dibekukan, kemudian ditambahkan Global Average Pooling, Dropout (0.4), dan Dense (softmax) untuk klasifikasi dua kelas.

Model dikompilasi menggunakan optimizer Adam (learning rate 1e-4) dan loss categorical crossentropy, serta dilatih selama 15 epoch. Hasil evaluasi menunjukkan akurasi validasi mencapai 100% dengan nilai loss yang sangat kecil. Namun, performa yang terlalu sempurna juga mengindikasikan potensi overfitting atau keterbatasan variasi data.

🖥️ Alur Kerja Aplikasi Dashboard (Streamlit)
1. Inisialisasi dan Pemuatan Model

Aplikasi mengimpor library yang dibutuhkan seperti Streamlit, NumPy, PIL, TensorFlow/Keras, dan OpenCV.
Model Deep Learning berformat .h5 dimuat menggunakan fungsi load_model_cached() yang dilengkapi dekorator @st.cache_resource, sehingga model hanya dimuat sekali dan disimpan di cache untuk menjaga performa aplikasi tetap cepat.

2. Pembangunan Tampilan Halaman

Pengaturan halaman dilakukan menggunakan st.set_page_config() untuk mengatur judul tab, ikon, dan layout.
Tampilan aplikasi dipercantik dengan CSS sederhana melalui st.markdown() yang membentuk header, kotak informasi, dan footer.

Komponen utama halaman meliputi:

Judul dan deskripsi aplikasi

Petunjuk penggunaan

Widget unggah gambar (st.file_uploader) untuk file JPG/PNG

3. Interaksi Pengguna (Upload Gambar)

Pengguna mengunggah satu citra wajah melalui widget upload.
Gambar yang diunggah dibaca menggunakan PIL dan ditampilkan sebagai pratinjau menggunakan st.image().

4. Proses Prediksi

Saat tombol Prediksi ditekan, sistem memproses gambar dengan langkah berikut:

Gambar dikonversi ke format OpenCV dan dilakukan deteksi wajah menggunakan Haar Cascade

Area wajah dipotong dan diubah ukurannya menjadi 128×128 piksel (jika wajah tidak terdeteksi, seluruh gambar digunakan)

Citra dipreproses menggunakan mobilenet_v2.preprocess_input

Model memprediksi probabilitas dua kelas (with_mask dan without_mask) dan menentukan kelas dengan probabilitas tertinggi

5. Menampilkan Hasil Prediksi

Hasil prediksi ditampilkan dalam bentuk label bahasa Indonesia:

with_mask → MEMAKAI MASKER

without_mask → TIDAK MEMAKAI MASKER

Jika hasilnya memakai masker, ditampilkan notifikasi hijau (st.success), sedangkan tidak memakai masker ditampilkan notifikasi merah (st.error), lengkap dengan nilai confidence.
Selain itu, probabilitas untuk kedua kelas juga ditampilkan sebagai informasi tambahan.

6. Siklus Interaksi Aplikasi

Setiap interaksi pengguna akan menjalankan ulang skrip Streamlit. Namun, model tetap diambil dari cache sehingga tidak perlu dimuat ulang.
Mekanisme ini membuat aplikasi terasa responsif, interaktif, dan real-time.

## 🔍 Hasil dan Analisis 🔍

## Tabel Perbandingan Hasil Classification Report

| Model                        | Precision | Recall | F1-Score | Accuracy | Jumlah Data Uji |
|-----------------------------|-----------|--------|----------|----------|-----------------|
| CNN Base (Non-Pretrained)   | 1.00      | 1.00   | 1.00     | 1.00     | 745             |
| VGG16 (Transfer Learning)   | 1.00      | 1.00   | 1.00     | 1.00     | 745             |
| MobileNetV2 (Transfer Learning) | 1.00  | 1.00   | 1.00     | 1.00     | 745             |


## 📊 Tabel Perbandingan Confusion Matrix Model

| Model | Confusion Matrix |
|------|------------------|
| **Base CNN** | <img width="691" height="607" alt="image" src="https://github.com/user-attachments/assets/6bcd82a2-2702-475e-b18e-54172be32fee" /> 
| **VGG16 Transfer Learning** | <img width="655" height="591" alt="image" src="https://github.com/user-attachments/assets/50608e35-9f70-4adb-be7f-af6a102e9eb7" /> 
| **MobileNetV2 Transfer Learning** | !<img width="651" height="597" alt="image" src="https://github.com/user-attachments/assets/6213e5fb-8f85-41b6-a3fd-32d725414065" /> 


## Learning Curves 📈


| Model | Learning Curves  |
|------|------------------|
| **Base CNN** | <img width="1289" height="473" alt="image" src="https://github.com/user-attachments/assets/1193e365-2cf8-40d4-a545-eb5cc49b871b" />
| **VGG16 Transfer Learning** | <img width="1283" height="486" alt="image" src="https://github.com/user-attachments/assets/1b0e3ba7-bac5-49dd-b553-9d3e924f22e7" />
| **MobileNetV2 Transfer Learning** |<img width="1257" height="478" alt="image" src="https://github.com/user-attachments/assets/aacabc82-fa6f-4320-affa-08f4525f5f1e" />


## 🎓 Sistem Sederhana Streamlit 🎓

Sistem sederhana berbasis Streamlit digunakan sebagai antarmuka pengguna (*user interface*) untuk menampilkan dan menguji hasil prediksi model secara interaktif. Aplikasi ini memungkinkan pengguna mengunggah citra wajah, kemudian sistem memproses citra tersebut menggunakan model *machine learning* yang telah dilatih untuk melakukan klasifikasi masker wajah (*Face Mask Detection*). Hasil prediksi ditampilkan secara langsung melalui tampilan web yang sederhana dan mudah digunakan. Streamlit memungkinkan sistem dijalankan tanpa konfigurasi frontend yang kompleks, sehingga cocok digunakan sebagai media demonstrasi, pengujian model, maupun validasi hasil penelitian.

## tampilan awal

<img width="1906" height="922" alt="image" src="https://github.com/user-attachments/assets/c02a7aad-795c-4d4f-a758-05e5aa674e93" />

## tampilan gambar dengan masker

<img width="1910" height="971" alt="image" src="https://github.com/user-attachments/assets/587e505a-914f-4fa8-bdc4-a0ddb3c39d30" />

## tampilan gambar tanpa masker

<img width="1912" height="969" alt="image" src="https://github.com/user-attachments/assets/87cc8e70-f56d-45eb-ac5c-9383919d1d35" />

## Link Live Demo

Coba aplikasi prediksi kesuksesan akademik mahasiswa secara langsung dengan mengunjungi tautan di bawah ini:

https://uapml081-qgqpcovuemu37ezxbcrdcw.streamlit.app/









