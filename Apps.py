
import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing import image, image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.backend import clear_session

st.title("Cari tahu jenis batik hanya dengan satu jepretan")
about = '''Tahu batik menyediakan layanan untuk mendeteksi jenis batik pada motif kain,
dengan cara mengambil gambar kain batik yang ingin dicari tahu motifnya, kemudian program
akan memindai jenis motif batik tersebut.'''
st.markdown(about)

st.header("Jenis batik yang tersedia")
info = '''Saat ini aplikasi mampu mengenali 15 jenis batik diantaranya

1. Batik Bali
2. Batik Betawi
3. Batik Cenderawasih
4. Batik Dayak
5. Batik Geblek Renteng
6. Batik Ikat Celup
7. Batik Insang
8. Batik Kawung
9. Batik Lasem
10. Batik Megamendung
11. Batik Pala
12. Batik Parang
13. Batik Poleng
14. Batik Sekar Jagad
15. Batik Tambal '''
st.markdown(info)

st.header("Cara Penggunaan")
guide = '''
1. Arahkan Kamera kearah kain yang ingin dipindai
2. Pastikan fokus kearah motif
3. Ambil Gambar
4. Tunggu hasil
5. Hasil pemindaian selesai'''
st.markdown(guide)

st.header("Pindai Batik")
picture = st.camera_input("Ambil gambar batik")

if picture:
    # Convert UploadedFile to string path
    model = load_model("mdl85.h5")
    file_path = picture

    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class_index = np.argmax(predictions[0])
    print('class index is:', predicted_class_index)
    st.image(picture)
    if(predicted_class_index==0):
        st.write("Batik Bali")
        bali = ''' Motif batik Bali kebanyakan menggambarkan fauna yang ada di pulau tersebut, baik nyata maupun dari mitos.

        Para pengerajin juga seringkali terinspirasi dari flora lokal, serta menggambarkan kegiatan sehari-hari masyarakat Bali.

        Secara garis besar, batik Bali memiliki ciri sebagai berikut: 

        Corak lokal berupa bunga kamboja, burung, pemandangan alam, ikan dan keseharian masyarakat Bali. 
        Motif upacara keagamaan atau ritual tertentu yang sudah menjadi adat dan budaya. 
        Ada lambang dari makhluk mitologi yang disebut dalam legenda atau mitos lokal.
        sumber https://www.orami.co.id/magazine/batik-bali'''
        st.markdown(bali)
    if(predicted_class_index==1):
        st.write("Batik Betawi")
        betawi = ''' Batik khas betawi memiliki makna yang mendalam dalam kehidupan masyarakat Betawi.
        Batik ini sebagai keseimbangan alam semesta dan pemenuhan hidup yang sejahtera serta penuh berkah.
        Batik ini juga menjadi upaya masyarakat Betawi dalam mempertahankan nilai-nilai budayanya yang telah ada secara turun-temurun dari leluhur mereka.
        Batik tradisional ini biasanya menggunakan warna dasar yang cerah, corak pada motif- motif batik dari betawi menggambarkan budaya yang ada di Betawi 
        seperti Ondel-ondel, Sungai Ciliwung, Tanjidor dan Peta Ceila serta beberapa pepohonan. 
        Keunikan tersebut menjadi ciri khas tersendiri bagi Batik tersebut.
        sumber https://museumnusantara.com/batik-betawi/''' 
        st.markdown(betawi)
    if(predicted_class_index==2):
        st.write("Batik Cendrawasih")
        cendrawasih=''' Jika menyangkut flora dan fauna Papua, motif satu ini sudah tak asing lagi.

        Burung cendrawasih menjadi salah satu ikon yang dimiliki Papua.

        Disebut juga dengan julukan 'birds of paradise', yakni warna mencolok dari tubuh burung tersebut.

        Sering dipakai dalam corak batik yang dipenuhi dengan dedaunan dan tanaman khas Papua.

        Warna yang menjadi khas dalam batik Papua ini adalah warna biru, hijau, merah, dan kuning. Bisa dikenakan untuk berbagai acara formal
        sumber https://www.orami.co.id/magazine/batik-papua'''
        st.markdown(cendrawasih)
    if(predicted_class_index==3):
        st.write("Batik Dayak")
        dayak=''' Batik Dayak, yang berasal dari pulau Kalimantan yang subur, memiliki makna budaya yang kaya. 
        Mari selidiki sejarah, filosofi, dan motif populernya.
        
        Orang-orang Dayak, yang tinggal di berbagai daerah di Kalimantan, adalah penjaga tradisi batik yang unik ini.
        Kehidupan sehari-hari mereka terkait erat dengan kegiatan sungai, karena sungai berfungsi sebagai mata pencaharian utama mereka. 
        Dengan demikian, Batik Dayak mewujudkan keberanian dan harmoni budaya dalam komunitas mereka.'''
        st.markdown(dayak)
    if(predicted_class_index==4):
        st.write("Batik Geblek Renteng")
        geblek='''makna dan filosofi yang terkandung dalam motif geblek renteng, yang kini menjadi icon Motif batik khas Kabupaten Kulon Progo.
        Motif yang telah menjadi ikon Kulon Progo itu terdiri dari gambar geblek sebagai motif utama dan sekian banyak simbol yang menunjukkkan kekayaan alam dan Kondisi Kabupaten Kulon Progo.
        Geblek dijadikan motif utama sebab geblek merupakan makanan khas pribumi Kulon Progo.

        Di antara motif geblek tersebut, ditorehkan emblem Binangun yang dicerminkan sebagai kuncup bunga yang bakal mekar, mempunyai makna bahwa Kulon progo merupakan wilayah yang sebentar lagi bakal mekar menjadi permata estetis dari pulau jawa. 
        Di sampingnya ada motif buah manggis yang merupakan tumbuhan khas Kulon Progo. Ketiga motif tersebut diciptakan dengan pola naik turun sebagai perlambang bahwa kenampakan alam di Kulon Progo yang 
        paling bervariasi, mulai dari pegunungan, dataran tinggi, sampai dataran rendah dan pantai. 
        sumber https://www.motifbatik.web.id/2019/01/filosofi-dan-sejarah-batik-motif-geblek.html'''
        st.markdown(geblek)
    if(predicted_class_index==5):
        st.write("Batik Ikat Celup")
        celup=''' Ikat celup atau Jumputan (tie-dye) adalah teknik mewarnai kain dengan cara mengikat kain dengan cara tertentu sebelum dilakukan pencelupan. 
        Di beberapa daerah di Indonesia, teknik ini dikenal dengan berbagai nama lain seperti jumputan pelangi atau cinde (Palembang), 
        tritik atau jumputan (Jawa), serta sasirangan (Banjarmasin). Teknik ikat celup sering dipadukan dengan teknik lain seperti batik.
        sumber https://id.wikipedia.org/wiki/Ikat_celup'''
        st.markdown(celup)
    if(predicted_class_index==6):
        st.write("Batik Insang")
        insang=''' 
        Motif batik insang adalah motif yang populer di kalangan ibu-ibu Pontianak.
        Batik insang di gunakan untuk perlengkapan adat atau upacara adat, misalnya pernikahan, sunatan dan perayaan hari.
        Corak kain ini berantai dan berombak namun simetris.
        sumber https://pelajarindo.com/batik-kalimantan-barat-sejarah-motif-gambar-penjelasan/'''
        st.markdown(insang)
    if(predicted_class_index==7):
        st.write("Batik Kawung")
        kawung='''Batik Kawung adalah motif batik yang bentuknya berupa bulatan mirip buah kawung (sejenis kelapa atau kadang juga dianggap sebagai aren atau kolang-kaling) yang ditata rapi secara geometris. 
        Kadang, motif ini juga ditafsirkan sebagai gambar bunga lotus (teratai) dengan empat lembar mahkota bunga yang merekah.
        Lotus adalah bunga yang melambangkan umur panjang dan kesucian.
        Batik kawung ini terkenal di bagian Jawa 
        sumber https://id.wikipedia.org/wiki/Batik_Kawung'''
        st.markdown(kawung)
    if(predicted_class_index==8):
        st.write("Batik Lasem")
        lasem='''
        Dalam sebuah artikel berjudul Aktualisasi Nilai Cina Dalam Batik Lasem oleh Rizali dan Sudardi, pada masa Kerajaan Hindu Majapahit abad 13-14 M,
        batik digunakan sebagai benda magis untuk sarana mistik. 
        Pola hias batik digunakan untuk kepentingan keagamaan bersifat simbolis dan bermakna sakral, seperti ragam hias Kawung,
        Bunga Padma Ceplok, Kalacakra atau Nitik Ceplok, Sayap Garuda (Lar, Sidomukti), Gringsing (Urna) dan Parang yang hanya digunakan oleh Raja dan anggota kerajaan.
        sumber https://katadata.co.id/berita/nasional/611e2ca006335/keindahan-batik-lasem-hasil-akulturasi-budaya-jawa-dan-tiongkok'''
        st.markdown(lasem)      
    if(predicted_class_index==9):
        st.write("Batik Megamendung")
        megamendung='''
        Batik Megamendung (Hanacaraka: ꦩꦺꦒꦩꦼꦤ꧀ꦢꦸꦁ) merupakan karya seni batik yang identik dan bahkan menjadi ikon batik daerah Cirebon dan daerah Indonesia lainnya.
        Motif batik ini mempunyai kekhasan yang tidak ditemui di daerah penghasil batik lain. 
        Bahkan karena hanya ada di Cirebon dan merupakan mahakarya, Departemen Kebudayaan dan Pariwisata akan mendaftarkan motif megamendung ke UNESCO untuk mendapatkan pengakuan sebagai salah satu warisan dunia. 
        sumber https://id.wikipedia.org/wiki/Batik_Megamendung'''
        st.markdown(megamendung)
    if(predicted_class_index==10):
        st.write("Batik Pala")
        pala=''' 
        Motif Batik yang dihasilkan oleh pengerajin di tanah Maluku sendiri terdiri dari beberapa motif, tetapi yang paling menjadi favorit dari warga Maluku, dan wisatawan dari luar daerah yaitu Batik dengan motif Pala, 
        Cengkih, dan Parang Salawaku. Pala dan Cengkih sendiri dijadikan motif Batik dari daerah Maluku dikarenakan Pala dan Cengkih merupakan hasil bumi yang paling banyak diminati dari Maluku. 
        Selain itu, Parang dan Salawaku turut menjadi motif dari Batik Maluku dikarenakan Parang dan Salawaku merupakan senjata khas dari daerah Maluku.
        sumber https://www.sisternet.co.id/read/280790-batik-maluku-kekayaan-budaya-yang-tidak-kalah-unik'''
        st.markdown(pala)
    if(predicted_class_index==11):
        st.write("Batik Parang")
        parang=''' Batik Parang adalah salah satu motif batik yang paling tua di Indonesia. Parang berasal dari kata "pèrèng" yang berarti "lèrèng". 
        Maksudnya, bentuk motif batik parang itu berupa huruf “S” yang digambar secara berkaitan satu sama lain dan membentuk diagonal miring layaknya lèrèng gunung.
        Perengan menggambarkan sebuah garis menurun dari tinggi ke rendah secara diagonal. Susunan motif S jalin-menjalin tidak terputus melambangkan kesinambungan.
        Bentuk dasar huruf S diambil dari ombak samudra yang menggambarkan semangat yang tidak pernah padam. Batik ini merupakan batik asli Indonesia yang sudah ada sejak zaman keraton Mataram Kartasura (Solo).
        sumber https://id.wikipedia.org/wiki/Batik_Parang'''
        st.markdown(parang)
    if(predicted_class_index==12):
        st.write("Batik Poleng")
        poleng='''Poleng atau corak papan catur adalah pola kotak-kotak sederhana yang terbentuk dari selang-seling warna gelap dan terang, biasanya hitam dan putih.
        Di Bali, kain dengan motif seperti ini disebut sebagai kain poleng. Kain poleng melambangkan keseimbangan antara dua hal yang bertolak belakang. 
        Corak poleng juga dikenal dalam kebudayaan Jawa, khususnya dalam kerajinan batik.
        sumber https://id.wikipedia.org/wiki/Poleng'''
        st.markdown(poleng)
    if(predicted_class_index==13):
        st.write("Batik Sekar Jagad")
        sekar='''
        Batik Sekar Jagad (aksara Jawa: ꦱꦼꦏꦂꦗꦒꦠ꧀) adalah salah satu motif batik yang berasal dari Solo dan Yogyakarta. Motif ini mengandung makna aneka rupa keindahan yang terjalin menjadi satu atau melingkupi keseluruhan keindahan. 
        Ada pula yang beranggapan bahwa motif Sekar Jagad sebenarnya berasal dari kata "kar jagad" yang diambil dari bahasa Jawa (kar "peta"; jagad "dunia"), sehingga motif ini juga melambangkan keragaman di seluruh dunia.
        sumber https://id.wikipedia.org/wiki/Batik_Sekar_Jagad'''
        st.markdown(sekar)
    if(predicted_class_index==14):
        st.write("Batik Tambal")
        tambal='''
        Batik tambal (aksara Jawa: ꦧꦛꦶꦏ꧀ꦠꦩ꧀ꦧꦭ꧀) adalah motif batik yang menggabungkan atau "menambal" berbagai macam motif batik lainnya dalam bidang-bidang segitiga yang disusun sedemikian rupa. 
        Bidang-bidang segitiga tersebut biasanya tercipta dari bidang persegi empat yang lebih besar, dengan garis-garis yang memotong dari setiap sudutnya. 
        sumber https://id.wikipedia.org/wiki/Batik_Tambal'''
        st.markdown(tambal)
