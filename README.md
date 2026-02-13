# Load-Extraction-Force-Relationship

Bar element - Joint Load Cap korelasyon analiz aracı.

BDF'den bar elementler ve bağlı CQUAD4/CTRIA3 elementleri çıkarır, OP2'den kuvvetleri/fluxları okur, H5'ten Joint Load Cap verilerini alır ve aralarındaki korelasyonu hesaplar.

## Kurulum

```bash
pip install -r requirements.txt
```

## Gerekli Input Dosyaları

| Dosya | Açıklama |
|-------|----------|
| **BDF** | Nastran bulk data dosyası (element bağlantıları) |
| **OP2** | Nastran output dosyası (axial force, fluxlar) |
| **H5** | Joint Load Extraction dosyası (Joint Load Cap tablosu) |
| **Excel** | "Bar Element Set" sheet'i olan Excel (bar element listesi) |

### Excel Input Formatı

"Bar Element Set" adlı sheet'te ilk kolonda bar element ID'leri bulunmalıdır:

| Bar_Element_ID |
|----------------|
| 100001 |
| 100002 |
| 100003 |

### H5 Dosya Yapısı

```
Joint Load Extraction (root)
└── Joint Load Cap
    └── table
        ├── Bar EID / ID
        ├── FBearingX
        ├── FBearingY
        ├── NX Bypass
        ├── NY Bypass
        └── NXY Bypass
```

## Kullanım

```bash
# Temel kullanım
python main.py --bdf model.bdf --op2 model.op2 --h5 joint_loads.h5 --excel input.xlsx

# Çıktı dosyası ve subcase belirterek
python main.py --bdf model.bdf --op2 model.op2 --h5 joint_loads.h5 --excel input.xlsx \
    --output results.xlsx --subcase 1

# H5 grup yolunu manuel belirterek
python main.py --bdf model.bdf --op2 model.op2 --h5 joint_loads.h5 --excel input.xlsx \
    --h5-group "Joint Load Cap/table"

# Detaylı çıktı
python main.py --bdf model.bdf --op2 model.op2 --h5 joint_loads.h5 --excel input.xlsx -v
```

## Çıktı Excel Sheet'leri

| Sheet | İçerik |
|-------|--------|
| **Bar Element Set** | Input bar element listesi |
| **Element Connectivity** | Bar elementler ve bağlı QUAD/TRIA elementleri |
| **OP2 Bar Forces** | Bar element axial force, shear, bending, torque |
| **OP2 Shell Fluxes** | Shell element NX, NY, NXY fluxları |
| **H5 Joint Loads** | Joint Load Cap tablosu |
| **Correlation Summary** | Pearson R, R², regresyon katsayıları |
| **Bar_XXXXX** | Her bar element için detay sayfası |

## İşlem Adımları

1. **Excel Okuma**: "Bar Element Set" sheet'inden bar element ID listesi okunur
2. **BDF Parse**: Bar elementlerin 2 node'u (GA, GB) bulunur; bu node'lara bağlı CQUAD4/CTRIA3 elementleri tespit edilir
3. **OP2 Okuma**: Bar elementler için axial force, shell elementler için membrane fluxları (NX, NY, NXY) çıkarılır
4. **H5 Okuma**: Joint Load Cap tablosundan FBearingX, FBearingY, NX/NY/NXY Bypass değerleri alınır
5. **Korelasyon**: Her bar element ve element tipi için ayrı ayrı Pearson korelasyonu ve lineer regresyon hesaplanır
6. **Rapor**: Tüm sonuçlar formatlanmış Excel dosyasına yazılır

## Korelasyon Analizi

Her bar element için 4 tip korelasyon hesaplanır:

### BAR Korelasyonu
- Bar Axial Force ↔ FBearingX, FBearingY
- Bar Shear ↔ FBearingX, FBearingY

### CQUAD4 Korelasyonu (her bağlı quad için ayrı)
- Shell NX ↔ NX Bypass
- Shell NY ↔ NY Bypass
- Shell NXY ↔ NXY Bypass

### CTRIA3 Korelasyonu (her bağlı tria için ayrı)
- Shell NX ↔ NX Bypass
- Shell NY ↔ NY Bypass
- Shell NXY ↔ NXY Bypass

### COMBINED Korelasyonu
- Bar kuvvetleri + ortalama shell fluxları birlikte analiz
- QUAD ve TRIA ortalamaları ayrı ayrı

## Fiziksel Arka Plan

Bir bağlantı noktasında (fastener = bar element):

- **Bearing Load (FBearing)**: Bağlantı elemanı üzerinden panele aktarılan yük. Bar axial force'un bileşenleridir.
- **Bypass Load (N Bypass)**: Bağlantı elemanını atlayarak panelden geçen yük. Panel toplam yükü - bearing katkısı.

```
N_total = N_bearing + N_bypass
FBearingX ∝ Bar Axial Force (X bileşeni)
FBearingY ∝ Bar Axial Force (Y bileşeni)
NX Bypass = NX_total - FBearingX katkısı
NY Bypass = NY_total - FBearingY katkısı
NXY Bypass = NXY_total - Bearing shear katkısı
```
