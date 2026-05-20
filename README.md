# Codec Vidéo MPEG-4 Simplifié

**Projet de Systèmes Multimédia**  
**Master 1 - Ingénierie du Logiciel**  
**Université des Sciences et de la Technologie Houari Boumediene (USTHB)**

---

##  Présentation

Implémentation complète d’un **codec vidéo hybride simplifié** inspiré des normes **MPEG-4 / H.264**.

Le projet couvre l’ensemble du pipeline moderne de compression vidéo :
- Conversion couleur RGB → YCbCr + sous-échantillonnage **4:2:0**
- Compression intra-trame (**I-frames**) via DCT 8×8
- Compression inter-trame (**P-frames**) avec estimation et compensation de mouvement + codage du résidu
- Codage entropique (Zigzag + RLE + Delta-coding des MV + zlib)

---

##  Performances

**Configuration standard (QF=79, GOP=8, 16 frames 128×128)**

| Métrique                    | Valeur           |
|-----------------------------|------------------|
| PSNR moyen                  | **34.39 dB**     |
| SSIM moyen                  | **0.9629**       |
| Ratio de compression        | **≈ 100×**       |
| Taille originale            | 768 KB           |
| Taille compressée           | ~7.7 KB          |

---

##  Utilisation

###  Mode Démo (recommandé pour tester rapidement)

```bash
python main.py --demo --qf 79 --gop 8 --analyse

# Extraire les frames d'une vidéo
python extract_frames.py video.mp4 --n 30 --size 320 240

# Encoder
python main.py --frames_dir ./frames --qf 70 --gop 8 --analyse

python main.py --decode video.bin --output_dir decoded_frames

.
.
├── main.py                 # Point d'entrée
├── encoder.py              # Cœur du codec (encode/decode)
├── evaluation.py           # Métriques + visualisation
├── extract_frames.py       # Extraction de frames depuis vidéo
├── pipeline_visualisation.png
├── qf_vs_compression.png
├── gop_vs_compression.png
├── video.bin               # Fichier compressé généré
└── Rapport final de projet .pdf   # Rapport du projet

Fonctionnalités implémentées

Prétraitement YCbCr 4:2:0
I-frames (DCT + Quantification + Zigzag + RLE)
P-frames (Motion Estimation + Compensation + Résidu)
Delta-coding des vecteurs de mouvement
RLE robuste (gestion des longs runs)
Visualisation détaillée du pipeline
Analyses paramétriques (QF et GOP)
Métriques PSNR / SSIM


Auteur : Younes Abdelmoutaleb Guiti / Mohamed Bouhafs
Année universitaire : 2025/2026
