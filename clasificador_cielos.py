"""
Sky Classifier — Professional Edition v3
Bugs corregidos: indices de features, umbral no-relacionado,
graficas de evaluacion, segmentacion simplificada.
"""

import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QTabWidget, QTextEdit,
    QFileDialog, QMessageBox, QDialog, QGridLayout, QLineEdit,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QProgressBar, QComboBox, QSlider, QAbstractScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap, QImage, QCursor
from PIL import Image

# ═══════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════
FONT_MONO = 'Courier New'
FONT_SANS = 'Segoe UI'

C_BG        = '#0a0a0f'
C_SURFACE   = '#111118'
C_PANEL     = '#161620'
C_CARD      = '#1c1c28'
C_BORDER    = '#2a2a3a'
C_BORDER_LT = '#3a3a50'
C_INK       = '#e8e8f0'
C_INK_DIM   = '#7070a0'
C_INK_FAINT = '#404060'
C_SKY       = '#4fc3f7'
C_CLOUD     = '#b0bec5'
C_SUN       = '#ffd54f'
C_STORM     = '#ef5350'
C_ACCENT    = '#7c6af4'

C_CLASSES = {0:C_SKY, 1:C_CLOUD, 2:'#e0e0e0', 3:C_STORM,
             4:'#66bb6a', 5:C_SUN, 6:C_ACCENT, 7:'#ff8a65'}

NOMBRES_DEFAULT = ['Despejado','Semi-Nublado','Nublado',
                   'No relacionado','Clase 5','Clase 6','Clase 7','Clase 8']

STYLE_BASE = f"""
QWidget {{ background: transparent; color: {C_INK};
           font-family: '{FONT_SANS}'; font-size: 11px; }}
QScrollBar:vertical {{ background:{C_SURFACE}; width:6px; border-radius:3px; }}
QScrollBar::handle:vertical {{ background:{C_BORDER_LT}; border-radius:3px; min-height:30px; }}
QScrollBar::handle:vertical:hover {{ background:{C_ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0px; }}
QScrollBar:horizontal {{ background:{C_SURFACE}; height:6px; border-radius:3px; }}
QScrollBar::handle:horizontal {{ background:{C_BORDER_LT}; border-radius:3px; }}
"""

def _lighten(h, a=20):
    h = h.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'#{min(255,r+a):02x}{min(255,g+a):02x}{min(255,b+a):02x}'

def style_btn(color=C_ACCENT, text_color='white', pad='9px 14px'):
    hov = _lighten(color, 20)
    return (f'QPushButton{{background:{color};color:{text_color};border:none;'
            f'padding:{pad};border-radius:6px;font-weight:600;font-size:11px;}}'
            f'QPushButton:hover{{background:{hov};}}'
            f'QPushButton:pressed{{background:{color};}}'
            f'QPushButton:disabled{{background:{C_BORDER};color:{C_INK_FAINT};}}')

def style_input():
    return (f'background:{C_CARD};color:{C_INK};border:1px solid {C_BORDER};'
            f'border-radius:6px;padding:7px 10px;font-size:11px;')

# ═══════════════════════════════════════════════════════════════
# PROCESAMIENTO DE IMAGEN
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# PROCESAMIENTO DE IMAGEN — Multi-cue sky detection
# ═══════════════════════════════════════════════════════════════

def filtro_paso_altas(canal):
    """Gradiente de bordes normalizado 0-1."""
    c  = canal.astype(np.float32) / 255.0
    gx = np.abs(np.diff(c, axis=1, prepend=c[:,:1]))
    gy = np.abs(np.diff(c, axis=0, prepend=c[:1,:]))
    return np.sqrt(gx**2 + gy**2)


def _rgb_a_hsv_np(arr):
    """Convierte arr uint8 RGB a H(0-360), S(0-1), V(0-1) float32."""
    from PIL import Image as _PILImage
    hsv_pil = _PILImage.fromarray(arr).convert('HSV')
    hsv = np.array(hsv_pil, dtype=np.float32)
    # PIL HSV: H 0-255 → 0-360, S 0-255 → 0-1, V 0-255 → 0-1
    return hsv[:,:,0]/255.0*360.0, hsv[:,:,1]/255.0, hsv[:,:,2]/255.0


def detectar_mascara_cielo(arr):
    """
    Detección multi-cue del cielo usando:
      RGB diferencial, HSV (matiz azul/blanco/gris),
      canal B relativo, gradiente de textura LOCAL, posición vertical.

    Clave anti-edificio: las nubes/cielo son SUAVES (bajo gradiente local),
    los edificios/cúpulas tienen TEXTURA (alto gradiente local), aunque
    sean blancos. Se penalizan las zonas texturadas.

    Retorna: mask(bool), score(float32), grad_norm(float32)
    """
    H, W = arr.shape[:2]
    r = arr[:,:,0].astype(np.float32)
    g = arr[:,:,1].astype(np.float32)
    b = arr[:,:,2].astype(np.float32)

    # ── Gradiente de textura ──────────────────────────────────
    gm = (filtro_paso_altas(arr[:,:,0]) +
          filtro_paso_altas(arr[:,:,1]) +
          filtro_paso_altas(arr[:,:,2])) / 3.0
    gn = gm / (gm.max() + 1e-8)
    suavidad = 1.0 - gn

    # ── Textura LOCAL (promedio del gradiente en ventana) ─────
    # Distingue cúpula/edificio (texturado) de nube/cielo (suave)
    try:
        from scipy.ndimage import uniform_filter
        grad_local = uniform_filter(gm, size=11)
    except Exception:
        # Fallback: promedio con convolución simple
        grad_local = gm.copy()
        for _ in range(3):
            grad_local = (grad_local +
                          np.roll(grad_local,1,0) + np.roll(grad_local,-1,0) +
                          np.roll(grad_local,1,1) + np.roll(grad_local,-1,1)) / 5.0
    grad_local_n = grad_local / (grad_local.max() + 1e-8)
    # Penalización por textura: 1.0 si suave, baja si texturado
    suavidad_local = 1.0 - np.clip(grad_local_n * 2.0, 0, 1)

    # ── HSV ──────────────────────────────────────────────────
    hue, sat, val = _rgb_a_hsv_np(arr)

    # ── Cue 1 — Azul cielo (RGB diferencial) ──────────────────
    c_az_rgb = ((b > r + 8) & (b > g + 4) & (b > 50)).astype(np.float32)

    # ── Cue 2 — Azul cielo (HSV matiz) ───────────────────────
    c_az_hsv = ((hue > 180) & (hue < 270) &
                (sat > 0.08) & (val > 0.12)).astype(np.float32)

    # ── Cue 3 — Nubes blancas (alta luminosidad, sat baja) ───
    c_blanco  = ((sat < 0.22) & (val > 0.65)).astype(np.float32)

    # ── Cue 4 — Nubes gris claro (overcast) ──────────────────
    c_gris    = ((sat < 0.25) & (val > 0.30) & (val < 0.72)).astype(np.float32)

    # ── Cue 5 — Cielo oscuro / atardecer (RGB) ────────────────
    c_az_osc  = ((b > r + 5) & (b > g + 3) &
                 (b > 25) & (val < 0.50)).astype(np.float32)

    # ── Cue 6 — Canal B relativo ─────────────────────────────
    b_rel     = b / (r + g + b + 1e-6)
    c_brel    = (b_rel > 0.33).astype(np.float32)

    # ── Cue 7 — Cielo gris uniforme (nubes densas) ────────────
    rg_diff   = np.abs(r - g)
    gb_diff   = np.abs(g - b)
    c_unif    = ((rg_diff < 18) & (gb_diff < 18) &
                 (r > 85) & (r < 210)).astype(np.float32)

    # ── Combinación ponderada de cues de color ────────────────
    color_score = (
        c_az_rgb * 0.22 +
        c_az_hsv * 0.22 +
        c_blanco * 0.16 +
        c_gris   * 0.12 +
        c_az_osc * 0.08 +
        c_brel   * 0.10 +
        c_unif   * 0.10
    )

    # ── Posición vertical ─────────────────────────────────────
    pesos = np.tile(
        np.linspace(1.0, 0.03, H).reshape(-1, 1), (1, W))

    # ── Score final (ahora multiplicado por suavidad_local) ───
    # La textura local actúa como FILTRO: una zona blanca pero
    # texturada (cúpula) pierde mucho score.
    score_base = (color_score * 0.55 +
                  suavidad    * 0.18 +
                  pesos       * 0.15)
    # Aplicar penalización por textura local de forma multiplicativa
    # para azul-cielo es menos estricto (el cielo azul casi no tiene textura)
    es_azul_fuerte = ((c_az_rgb + c_az_hsv) > 0.5).astype(np.float32)
    # Para zonas no claramente azules, la textura local pesa más
    factor_textura = 0.45 + 0.55 * suavidad_local
    factor_textura = np.where(es_azul_fuerte > 0.5,
                              0.75 + 0.25*suavidad_local,  # azul: leve
                              factor_textura)               # blanco/gris: fuerte
    score = score_base * factor_textura + suavidad_local * 0.12

    # ── Umbral adaptativo ─────────────────────────────────────
    sup_mean = score[:H//3, :].mean()
    umb = float(np.clip(sup_mean * 0.78, 0.20, 0.55))

    return (score >= umb), score, gn


def _morfologia_mascara(mask, iters=3):
    """
    Limpieza morfológica: rellena huecos internos y suaviza bordes.
    Usa scipy si está disponible, fallback con numpy puro.
    """
    try:
        from scipy.ndimage import (binary_fill_holes,
                                   binary_dilation,
                                   binary_erosion)
        m = binary_fill_holes(mask)
        m = binary_dilation(m, iterations=iters)
        m = binary_erosion(m,  iterations=iters)
        return m
    except Exception:
        # Fallback: erosión/dilatación manual con ventana 3x3
        m = mask.astype(np.uint8)
        for _ in range(iters):
            # dilatación
            d = np.zeros_like(m)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    d = np.maximum(d, np.roll(np.roll(m,dy,0),dx,1))
            m = d
        for _ in range(iters):
            # erosión
            e = np.ones_like(m)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    e = np.minimum(e, np.roll(np.roll(m,dy,0),dx,1))
            m = e
        return m.astype(bool)


def segmentar_cielo(img_path, size=(520, 346)):
    """
    Segmenta el cielo en una imagen.

    Retorna:
      arr     — imagen original (H,W,3) uint8
      mask    — máscara booleana limpia (True = cielo)
      overlay — imagen con cielo sombreado azul, no-cielo oscurecido
      score   — mapa de confianza continuo
      grad    — gradiente normalizado
    """
    try:
        img = Image.open(img_path).convert('RGB').resize(
            size, Image.LANCZOS)
        arr = np.array(img, dtype=np.uint8)

        # 1. Detección multi-cue
        mask_raw, score, grad = detectar_mascara_cielo(arr)

        # 2. Limpieza morfológica
        mask = _morfologia_mascara(mask_raw, iters=3)

        # 3. Construir overlay
        #    Cielo: tinte azul celeste (alpha 0.55)
        #    No-cielo: oscurecer al 26%
        ov    = arr.copy().astype(np.float32)
        alpha = 0.55
        fill  = np.array([25, 120, 255], dtype=np.float32)  # azul cielo
        ov[mask,  0] = ov[mask,  0] * (1-alpha) + fill[0] * alpha
        ov[mask,  1] = ov[mask,  1] * (1-alpha) + fill[1] * alpha
        ov[mask,  2] = ov[mask,  2] * (1-alpha) + fill[2] * alpha
        ov[~mask, :] = ov[~mask, :] * 0.26
        overlay = np.clip(ov, 0, 255).astype(np.uint8)

        return arr, mask, overlay, score, grad
    except Exception:
        return None, None, None, None, None


def lbp_simple(canal):
    h, w = canal.shape
    lbp = np.zeros((h,w), dtype=np.float32)
    c   = canal.astype(np.float32)
    for p, angle in enumerate(np.linspace(0, 2*np.pi, 8, endpoint=False)):
        dx = int(round(np.cos(angle)))
        dy = int(round(np.sin(angle)))
        shifted = np.roll(np.roll(c, dy, axis=0), dx, axis=1)
        lbp += (shifted >= c).astype(np.float32) * (2**p)
    return lbp / 255.0

def extraer_features(img_path, size=(64,64)):
    """
    Extrae 100 features por imagen:
      [0:48]   histograma RGB (16 bins x 3)
      [48:54]  media(3) + std(3) RGB
      [54:61]  ratios color global + saturacion + contraste
      [61:64]  ratios color zona superior
      [64:72]  gradiente / textura
      [72:74]  score_cielo_global, score_cielo_sup   ← INDICES CLAVE
      [74:78]  LBP zona superior
      [78:82]  luminancia zona superior
      [82:88]  HSV media+std (H,S,V) globales
      [88:94]  HSV media+std zona superior
      [94:100] cues multi-canal (az_hsv_sup, bl_hsv_sup, gr_hsv_sup,
                                  b_rel_sup, uniformidad_sup, sat_sup)
    """
    try:
        img  = Image.open(img_path).convert('RGB').resize(size)
        arr  = np.array(img, dtype=np.float32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        H, W    = size[1], size[0]
        total   = H * W

        # ── [0:48] Histogramas RGB ─────────────────────────
        hr = np.histogram(r, bins=16, range=(0,255))[0] / total
        hg = np.histogram(g, bins=16, range=(0,255))[0] / total
        hb = np.histogram(b, bins=16, range=(0,255))[0] / total

        # ── [48:54] Media + Std ─────────────────────────────
        media = arr.mean(axis=(0,1)) / 255.0
        std   = arr.std(axis=(0,1))  / 255.0

        # ── [54:61] Ratios de color globales ───────────────
        pct_az = np.sum((b>r+8) &(b>g+4) &(b>50)) / total
        pct_gr = np.sum((np.abs(r.astype(int)-g)<20)&
                        (np.abs(g.astype(int)-b)<20)&(r<200)) / total
        pct_bl = np.sum((r>185)&(g>185)&(b>185)) / total
        pct_ve = np.sum((g>r+15)&(g>b+15)) / total
        pct_os = np.sum((r<55) &(g<55) &(b<55)) / total
        mx  = arr.max(axis=2); mn_a = arr.min(axis=2)
        sat_rgb  = np.where(mx>0, (mx-mn_a)/mx, 0).mean()
        contr    = arr.std() / 255.0

        # ── [61:64] Zona superior ───────────────────────────
        zs = arr[:H//3,:,:]
        r_s, g_s, b_s = zs[:,:,0], zs[:,:,1], zs[:,:,2]
        ts = (H//3) * W
        paz = np.sum((b_s>r_s+8)&(b_s>g_s+4)&(b_s>50)) / ts
        pbl = np.sum((r_s>185)&(g_s>185)&(b_s>185)) / ts
        pgr = np.sum((np.abs(r_s.astype(int)-g_s)<25)&
                     (np.abs(g_s.astype(int)-b_s)<25)&(r_s<195)) / ts

        # ── [64:72] Gradiente y textura ─────────────────────
        lum  = (0.299*r + 0.587*g + 0.114*b) / 255.0
        gx   = np.abs(np.diff(lum, axis=1, prepend=lum[:,:1]))
        gy   = np.abs(np.diff(lum, axis=0, prepend=lum[:1,:]))
        gm   = np.sqrt(gx**2 + gy**2)
        gg   = gm.mean()
        gs_g = gm[:H//3,:].mean()
        gmd  = gm[H//3:2*H//3,:].mean()
        gi   = gm[H//2:,:].mean()
        gv   = gm.std()
        rtex = gs_g / (gi + 1e-6)
        suav = 1.0 - (b_s/255.0).std()
        rhv  = gx.mean() / (gy.mean() + 1e-6)

        # ── [72:74] Score de cielo multi-cue (índices CLAVE) ─
        arr_u8 = arr.astype(np.uint8)
        _, sm, _ = detectar_mascara_cielo(arr_u8)
        scg = float(sm.mean())            # idx 72
        scs = float(sm[:H//3,:].mean())   # idx 73

        # ── [74:78] LBP textura zona superior ───────────────
        lbp_sup = lbp_simple(lum[:H//3,:])
        lm  = lbp_sup.mean()
        ls  = lbp_sup.std()
        lh  = np.histogram(lbp_sup.flatten(), bins=8, range=(0,1))[0]
        lh  = lh / (lh.sum() + 1e-9)
        le  = -np.sum(lh * np.log(lh+1e-9)) / np.log(8)
        lu  = float(np.sum(lbp_sup > 0.8)) / (ts + 1e-6)

        # ── [78:82] Luminancia zona superior ────────────────
        hl  = np.histogram(lum[:H//3,:], bins=8, range=(0,1))[0]
        hl  = hl / (hl.sum() + 1e-9)
        el  = -np.sum(hl * np.log(hl+1e-9)) / np.log(8)
        pbr = np.sum(lum[:H//3,:] > 0.82) / ts
        pmi = np.sum((lum[:H//3,:]>0.35)&(lum[:H//3,:]<=0.82)) / ts
        pdk = np.sum(lum[:H//3,:] <= 0.35) / ts

        # ── [82:88] HSV global (media + std) ────────────────
        hsv_pil = img.convert('HSV')
        hsv_arr = np.array(hsv_pil, dtype=np.float32)
        hue_g   = hsv_arr[:,:,0] / 255.0   # 0-1
        sat_g   = hsv_arr[:,:,1] / 255.0
        val_g   = hsv_arr[:,:,2] / 255.0
        hue_mean_g = hue_g.mean(); hue_std_g = hue_g.std()
        sat_mean_g = sat_g.mean(); sat_std_g = sat_g.std()
        val_mean_g = val_g.mean(); val_std_g = val_g.std()

        # ── [88:94] HSV zona superior ────────────────────────
        hue_s = hue_g[:H//3,:]
        sat_s = sat_g[:H//3,:]
        val_s = val_g[:H//3,:]
        hue_mean_s = hue_s.mean(); hue_std_s  = hue_s.std()
        sat_mean_s = sat_s.mean(); val_mean_s  = val_s.mean()
        sat_std_s  = sat_s.std();  val_std_s   = val_s.std()

        # ── [94:100] Cues multi-canal zona superior ──────────
        hue_deg_s = hue_s * 360.0
        # Azul cielo (HSV)
        c_az_hsv_s = np.sum((hue_deg_s>180)&(hue_deg_s<270)&
                             (sat_s>0.08)&(val_s>0.12)) / ts
        # Blanco nube
        c_bl_hsv_s = np.sum((sat_s<0.22)&(val_s>0.65)) / ts
        # Gris nube
        c_gr_hsv_s = np.sum((sat_s<0.25)&(val_s>0.30)&(val_s<0.72)) / ts
        # Canal B relativo zona superior
        b_rel_s    = b_s / (r_s + g_s + b_s + 1e-6)
        b_rel_mean = float(b_rel_s.mean())
        # Uniformidad zona superior (inverso de std de luminancia)
        unif_s     = 1.0 - float(lum[:H//3,:].std())
        # Saturación zona superior
        sat_sup    = float(sat_s.mean())

        return np.concatenate([
            hr, hg, hb,                        # 0-47
            media, std,                        # 48-53
            [pct_az, pct_gr, pct_bl,
             pct_ve, pct_os, sat_rgb, contr],  # 54-60
            [paz, pbl, pgr],                   # 61-63
            [gg, gs_g, gmd, gi, gv,
             rtex, suav, rhv],                 # 64-71
            [scg, scs],                        # 72-73  score cielo
            [lm,  ls,  le,  lu],               # 74-77  LBP
            [el,  pbr, pmi, pdk],              # 78-81  lum sup
            [hue_mean_g, hue_std_g,
             sat_mean_g, sat_std_g,
             val_mean_g, val_std_g],           # 82-87  HSV global
            [hue_mean_s, hue_std_s,
             sat_mean_s, val_mean_s,
             sat_std_s,  val_std_s],           # 88-93  HSV sup
            [c_az_hsv_s, c_bl_hsv_s,
             c_gr_hsv_s, b_rel_mean,
             unif_s, sat_sup],                 # 94-99  cues sup
        ])   # total = 100 features
    except Exception:
        return None


def lbp_simple(canal):
    """Local Binary Pattern simplificado (8 vecinos, radio 1)."""
    h, w = canal.shape
    lbp  = np.zeros((h,w), dtype=np.float32)
    c    = canal.astype(np.float32)
    for p, angle in enumerate(np.linspace(0, 2*np.pi, 8, endpoint=False)):
        dx = int(round(np.cos(angle)))
        dy = int(round(np.sin(angle)))
        shifted = np.roll(np.roll(c, dy, axis=0), dx, axis=1)
        lbp += (shifted >= c).astype(np.float32) * (2**p)
    return lbp / 255.0


def _sigmoide(x, centro, escala):
    """Sigmoide suave: ~0 si x<<centro, ~1 si x>>centro."""
    return 1.0 / (1.0 + np.exp(-(x - centro) / (escala + 1e-9)))


def score_clase(c):
    """
    Discriminación de clases de cielo con scores sigmoides suaves.
    Validado experimentalmente sobre imágenes reales del dataset.

    Discriminadores clave:
      Despejado    = mucho azul HSV (>0.35) Y pocas nubes blancas (<0.18)
      Semi-Nublado = azul + nubes simultáneos, O cielo blanco con edificios debajo
      Nublado      = sin azul, uniforme (sin edificios), o gris oscuro
    """
    n = len(c)
    if n < 82:
        return {0: 0.1, 1: 0.5, 2: 1.0, 3: 0.0}

    # ── Features RGB ──────────────────────────────────────────
    pct_az = c[54]; pct_gr = c[55]; pct_bl = c[56]
    pct_ve = c[57]; pct_os = c[58]; sat_rgb = c[59]
    mb     = c[50]; mr = c[48]; mg = c[49]
    paz    = c[61]; pbl = c[62]; pgr = c[63]
    gs     = c[65]; gi = c[67]
    suav   = c[70]
    scg    = c[72]; scs = c[73]
    le     = c[76]; pbr = c[79]; pmi = c[80]

    # ── Features HSV (si n>=100) ──────────────────────────────
    if n >= 100:
        c_az_hsv = c[94]; c_bl_hsv = c[95]; c_gr_hsv = c[96]
        b_rel    = c[97]; unif = c[98]; sat_sup = c[99]
        val_sup  = c[91]
    else:
        c_az_hsv = paz; c_bl_hsv = pbl; c_gr_hsv = pgr
        b_rel = mb / (mr + mg + mb + 1e-6)
        unif = 1.0 - gs; sat_sup = sat_rgb; val_sup = mb

    # ── Derivadas ─────────────────────────────────────────────
    base      = scs * 5.0 + scg * 3.0     # "hay cielo"
    ratio_inf = gi / (gs + 1e-6)          # edificios debajo si > 1.5

    # ── DESPEJADO ─────────────────────────────────────────────
    f_az_alto  = _sigmoide(c_az_hsv, 0.35, 0.12)
    f_bl_bajo  = 1.0 - _sigmoide(c_bl_hsv, 0.18, 0.06)
    f_sat_alto = _sigmoide(sat_sup, 0.25, 0.10)
    sd = (base +
          (f_az_alto * f_bl_bajo) * 18.0 +   # azul puro sin nubes
          f_sat_alto * 4.0 +
          c_az_hsv   * 3.0 +
          b_rel      * 1.5)

    # ── SEMI-NUBLADO ──────────────────────────────────────────
    f_az_pres   = _sigmoide(c_az_hsv, 0.12, 0.06)
    f_bl_pres   = _sigmoide(c_bl_hsv, 0.15, 0.06)
    f_mezcla    = f_az_pres * f_bl_pres        # azul + nubes a la vez
    f_edificios = _sigmoide(ratio_inf, 1.3, 0.4)
    f_brillante = _sigmoide(val_sup, 0.6, 0.1)
    f_sin_az    = 1.0 - _sigmoide(c_az_hsv, 0.10, 0.05)
    f_blanco_ed = (f_sin_az * f_edificios * f_brillante *
                   _sigmoide(c_bl_hsv, 0.4, 0.15))
    ss = (base +
          f_mezcla    * 16.0 +    # azul + nubes
          f_blanco_ed * 14.0 +    # blanco + edificios
          c_bl_hsv    * 2.0 +
          pmi         * 1.5)

    # ── NUBLADO ───────────────────────────────────────────────
    f_uniforme = 1.0 - _sigmoide(ratio_inf, 1.3, 0.4)
    f_oscuro   = 1.0 - _sigmoide(val_sup, 0.5, 0.1)
    f_gris     = _sigmoide(pct_gr, 0.7, 0.15)
    f_sat_bajo = 1.0 - _sigmoide(sat_sup, 0.10, 0.05)
    sn_blanco  = (f_sin_az * f_uniforme *
                  _sigmoide(c_bl_hsv, 0.4, 0.2) * 16.0)
    sn_gris    = (f_oscuro * f_gris + f_sat_bajo * f_gris) * 10.0
    sn = (base +
          max(sn_blanco, sn_gris) +
          (1.0 - c_az_hsv) * 2.0)

    # ── NO RELACIONADO ────────────────────────────────────────
    sr_raw = (pct_ve * 4.0 + pct_os * 3.5 +
              (1.0-scg) * 2.0 + (1.0-scs) * 1.5 + le * 1.0)
    sr = sr_raw * (1.0 - min(scs, 1.0))

    return {0: max(sd, 0.0),
            1: max(ss, 0.0),
            2: max(sn, 0.0),
            3: max(sr, 0.0)}








def asignar_clases(centros, k):
    """
    Asigna clusters a clases de cielo.

    REGLAS:
    - k=3  -> exactamente Despejado / Semi-Nublado / Nublado.
              NUNCA aparece "No relacionado".
    - k=4  -> puede aparecer "No relacionado" SOLO si el cluster
              tiene score_cielo_sup < 0.20 Y (pct_verde + pct_oscuro) > 0.40.
              Si no cumple ambas condiciones, va a la clase de cielo más parecida.
    - k>4  -> los clusters extra siempre van a la clase de cielo más parecida.

    Esto garantiza que un dataset de 900 cielos con K=3
    jamás produce imágenes "No relacionado".
    """
    scores       = [score_clase(c) for c in centros]
    clases_cielo = [0, 1, 2]   # Despejado, Semi-Nublado, Nublado
    asign        = {}
    usados_c     = set()
    usados_cl    = set()

    # ── Paso 1: asignación greedy solo entre clases de cielo ──
    pares = [(scores[cid].get(cls, 0), cid, cls)
             for cid in range(k) for cls in clases_cielo]
    pares.sort(reverse=True)

    for sc, cid, cls in pares:
        if cid not in usados_c and cls not in usados_cl:
            asign[cid] = cls
            usados_c.add(cid); usados_cl.add(cls)

    # ── Paso 2: clusters sin clase → mejor clase cielo disponible ──
    # (puede haber empates si k > 3; siempre cielo, nunca no-relacionado)
    for cid in range(k):
        if cid not in asign:
            best = max(clases_cielo, key=lambda c: scores[cid].get(c, 0))
            asign[cid] = best

    # ── Paso 3: post-proceso "No relacionado" SOLO si k=4 ──────
    # Criterio DOBLE muy estricto:
    #   a) score_cielo_sup < 0.20  (casi nada de cielo arriba)
    #   b) (pct_verde + pct_oscuro) > 0.40  (domina vegetación u oscuro)
    # Si un cluster de 900 cielos llega aquí, ambas condiciones
    # serán falsas y no se reclasificará.
    if k == 4:
        # Encontrar el cluster con menor score de cielo
        scores_cielo = []
        for cid in range(k):
            sc_sup = centros[cid][73] if len(centros[cid]) >= 74 else 1.0
            sc_glb = centros[cid][72] if len(centros[cid]) >= 74 else 1.0
            scores_cielo.append((sc_sup + sc_glb) / 2.0)

        peor_cid = int(np.argmin(scores_cielo))
        c = centros[peor_cid]
        if len(c) >= 74:
            scs = c[73]   # score_cielo_sup
            pve = c[57]   # pct_verde
            pos = c[58]   # pct_oscuro
            # Solo reclasifica si cumple AMBAS condiciones
            if scs < 0.20 and (pve + pos) > 0.40:
                asign[peor_cid] = 3

    return asign


# ═══════════════════════════════════════════════════════════════
# K-MEANS MANUAL (mejorado con K-Means++ init)
# ═══════════════════════════════════════════════════════════════

def kmeans_pp_init(X, k):
    """K-Means++ initialization para mejor convergencia"""
    n = len(X)
    idx = np.random.randint(n)
    centros = [X[idx].copy()]
    for _ in range(k-1):
        dists = np.array([min(np.sum((x-c)**2) for c in centros) for x in X])
        probs = dists / dists.sum()
        idx   = np.random.choice(n, p=probs)
        centros.append(X[idx].copy())
    return np.array(centros)

def kmeans_manual(X, k=4, max_iter=150, n_init=10):
    best_i, best_c, best_e = np.inf, None, None
    for trial in range(n_init):
        # Alternar entre init aleatoria y K-Means++
        if trial % 2 == 0:
            C = kmeans_pp_init(X, k)
        else:
            C = X[np.random.choice(len(X), k, replace=False)].copy()

        for _ in range(max_iter):
            D = np.linalg.norm(X[:,None,:] - C[None,:,:], axis=2)
            E = D.argmin(axis=1)
            N = np.array([X[E==i].mean(axis=0) if np.any(E==i) else C[i]
                          for i in range(k)])
            if np.allclose(C, N, atol=1e-6): break
            C = N
        inr = np.sum((X - C[E])**2)
        if inr < best_i: best_i, best_c, best_e = inr, C.copy(), E.copy()
    return best_e, best_c, best_i


# ═══════════════════════════════════════════════════════════════
# CLASIFICADORES
# ═══════════════════════════════════════════════════════════════

def calcular_params(datos):
    m  = datos.mean(axis=0)
    S  = np.cov(datos.T) + 1e-6 * np.eye(datos.shape[1])
    ic = np.linalg.inv(S)
    dc = np.linalg.det(S)
    return m, ic, dc

def clasificar_punto(vec, medias, inv_covs, det_covs, metodo, nc):
    vals = np.zeros(nc)
    for i in range(nc):
        d = vec - medias[i]
        if metodo == 0:
            vals[i] = np.linalg.norm(d)
        elif metodo == 1:
            vals[i] = np.sqrt(max(0.0, d @ inv_covs[i] @ d))
        else:
            # Bayes con LOG-verosimilitud (evita underflow con 100 features).
            # log P = -0.5*(d·Σ⁻¹·d) - 0.5*log(det) - (k/2)*log(2π)
            mahal = float(d @ inv_covs[i] @ d)
            logdet = np.log(abs(det_covs[i]) + 1e-300)
            vals[i] = -0.5*mahal - 0.5*logdet - 0.5*len(d)*np.log(2*np.pi)

    if metodo == 2:
        # vals son log-verosimilitudes. Confianza = softmax estable.
        cl = int(np.argmax(vals))
        vmax = vals.max()
        exp_vals = np.exp(vals - vmax)   # estabilización numérica
        tot = exp_vals.sum() + 1e-300
        conf = float(exp_vals[cl] / tot * 100)
        if not np.isfinite(conf):
            conf = 100.0
    else:
        cl  = int(np.argmin(vals))
        mx  = vals.max(); mnv = vals.min()
        conf = (1-(vals[cl]-mnv)/(mx-mnv+1e-9))*100 if mx>mnv else 100.0
        if not np.isfinite(conf):
            conf = 100.0
    return int(cl), float(conf)


# ═══════════════════════════════════════════════════════════════
# SIMILITUD ENTRE IMÁGENES Y DETECCIÓN DE OUTLIERS
# ═══════════════════════════════════════════════════════════════

def similitud_coseno(v1, v2):
    """Similitud coseno entre dos vectores de features. Rango 0-100%."""
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = np.dot(v1, v2) / (n1 * n2)
    return float(np.clip(cos, 0, 1) * 100)


def similitud_euclidiana(v1, v2, escala=None):
    """Similitud por distancia euclidiana normalizada. Rango 0-100%."""
    d = np.linalg.norm(v1 - v2)
    if escala is None:
        escala = np.sqrt(len(v1))
    sim = np.exp(-d / (escala + 1e-9))
    return float(sim * 100)


def similitud_a_clase(feat, medias_clase, idx_clase):
    """
    Similitud de una imagen a la media de su clase asignada.
    Combina coseno + euclidiana. Rango 0-100%.
    """
    if idx_clase >= len(medias_clase):
        return 0.0
    media = medias_clase[idx_clase]
    s_cos = similitud_coseno(feat, media)
    s_euc = similitud_euclidiana(feat, media)
    return (s_cos * 0.5 + s_euc * 0.5)


def detectar_outlier(feat, medias_clase, inv_covs, det_covs,
                     umbral_mahal=None, score_cielo_min=0.25):
    """
    Determina si una imagen es "No relacionado" (no es cielo).

    Criterios:
      1. score_cielo_global muy bajo (< score_cielo_min)
      2. Distancia de Mahalanobis a TODAS las clases por encima del umbral
      3. Presencia alta de verde/oscuro

    Retorna: (es_outlier(bool), razon(str), dist_min(float))
    """
    if len(feat) < 74:
        return False, '', 0.0

    scg = feat[72]   # score_cielo_global
    scs = feat[73]   # score_cielo_sup

    # Verde / oscuro
    pct_ve = feat[57] if len(feat) > 57 else 0.0
    pct_os = feat[58] if len(feat) > 58 else 0.0

    # Distancia de Mahalanobis mínima a las clases de cielo
    dists = []
    for i in range(len(medias_clase)):
        d = feat - medias_clase[i]
        try:
            dm = np.sqrt(max(0.0, d @ inv_covs[i] @ d))
        except Exception:
            dm = np.linalg.norm(d)
        dists.append(dm)
    dist_min = float(min(dists)) if dists else 0.0

    # Criterio 1: casi sin cielo
    if scg < score_cielo_min and scs < score_cielo_min + 0.05:
        if (pct_ve + pct_os) > 0.30:
            return True, 'sin_cielo_verde_oscuro', dist_min

    # Criterio 2: muy lejos de todas las clases (outlier estadístico)
    if umbral_mahal is not None and dist_min > umbral_mahal:
        return True, 'outlier_estadistico', dist_min

    return False, '', dist_min


def clasificar_con_outlier(feat, medias_clase, inv_covs, det_covs,
                           metodo, nc, umbral_mahal=None):
    """
    Clasifica una imagen. Si es outlier (no-cielo), retorna clase 3
    (No relacionado). Si no, clasifica normalmente entre clases de cielo.

    Retorna: (clase(int), confianza(float), es_outlier(bool))
    """
    es_out, razon, dist = detectar_outlier(
        feat, medias_clase, inv_covs, det_covs,
        umbral_mahal=umbral_mahal)
    if es_out:
        return 3, 95.0, True
    cl, conf = clasificar_punto(feat, medias_clase, inv_covs,
                                det_covs, metodo, nc)
    return cl, conf, False


def evaluar_clasificador(X, y, metodo, nc, modo, k_fold=5):
    n = len(X)

    def run(trX, trY, teX, teY):
        ms, ics, dcs = [], [], []
        for i in range(nc):
            idx = trY == i
            if idx.sum() < 2:
                return 0.0, np.zeros((nc,nc), dtype=int)
            m, ic, dc = calcular_params(trX[idx])
            ms.append(m); ics.append(ic); dcs.append(dc)
        mc = np.zeros((nc,nc), dtype=int); hits = 0
        for j in range(len(teY)):
            pred,_ = clasificar_punto(teX[j], ms, ics, dcs, metodo, nc)
            mc[teY[j], pred] += 1
            if pred == teY[j]: hits += 1
        return hits/len(teY)*100, mc

    if modo == 0:
        idx = np.random.permutation(n); ntr = int(0.7*n)
        return run(X[idx[:ntr]], y[idx[:ntr]], X[idx[ntr:]], y[idx[ntr:]])
    elif modo == 1:
        idx = np.random.permutation(n); Xs,ys = X[idx], y[idx]
        fsz = n//k_fold; accs=[]; mct=np.zeros((nc,nc),dtype=int)
        for f in range(k_fold):
            ti = list(range(f*fsz,(f+1)*fsz))
            ri = [i for i in range(n) if i not in ti]
            a, mc = run(Xs[ri],ys[ri],Xs[ti],ys[ti])
            accs.append(a); mct += mc
        return np.mean(accs), mct
    else:
        ms,ics,dcs=[],[],[]
        for i in range(nc):
            idx = y==i
            if idx.sum()<2:
                ms.append(X.mean(axis=0)); ics.append(np.eye(X.shape[1])); dcs.append(1.0); continue
            m,ic,dc=calcular_params(X[idx]); ms.append(m); ics.append(ic); dcs.append(dc)
        mc=np.zeros((nc,nc),dtype=int); hits=0
        for j in range(n):
            pred,_=clasificar_punto(X[j],ms,ics,dcs,metodo,nc)
            mc[y[j],pred]+=1
            if pred==y[j]: hits+=1
        return hits/n*100, mc


# ═══════════════════════════════════════════════════════════════
# MÉTRICAS AVANZADAS DE CLUSTERING Y CLASIFICACIÓN
# ═══════════════════════════════════════════════════════════════

def silhouette_score_manual(X, etiquetas):
    """
    Coeficiente de silueta promedio. Rango [-1, 1].
    Mide qué tan bien separados están los clusters.
    >0.5 bueno, 0.25-0.5 razonable, <0.25 débil.
    """
    n = len(X)
    if n > 600:   # submuestrear para eficiencia
        idx = np.random.choice(n, 600, replace=False)
        X = X[idx]; etiquetas = etiquetas[idx]; n = 600
    labels = np.unique(etiquetas)
    if len(labels) < 2:
        return 0.0
    sils = []
    # Precalcular distancias
    D = np.linalg.norm(X[:,None,:] - X[None,:,:], axis=2)
    for i in range(n):
        propia = etiquetas == etiquetas[i]
        propia[i] = False
        if propia.sum() == 0:
            sils.append(0.0); continue
        a = D[i, propia].mean()
        b = np.inf
        for lbl in labels:
            if lbl == etiquetas[i]: continue
            otros = etiquetas == lbl
            if otros.sum() > 0:
                b = min(b, D[i, otros].mean())
        s = (b - a) / (max(a, b) + 1e-9) if b < np.inf else 0.0
        sils.append(s)
    return float(np.mean(sils))


def davies_bouldin_manual(X, etiquetas, centros):
    """
    Índice Davies-Bouldin. Menor es mejor (clusters compactos y separados).
    """
    k = len(centros)
    if k < 2:
        return 0.0
    # Dispersión intra-cluster
    disp = np.zeros(k)
    for i in range(k):
        pts = X[etiquetas == i]
        if len(pts) > 0:
            disp[i] = np.mean(np.linalg.norm(pts - centros[i], axis=1))
    db = 0.0
    for i in range(k):
        max_ratio = 0.0
        for j in range(k):
            if i == j: continue
            sep = np.linalg.norm(centros[i] - centros[j]) + 1e-9
            max_ratio = max(max_ratio, (disp[i] + disp[j]) / sep)
        db += max_ratio
    return float(db / k)


def calinski_harabasz_manual(X, etiquetas, centros):
    """
    Índice Calinski-Harabasz (varianza inter/intra). Mayor es mejor.
    """
    n = len(X); k = len(centros)
    if k < 2 or n <= k:
        return 0.0
    centro_global = X.mean(axis=0)
    # Between-cluster
    bc = 0.0
    for i in range(k):
        ni = np.sum(etiquetas == i)
        bc += ni * np.sum((centros[i] - centro_global)**2)
    # Within-cluster
    wc = 0.0
    for i in range(k):
        pts = X[etiquetas == i]
        if len(pts) > 0:
            wc += np.sum((pts - centros[i])**2)
    if wc < 1e-9:
        return 0.0
    return float((bc / (k-1)) / (wc / (n-k)))


def metricas_por_clase(mc):
    """
    Calcula precision, recall, F1 por clase desde una matriz de confusión.
    Retorna dict {clase: {precision, recall, f1}}
    """
    nc = mc.shape[0]
    resultado = {}
    for i in range(nc):
        tp = mc[i, i]
        fp = mc[:, i].sum() - tp     # predichos como i pero no son i
        fn = mc[i, :].sum() - tp     # son i pero predichos como otro
        prec = tp / (tp + fp) if (tp+fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp+fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        resultado[i] = {'precision': prec*100,
                        'recall': rec*100,
                        'f1': f1*100,
                        'soporte': int(mc[i, :].sum())}
    return resultado


def correlacion_features(X, indices=None):
    """Matriz de correlación de Pearson entre features seleccionadas."""
    if indices is not None:
        Xs = X[:, indices]
    else:
        Xs = X
    # Estandarizar
    mu = Xs.mean(axis=0); sd = Xs.std(axis=0) + 1e-9
    Xn = (Xs - mu) / sd
    n = len(Xn)
    corr = (Xn.T @ Xn) / n
    return np.clip(corr, -1, 1)


def importancia_features(X, etiquetas, nombres_feat=None):
    """
    Importancia de cada feature según el ratio de varianza
    inter-clase / intra-clase (F-statistic estilo ANOVA).
    Mayor = mejor discrimina entre clases.
    """
    labels = np.unique(etiquetas)
    n, d = X.shape
    media_global = X.mean(axis=0)
    f_scores = np.zeros(d)
    for j in range(d):
        var_inter = 0.0; var_intra = 0.0
        for lbl in labels:
            pts = X[etiquetas == lbl, j]
            ni = len(pts)
            if ni < 2: continue
            var_inter += ni * (pts.mean() - media_global[j])**2
            var_intra += np.sum((pts - pts.mean())**2)
        k = len(labels)
        if var_intra > 1e-9 and k > 1:
            f_scores[j] = (var_inter/(k-1)) / (var_intra/(n-k))
    return f_scores


def varianza_explicada_pca(X, n_comp=10):
    """Retorna el % de varianza explicada por las primeras n_comp componentes."""
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc.T)
    eigvals = np.linalg.eigvalsh(cov)[::-1]   # descendente
    eigvals = np.maximum(eigvals, 0)
    total = eigvals.sum() + 1e-9
    var_exp = eigvals[:n_comp] / total * 100
    var_acum = np.cumsum(var_exp)
    return var_exp, var_acum




class Worker(QThread):
    prog = pyqtSignal(int, str)
    done = pyqtSignal(object)
    err  = pyqtSignal(str)

    def __init__(self, carpeta, k):
        super().__init__()
        self.carpeta = carpeta; self.k = k

    def run(self):
        try:
            exts  = ['*.jpg','*.jpeg','*.png','*.bmp','*.webp',
                     '*.JPG','*.JPEG','*.PNG','*.BMP','*.WEBP']
            files = list(set(f for e in exts
                             for f in glob.glob(os.path.join(self.carpeta,e))))
            if not files: self.err.emit('No se encontraron imágenes.'); return

            self.prog.emit(0, f'Extrayendo features de {len(files)} imágenes...')
            feats, valids = [], []
            for i, p in enumerate(files):
                f = extraer_features(p)
                if f is not None: feats.append(f); valids.append(p)
                self.prog.emit(int((i+1)/len(files)*55),
                               f'Features {i+1}/{len(files)}')

            if len(feats) < self.k:
                self.err.emit(f'Solo {len(feats)} imágenes válidas.'); return

            X = np.array(feats)
            self.prog.emit(58, 'Ejecutando K-Means++...')
            etqs, centros, inercia = kmeans_manual(
                X, k=self.k, max_iter=150, n_init=10)

            self.prog.emit(90, 'Asignando clases...')
            asign = asignar_clases(centros, self.k)

            self.prog.emit(97, 'Listo')
            self.done.emit({
                'X':X, 'etqs':etqs, 'centros':centros, 'inercia':inercia,
                'archivos':valids, 'k':self.k, 'asign':asign,
                'n_found':len(files), 'n_ok':len(feats),
            })
        except Exception as e:
            self.err.emit(str(e))


# ═══════════════════════════════════════════════════════════════
# WIDGETS
# ═══════════════════════════════════════════════════════════════

class SectionLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            f"color:{C_INK_DIM};font-family:'{FONT_MONO}';font-size:9px;"
            f"font-weight:bold;letter-spacing:2px;padding:8px 0 3px 0;border:none;")

class Divider(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setStyleSheet(f'background:{C_BORDER};max-height:1px;border:none;')

class StatCard(QFrame):
    def __init__(self, label, value, color=C_SKY, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{background:{C_CARD};border:1px solid {C_BORDER};"
            f"border-top:3px solid {color};border-radius:8px;}}")
        lay = QVBoxLayout(self); lay.setContentsMargins(12,8,12,8); lay.setSpacing(2)
        self.lv = QLabel(str(value))
        self.lv.setStyleSheet(
            f"color:{color};font-size:20px;font-weight:bold;"
            f"font-family:'{FONT_MONO}';border:none;")
        self.lv.setAlignment(Qt.AlignCenter)
        ll = QLabel(label)
        ll.setStyleSheet(
            f"color:{C_INK_DIM};font-size:8px;letter-spacing:1px;"
            f"font-family:'{FONT_MONO}';border:none;")
        ll.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.lv); lay.addWidget(ll)

    def set_value(self, v): self.lv.setText(str(v))

class ImageCard(QFrame):
    clicked = pyqtSignal(str, int)

    def __init__(self, img_path, clase_idx, nombre_clase, parent=None):
        super().__init__(parent)
        self.img_path = img_path; self.clase_idx = clase_idx
        col = C_CLASSES.get(clase_idx, C_INK_DIM)
        self.setStyleSheet(
            f"QFrame{{background:{C_CARD};border:1px solid {C_BORDER};border-radius:8px;}}"
            f"QFrame:hover{{border:1px solid {col};background:{C_PANEL};}}")
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setFixedSize(118, 108)
        lay = QVBoxLayout(self); lay.setContentsMargins(4,4,4,4); lay.setSpacing(3)
        lbl_img = QLabel(); lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setStyleSheet('border:none;'); lbl_img.setFixedHeight(70)
        try:
            img = Image.open(img_path).convert('RGB').resize((106,66))
            arr = np.array(img)
            qi  = QImage(arr.data, 106, 66, 318, QImage.Format_RGB888)
            lbl_img.setPixmap(QPixmap.fromImage(qi).scaled(
                106, 66, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except: pass
        lay.addWidget(lbl_img)
        badge = QLabel(nombre_clase[:13])
        badge.setStyleSheet(
            f"color:white;background:{col}22;border:1px solid {col};"
            f"border-radius:9px;padding:2px 6px;font-size:8px;"
            f"font-weight:bold;font-family:'{FONT_MONO}';")
        badge.setAlignment(Qt.AlignCenter); lay.addWidget(badge)
        fn = QLabel(os.path.basename(img_path)[:13]+'…')
        fn.setStyleSheet(f'color:{C_INK_FAINT};font-size:7px;border:none;')
        fn.setAlignment(Qt.AlignCenter); lay.addWidget(fn)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.img_path, self.clase_idx)

class ImageViewer(QDialog):
    def __init__(self, img_path, clase_idx, nombre_clase,
                 medias=None, inv_covs=None, det_covs=None,
                 nombres_clase=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(img_path))
        self.setMinimumSize(960, 600)
        self.setStyleSheet(STYLE_BASE + f'QDialog{{background:{C_BG};}}')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20,20,20,20); lay.setSpacing(20)

        # Imagen grande
        lbl_img = QLabel(); lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setStyleSheet(
            f'border:1px solid {C_BORDER};border-radius:8px;background:{C_SURFACE};')
        try:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail((640,480), Image.LANCZOS)
            arr = np.array(img)
            qi  = QImage(arr.data, img.width, img.height,
                         img.width*3, QImage.Format_RGB888)
            lbl_img.setPixmap(QPixmap.fromImage(qi))
        except: lbl_img.setText('Error cargando imagen')
        lbl_img.setFixedSize(650, 480); lay.addWidget(lbl_img)

        # Panel info
        info = QFrame()
        info.setStyleSheet(
            f'background:{C_PANEL};border-radius:10px;border:1px solid {C_BORDER};')
        info.setFixedWidth(255)
        il = QVBoxLayout(info); il.setContentsMargins(16,16,16,16); il.setSpacing(8)

        t = QLabel('ANÁLISIS')
        t.setStyleSheet(
            f"color:{C_INK_DIM};font-family:'{FONT_MONO}';font-size:9px;"
            f"letter-spacing:2px;border:none;")
        il.addWidget(t)

        col = C_CLASSES.get(clase_idx, C_INK_DIM)
        lc  = QLabel(nombre_clase)
        lc.setStyleSheet(
            f"color:{col};font-size:18px;font-weight:bold;"
            f"font-family:'{FONT_MONO}';border:none;padding:4px 0;")
        il.addWidget(lc); il.addWidget(Divider())

        fn_l = QLabel('ARCHIVO')
        fn_l.setStyleSheet(
            f"color:{C_INK_DIM};font-size:9px;letter-spacing:1px;border:none;")
        il.addWidget(fn_l)
        fn_v = QLabel(os.path.basename(img_path))
        fn_v.setStyleSheet(f"color:{C_INK};font-size:10px;border:none;")
        fn_v.setWordWrap(True); il.addWidget(fn_v); il.addWidget(Divider())

        # Segmentación mini
        try:
            arr_s, mask, overlay, score, grad = segmentar_cielo(img_path, (160,110))
            if arr_s is not None:
                # Mostrar overlay con línea de contorno
                fig_mini, ax_mini = plt.subplots(figsize=(2.5,1.8))
                fig_mini.patch.set_facecolor('none')
                ax_mini.imshow(overlay)
                if mask is not None:
                    ax_mini.contour(mask.astype(float), levels=[0.5],
                                    colors=['#00e5ff'], linewidths=[1.5])
                ax_mini.axis('off')
                fig_mini.tight_layout(pad=0)
                # Convertir figura a QPixmap
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                canvas_mini = FigureCanvasAgg(fig_mini)
                canvas_mini.draw()
                buf = canvas_mini.buffer_rgba()
                mini_arr = np.asarray(buf)
                plt.close(fig_mini)
                h_m, w_m = mini_arr.shape[:2]
                qi2 = QImage(mini_arr.data, w_m, h_m,
                             w_m*4, QImage.Format_RGBA8888)
                ls = QLabel()
                ls.setPixmap(QPixmap.fromImage(qi2).scaled(
                    160, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                ls.setStyleSheet(
                    f'border:1px solid {C_BORDER};border-radius:4px;')
                il.addWidget(ls)
                pct = float(mask.mean())*100
                sp  = QLabel(f'Cielo: {pct:.1f}%')
                sp.setStyleSheet(
                    f"color:{C_SKY};font-size:10px;font-weight:bold;border:none;")
                il.addWidget(sp)
        except: pass

        # Clasificación por los 3 métodos
        if medias is not None:
            il.addWidget(Divider())
            mc_l = QLabel('CLASIFICACIÓN')
            mc_l.setStyleSheet(
                f"color:{C_INK_DIM};font-size:9px;letter-spacing:1px;border:none;")
            il.addWidget(mc_l)
            feat = extraer_features(img_path)
            if feat is not None:
                nc = len(medias)
                for mi, mn in enumerate(['Euclid.','Mahal.','Bayes']):
                    try:
                        cl, conf = clasificar_punto(
                            feat, medias, inv_covs, det_covs, mi, nc)
                        nc_name = nombres_clase.get(cl, f'C{cl}')
                        col2    = C_CLASSES.get(cl, C_INK_DIM)
                        rw = QWidget(); rw.setStyleSheet('background:transparent;')
                        rl = QHBoxLayout(rw); rl.setContentsMargins(0,0,0,0)
                        lm = QLabel(mn)
                        lm.setStyleSheet(
                            f"color:{C_INK_DIM};font-size:9px;border:none;min-width:42px;")
                        lv = QLabel(nc_name)
                        lv.setStyleSheet(
                            f"color:{col2};font-size:10px;font-weight:bold;border:none;")
                        lc2 = QLabel(f'{conf:.0f}%')
                        lc2.setStyleSheet(
                            f"color:{C_INK_FAINT};font-size:9px;border:none;")
                        rl.addWidget(lm); rl.addWidget(lv); rl.addWidget(lc2)
                        il.addWidget(rw)
                    except: pass

        il.addStretch()
        bc = QPushButton('Cerrar')
        bc.setStyleSheet(style_btn(C_BORDER, C_INK))
        bc.setStyleSheet(bc.styleSheet()+f'QPushButton{{border:1px solid {C_BORDER_LT};}}')
        bc.clicked.connect(self.close); il.addWidget(bc)
        lay.addWidget(info)


# ═══════════════════════════════════════════════════════════════
# VENTANA PRINCIPAL
# ═══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sky Classifier')
        self.setMinimumSize(1440, 860)
        self.setStyleSheet(STYLE_BASE + f'QMainWindow{{background:{C_BG};}}')

        self.resultado     = None
        self.archivos      = None
        self.etqs_raw      = None
        self.etqs_map      = None
        self.nombres_clase = {}
        self.asignacion    = {}
        self.accs_mat      = None
        self.mcs_mat       = None
        self.medias_clf    = None
        self.inv_covs_clf  = None
        self.det_covs_clf  = None
        self.umbral_mahal  = None
        self.dist_media_dataset = None
        self._build()

    # ── BUILD ─────────────────────────────────────────────────

    def _build(self):
        root = QWidget(); self.setCentralWidget(root)
        root.setStyleSheet(f'background:{C_BG};')
        main = QHBoxLayout(root)
        main.setSpacing(0); main.setContentsMargins(0,0,0,0)
        main.addWidget(self._build_sidebar())
        main.addWidget(self._build_content(), 1)

    def _build_sidebar(self):
        sbf = QFrame(); sbf.setFixedWidth(278)
        sbf.setStyleSheet(
            f'QFrame{{background:{C_SURFACE};border-right:1px solid {C_BORDER};}}')
        sb = QVBoxLayout(sbf); sb.setContentsMargins(0,0,0,0); sb.setSpacing(0)

        # Header
        hdr = QFrame()
        hdr.setStyleSheet(
            f'background:{C_PANEL};border-bottom:1px solid {C_BORDER};'
            f'border-right:none;')
        hdr.setFixedHeight(70)
        hl = QVBoxLayout(hdr); hl.setContentsMargins(18,12,18,12)
        logo = QLabel('SKY CLASSIFIER')
        logo.setStyleSheet(
            f"color:{C_INK};font-family:'{FONT_MONO}';font-size:13px;"
            f"font-weight:bold;letter-spacing:3px;border:none;")
        sub = QLabel('K-Means++ · LBP · Textura · Estadístico')
        sub.setStyleSheet(
            f"color:{C_INK_FAINT};font-size:9px;border:none;letter-spacing:1px;")
        hl.addWidget(logo); hl.addWidget(sub); sb.addWidget(hdr)

        # Controles en scroll
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f'background:{C_SURFACE};border:none;')
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cont = QWidget(); cont.setStyleSheet(f'background:{C_SURFACE};')
        cl = QVBoxLayout(cont); cl.setContentsMargins(16,10,16,14); cl.setSpacing(4)

        def isp():
            return (f'QSpinBox{{{style_input()}}}'
                    f'QSpinBox::up-button,QSpinBox::down-button{{width:0;}}')

        # Dataset
        cl.addWidget(SectionLabel('DATASET'))
        self.lbl_carp = QLabel('Sin seleccionar')
        self.lbl_carp.setStyleSheet(
            f"color:{C_INK_DIM};font-size:9px;padding:8px 10px;"
            f"background:{C_CARD};border:1px solid {C_BORDER};border-radius:6px;")
        self.lbl_carp.setWordWrap(True); cl.addWidget(self.lbl_carp)
        bcar = QPushButton('Seleccionar Carpeta')
        bcar.setStyleSheet(style_btn(C_ACCENT))
        bcar.clicked.connect(self.sel_carpeta); cl.addWidget(bcar)

        # K
        cl.addWidget(Divider()); cl.addWidget(SectionLabel('CLUSTERS (K)'))
        self.spin_k = QSpinBox(); self.spin_k.setRange(2,8); self.spin_k.setValue(3)
        self.spin_k.setStyleSheet(isp())
        self.spin_k.valueChanged.connect(self._sync_clases); cl.addWidget(self.spin_k)
        lbl_k_tip = QLabel('K=3 → Despejado / Semi-Nublado / Nublado')
        lbl_k_tip.setStyleSheet(f'color:{C_INK_FAINT};font-size:9px;border:none;')
        cl.addWidget(lbl_k_tip)

        # Nombres de clase
        cl.addWidget(Divider()); cl.addWidget(SectionLabel('NOMBRES DE CLASES'))
        self.cont_clases = QWidget(); self.cont_clases.setStyleSheet('background:transparent;')
        self.lay_clases  = QVBoxLayout(self.cont_clases)
        self.lay_clases.setSpacing(3); self.lay_clases.setContentsMargins(0,0,0,0)
        cl.addWidget(self.cont_clases); self.entradas_clase = []; self._sync_clases(4)

        cl.addWidget(Divider())
        bas = QPushButton('Corregir Asignación')
        bas.setStyleSheet(style_btn(C_PANEL, C_INK))
        bas.setStyleSheet(bas.styleSheet()+f'QPushButton{{border:1px solid {C_BORDER_LT};}}')
        bas.clicked.connect(self.abrir_asig); cl.addWidget(bas)

        # Folds
        cl.addWidget(Divider()); cl.addWidget(SectionLabel('FOLDS CROSS-VAL'))
        self.spin_folds = QSpinBox(); self.spin_folds.setRange(2,10); self.spin_folds.setValue(5)
        self.spin_folds.setStyleSheet(isp()); cl.addWidget(self.spin_folds)

        cl.addWidget(Divider())
        brun = QPushButton('▶  EJECUTAR')
        brun.setStyleSheet(style_btn('#1a7a4a','white','11px 16px'))
        brun.clicked.connect(self.ejecutar); cl.addWidget(brun)

        cl.addWidget(Divider())
        btxt = QPushButton('Exportar CSV (Excel)')
        btxt.setStyleSheet(style_btn(C_PANEL,C_INK))
        btxt.setStyleSheet(btxt.styleSheet()+f'QPushButton{{border:1px solid {C_BORDER};}}')
        btxt.clicked.connect(self.guardar_txt); cl.addWidget(btxt)
        bpdf = QPushButton('Exportar PDF')
        bpdf.setStyleSheet(style_btn(C_PANEL,C_INK))
        bpdf.setStyleSheet(bpdf.styleSheet()+f'QPushButton{{border:1px solid {C_BORDER};}}')
        bpdf.clicked.connect(self.guardar_pdf); cl.addWidget(bpdf)
        bhtml = QPushButton('Reporte HTML')
        bhtml.setStyleSheet(style_btn(C_PANEL,C_INK))
        bhtml.setStyleSheet(bhtml.styleSheet()+f'QPushButton{{border:1px solid {C_BORDER};}}')
        bhtml.clicked.connect(self.guardar_html); cl.addWidget(bhtml)
        cl.addStretch(); scroll.setWidget(cont); sb.addWidget(scroll,1)

        # Footer progress
        footer = QFrame()
        footer.setStyleSheet(
            f'background:{C_PANEL};border-top:1px solid {C_BORDER};border-right:none;')
        footer.setFixedHeight(62)
        fl = QVBoxLayout(footer); fl.setContentsMargins(16,8,16,8); fl.setSpacing(4)
        self.progress = QProgressBar()
        self.progress.setRange(0,100); self.progress.setValue(0)
        self.progress.setFixedHeight(4); self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            f"QProgressBar{{background:{C_BORDER};border:none;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {C_ACCENT},stop:1 {C_SKY});border-radius:2px;}}")
        self.lbl_status = QLabel('Ready')
        self.lbl_status.setStyleSheet(
            f"color:{C_INK_DIM};font-size:9px;font-family:'{FONT_MONO}';border:none;")
        fl.addWidget(self.progress); fl.addWidget(self.lbl_status); sb.addWidget(footer)
        return sbf

    def _build_content(self):
        cont = QWidget(); cont.setStyleSheet(f'background:{C_BG};')
        lay  = QVBoxLayout(cont); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        # Stats bar
        sbar = QFrame()
        sbar.setStyleSheet(
            f'background:{C_SURFACE};border-bottom:1px solid {C_BORDER};')
        sbar.setFixedHeight(76)
        sl = QHBoxLayout(sbar); sl.setContentsMargins(20,10,20,10); sl.setSpacing(10)
        self.s_imgs = StatCard('IMÁGENES','—',C_SKY)
        self.s_k    = StatCard('CLUSTERS','—',C_ACCENT)
        self.s_iner = StatCard('INERCIA','—',C_SUN)
        self.s_best = StatCard('MEJOR PREC.','—','#66bb6a')
        for s in [self.s_imgs,self.s_k,self.s_iner,self.s_best]: sl.addWidget(s)
        lay.addWidget(sbar)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane{{background:{C_BG};border:none;
                border-top:1px solid {C_BORDER};}}
            QTabBar::tab{{background:{C_SURFACE};color:{C_INK_DIM};
                padding:12px 20px;font-size:10px;font-family:'{FONT_MONO}';
                letter-spacing:1px;border:none;
                border-bottom:2px solid transparent;margin-right:1px;}}
            QTabBar::tab:selected{{color:{C_INK};
                border-bottom:2px solid {C_ACCENT};background:{C_BG};}}
            QTabBar::tab:hover{{color:{C_INK};background:{C_PANEL};}}
        """)

        self.tab_dash = self._mt()
        self.tab_cl  = self._mt(); self.tab_sg = self._mt()
        self.tab_ev  = self._mt(); self.tab_anal = self._mt()
        self.tab_stats = self._mt()
        self.tab_gal= self._mt(); self.tab_log = self._mt()
        self.tabs.addTab(self.tab_dash, 'DASHBOARD')
        self.tabs.addTab(self.tab_cl, 'CLUSTERS 3D')
        self.tabs.addTab(self.tab_sg, 'SEGMENTACIÓN')
        self.tabs.addTab(self.tab_ev, 'EVALUACIÓN')
        self.tabs.addTab(self.tab_anal, 'ANÁLISIS')
        self.tabs.addTab(self.tab_stats, 'ESTADÍSTICAS')
        self.tabs.addTab(self.tab_gal,'GALERÍA')
        self.tabs.addTab(self.tab_log,'LOG')

        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet(
            f"QTextEdit{{background:{C_SURFACE};color:#4ade80;"
            f"font-family:'{FONT_MONO}';font-size:10px;border:none;padding:14px;}}")
        self.tab_log.layout().addWidget(self.txt_log)
        lay.addWidget(self.tabs,1); return cont

    def _mt(self):
        w = QWidget(); w.setStyleSheet(f'background:{C_BG};')
        l = QVBoxLayout(w); l.setContentsMargins(18,18,18,18); l.setSpacing(10)
        return w

    def _sync_clases(self, k):
        for e in self.entradas_clase: e.setParent(None)
        self.entradas_clase = []
        while self.lay_clases.count():
            item = self.lay_clases.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for i in range(k):
            row = QWidget(); row.setStyleSheet('background:transparent;')
            rl  = QHBoxLayout(row); rl.setContentsMargins(0,2,0,2); rl.setSpacing(6)
            dot = QLabel('●')
            dot.setStyleSheet(
                f"color:{C_CLASSES.get(i,'#888')};font-size:10px;border:none;")
            rl.addWidget(dot)
            e = QLineEdit()
            e.setText(NOMBRES_DEFAULT[i] if i < len(NOMBRES_DEFAULT) else f'Clase {i}')
            e.setStyleSheet(style_input()+'QLineEdit{padding:5px 8px;}')
            rl.addWidget(e,1); self.entradas_clase.append(e); self.lay_clases.addWidget(row)

    def log(self, msg): self.txt_log.append(msg)

    def _limpiar(self, tab):
        lay = tab.layout()
        while lay.count():
            item = lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    # ── ACCIONES ──────────────────────────────────────────────

    def sel_carpeta(self):
        carp = QFileDialog.getExistingDirectory(self,'Selecciona carpeta')
        if carp:
            self.carpeta = carp; self.lbl_carp.setText(os.path.basename(carp))
            exts  = ['*.jpg','*.jpeg','*.png','*.bmp','*.webp',
                     '*.JPG','*.JPEG','*.PNG','*.BMP','*.WEBP']
            total = len(list(set(
                f for e in exts for f in glob.glob(os.path.join(carp,e)))))
            self.log(f'📁 {os.path.basename(carp)} — {total} imágenes')
            self.s_imgs.set_value(total)

    def ejecutar(self):
        if not hasattr(self,'carpeta'):
            QMessageBox.warning(self,'Aviso','Selecciona una carpeta.'); return
        k = self.spin_k.value()
        self.nombres_clase = {i:self.entradas_clase[i].text() for i in range(k)}
        self.progress.setValue(0); self.lbl_status.setText('Processing...')
        self.log(f'\n▶  K={k}')
        self.worker = Worker(self.carpeta, k)
        self.worker.prog.connect(lambda v,m:(
            self.progress.setValue(v), self.lbl_status.setText(m)))
        self.worker.done.connect(self._on_done)
        self.worker.err.connect(lambda e:(
            QMessageBox.critical(self,'Error',e), self.log(f'❌ {e}')))
        self.worker.start()

    def _on_done(self, res):
        self.resultado = res
        self.archivos  = res['archivos']
        self.etqs_raw  = res['etqs']
        k = res['k']
        auto_asig = res['asign']

        self.asignacion = {
            cid: self.nombres_clase.get(cls_id,
                  NOMBRES_DEFAULT[cls_id] if cls_id<len(NOMBRES_DEFAULT)
                  else f'C{cls_id}')
            for cid, cls_id in auto_asig.items()
        }
        self._aplicar_asign()
        self._calc_params()

        self.s_imgs.set_value(res['n_ok'])
        self.s_k.set_value(k)
        self.s_iner.set_value(f'{res["inercia"]:.0f}')
        self.log(f'✅ {res["n_ok"]}/{res["n_found"]} imágenes procesadas')
        for cid, nombre in self.asignacion.items():
            cnt = int(np.sum(self.etqs_raw==cid))
            self.log(f'   Cluster {cid} → {nombre} ({cnt})')

        self._evaluar()
        self._calcular_metricas_avanzadas()
        self._show_dashboard()
        self._show_clusters(); self._show_seg()
        self._show_eval(); self._show_analisis()
        self._show_estadisticas(); self._show_galeria()
        self.tabs.setCurrentIndex(0)
        self.progress.setValue(100); self.lbl_status.setText('Done ✓')

    def _calcular_metricas_avanzadas(self):
        """Calcula silhouette, davies-bouldin, calinski-harabasz, F1 por clase."""
        X = self.resultado['X']
        etqs = self.etqs_map
        centros = self.resultado['centros']
        k = self.resultado['k']
        self.log('\n📐 Métricas de clustering...')
        try:
            self.m_silhouette = silhouette_score_manual(X, etqs)
            self.m_davies     = davies_bouldin_manual(X, etqs, centros)
            self.m_calinski   = calinski_harabasz_manual(X, etqs, centros)
            self.log(f'   Silhouette: {self.m_silhouette:.3f}')
            self.log(f'   Davies-Bouldin: {self.m_davies:.3f}')
            self.log(f'   Calinski-Harabasz: {self.m_calinski:.1f}')
        except Exception as e:
            self.m_silhouette = self.m_davies = self.m_calinski = 0.0
            self.log(f'   Error métricas: {e}')

        # F1 por clase usando la mejor matriz de confusión (Bayes Robust)
        try:
            mc = self.mcs_mat[2][0]   # Bayes, Robust
            self.metricas_clase = metricas_por_clase(mc)
        except Exception:
            self.metricas_clase = {}

        # Importancia de features
        try:
            self.feat_importance = importancia_features(X, etqs)
        except Exception:
            self.feat_importance = np.zeros(X.shape[1])

    def _aplicar_asign(self):
        n2i = {nombre:idx for idx,nombre in self.nombres_clase.items()}
        self.etqs_map = np.array([
            n2i.get(self.asignacion.get(int(e),''), int(e))
            for e in self.etqs_raw
        ])

    def _calc_params(self):
        X = self.resultado['X']; y = self.etqs_map; k = self.resultado['k']
        ms,ics,dcs=[],[],[]
        for i in range(k):
            idx = y==i
            if idx.sum()<2:
                ms.append(X.mean(axis=0)); ics.append(np.eye(X.shape[1]))
                dcs.append(1.0); continue
            m,ic,dc = calcular_params(X[idx])
            ms.append(m); ics.append(ic); dcs.append(dc)
        self.medias_clf = ms; self.inv_covs_clf = ics; self.det_covs_clf = dcs

        # ── Calcular umbral de Mahalanobis para detección de outliers ──
        # Se basa en la distribución de distancias de las imágenes del
        # dataset a su clase más cercana. Una imagen externa estará más
        # lejos que el percentil 95 del dataset de cielos.
        dists_dataset = []
        for j in range(len(X)):
            dmin = np.inf
            for i in range(k):
                d = X[j] - ms[i]
                try:
                    dm = np.sqrt(max(0.0, d @ ics[i] @ d))
                except Exception:
                    dm = np.linalg.norm(d)
                dmin = min(dmin, dm)
            dists_dataset.append(dmin)
        dists_dataset = np.array(dists_dataset)
        # Umbral = percentil 97 + margen. Imágenes externas lo superan.
        self.umbral_mahal = float(
            np.percentile(dists_dataset, 97) * 1.5 + 1e-6)
        self.dist_media_dataset = float(dists_dataset.mean())
        self.log(f'   Umbral outlier (Mahalanobis): {self.umbral_mahal:.2f}')

    def abrir_asig(self):
        if not self.resultado:
            QMessageBox.warning(self,'Aviso','Ejecuta primero.'); return
        dlg = DialogAsignacion(
            self.resultado['k'], self.archivos, self.etqs_raw,
            self.asignacion, self.nombres_clase, self)
        if dlg.exec_() == QDialog.Accepted:
            ni = dlg.get_asignacion()
            self.asignacion = {
                cid: self.nombres_clase.get(cls_id,
                      NOMBRES_DEFAULT[cls_id] if cls_id<len(NOMBRES_DEFAULT)
                      else f'C{cls_id}')
                for cid, cls_id in ni.items()
            }
            self._aplicar_asign(); self._calc_params()
            self.log('\n🔀 Asignación actualizada')
            for cid,nombre in self.asignacion.items():
                cnt=int(np.sum(self.etqs_raw==cid)); self.log(f'   {cid}→{nombre}({cnt})')
            self._evaluar()
            self._calcular_metricas_avanzadas()
            self._show_dashboard()
            self._show_clusters(); self._show_eval()
            self._show_analisis(); self._show_estadisticas()
            self._show_galeria()

    def _evaluar(self):
        X = self.resultado['X']; y = self.etqs_map; k = self.resultado['k']
        nmet  = ['Euclidiana','Mahalanobis','Bayes']
        nmodo = ['Robust','Cross-Val','LOO']
        self.log('\n📊 Evaluando...'); self.accs_mat=[]; self.mcs_mat=[]
        for mi in range(3):
            am,mm=[],[]
            for mo in range(3):
                try:
                    a,mc = evaluar_clasificador(X,y,mi,k,mo,self.spin_folds.value())
                    am.append(a); mm.append(mc)
                    self.log(f'   {nmet[mi]}+{nmodo[mo]}: {a:.1f}%')
                except Exception as e:
                    am.append(0.0); mm.append(np.zeros((k,k),dtype=int))
                    self.log(f'   {nmet[mi]}+{nmodo[mo]}: Error({e})')
            self.accs_mat.append(am); self.mcs_mat.append(mm)
        todas  = [(self.accs_mat[mi][mo],nmet[mi],nmodo[mo])
                  for mi in range(3) for mo in range(3)]
        mejor  = max(todas, key=lambda x:x[0])
        self.log(f'\n★  MEJOR: {mejor[1]}+{mejor[2]}={mejor[0]:.1f}%')
        self.s_best.set_value(f'{mejor[0]:.0f}%')

    # ── TABS ──────────────────────────────────────────────────

    def _show_dashboard(self):
        """Panel ejecutivo con métricas clave y resumen visual, en scroll."""
        self._limpiar(self.tab_dash)
        lay = self.tab_dash.layout()
        k = self.resultado['k']
        etqs = self.etqs_map

        # Scroll para que todo respire
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f'background:{C_BG};border:none;')
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cont = QWidget(); cont.setStyleSheet(f'background:{C_BG};')
        dl = QVBoxLayout(cont); dl.setContentsMargins(4,4,4,4); dl.setSpacing(16)

        # ── Título ────────────────────────────────────────────
        titulo = QLabel('RESUMEN EJECUTIVO DEL ANÁLISIS')
        titulo.setStyleSheet(
            f"color:{C_INK};font-size:15px;font-weight:bold;letter-spacing:2px;"
            f"font-family:'{FONT_MONO}';border:none;padding:4px 0;")
        dl.addWidget(titulo)

        # ── Fila de tarjetas (más grandes) ────────────────────
        cards_row = QWidget(); cards_row.setStyleSheet('background:transparent;')
        cr = QHBoxLayout(cards_row); cr.setContentsMargins(0,0,0,0); cr.setSpacing(12)

        def metric_card(titulo, valor, sub, color):
            c = QFrame()
            c.setStyleSheet(
                f"QFrame{{background:{C_CARD};border:1px solid {C_BORDER};"
                f"border-top:4px solid {color};border-radius:12px;}}")
            c.setMinimumHeight(110)
            cl = QVBoxLayout(c); cl.setContentsMargins(18,16,18,16); cl.setSpacing(4)
            t = QLabel(titulo)
            t.setStyleSheet(f"color:{C_INK_DIM};font-size:10px;letter-spacing:1px;"
                            f"font-family:'{FONT_MONO}';border:none;")
            v = QLabel(valor)
            v.setStyleSheet(f"color:{color};font-size:32px;font-weight:bold;"
                            f"font-family:'{FONT_MONO}';border:none;")
            s = QLabel(sub)
            s.setStyleSheet(f"color:{C_INK_FAINT};font-size:10px;border:none;")
            s.setWordWrap(True)
            cl.addWidget(t); cl.addWidget(v); cl.addWidget(s)
            return c

        sil = getattr(self, 'm_silhouette', 0)
        calidad = ('Excelente' if sil>0.5 else 'Buena' if sil>0.35
                   else 'Razonable' if sil>0.2 else 'Débil')
        mejor_acc = 0; mejor_nombre = ''
        if self.accs_mat:
            nmet=['Euclidiana','Mahalanobis','Bayes']; nmodo=['Robust','CV','LOO']
            for mi in range(3):
                for mo in range(3):
                    if self.accs_mat[mi][mo] > mejor_acc:
                        mejor_acc = self.accs_mat[mi][mo]
                        mejor_nombre = f'{nmet[mi]}+{nmodo[mo]}'

        cr.addWidget(metric_card('IMÁGENES', str(self.resultado['n_ok']),
                                 f"de {self.resultado['n_found']} encontradas", C_SKY))
        cr.addWidget(metric_card('PRECISIÓN MÁX', f'{mejor_acc:.1f}%',
                                 mejor_nombre, '#4ade80'))
        cr.addWidget(metric_card('CALIDAD CLUSTERS', f'{sil:.3f}',
                                 f'Silhouette: {calidad}', C_ACCENT))
        cr.addWidget(metric_card('CLUSTERS', str(k),
                                 'grupos detectados', C_SUN))
        dl.addWidget(cards_row)

        # ── Fila 2 de métricas técnicas ───────────────────────
        cards_row2 = QWidget(); cards_row2.setStyleSheet('background:transparent;')
        cr2 = QHBoxLayout(cards_row2); cr2.setContentsMargins(0,0,0,0); cr2.setSpacing(12)
        db = getattr(self,'m_davies',0); ch = getattr(self,'m_calinski',0)
        db_cal = 'Excelente' if db<0.5 else 'Buena' if db<1.0 else 'Regular'
        cr2.addWidget(metric_card('DAVIES-BOULDIN', f'{db:.3f}',
                                  f'{db_cal} (menor=mejor)', '#ff8a65'))
        cr2.addWidget(metric_card('CALINSKI-H', f'{ch:.0f}',
                                  'separación (mayor=mejor)', '#22d3ee'))
        cr2.addWidget(metric_card('INERCIA', f'{self.resultado["inercia"]:.0f}',
                                  'compacidad K-Means', '#a78bfa'))
        # F1 promedio
        mc = getattr(self,'metricas_clase',{})
        f1_prom = np.mean([mc[c]['f1'] for c in mc]) if mc else 0
        cr2.addWidget(metric_card('F1 PROMEDIO', f'{f1_prom:.1f}%',
                                  'balance prec/recall', '#4ade80'))
        dl.addWidget(cards_row2)

        # ── Gráfica 1: Distribución (dona grande) ─────────────
        cols = plt.cm.plasma(np.linspace(0.1,0.9,k))
        counts = [int(np.sum(etqs==i)) for i in range(k)]
        labels = [self.nombres_clase.get(i,f'C{i}') for i in range(k)]

        fig1 = plt.figure(figsize=(13, 4.5)); fig1.patch.set_facecolor(C_BG)
        gs1 = gridspec.GridSpec(1, 2, figure=fig1, wspace=0.25,
                                left=0.04, right=0.96, top=0.86, bottom=0.14,
                                width_ratios=[1, 1.3])

        ax1 = fig1.add_subplot(gs1[0]); ax1.set_facecolor(C_BG)
        wedges, _ = ax1.pie(counts, colors=[cols[i] for i in range(k)],
                            startangle=90, counterclock=False,
                            wedgeprops=dict(width=0.40, edgecolor=C_BG, linewidth=3))
        ax1.add_artist(plt.Circle((0,0), 0.60, color=C_BG))
        ax1.text(0, 0.10, str(sum(counts)), ha='center', va='center',
                 color=C_INK, fontsize=26, fontweight='bold', fontfamily=FONT_MONO)
        ax1.text(0, -0.18, 'imágenes', ha='center', va='center',
                 color=C_INK_DIM, fontsize=10)
        ax1.set_title('Distribución por clase', color=C_INK, fontsize=12,
                      fontfamily=FONT_MONO, pad=12)

        # Barras horizontales con conteo a la derecha
        ax1b = fig1.add_subplot(gs1[1]); ax1b.set_facecolor('#0a0a12')
        y_pos = np.arange(k)
        bars = ax1b.barh(y_pos, counts, color=[cols[i] for i in range(k)],
                         alpha=0.9, height=0.6)
        for i, (bar, cnt) in enumerate(zip(bars, counts)):
            pct = cnt/sum(counts)*100
            ax1b.text(cnt + max(counts)*0.02, bar.get_y()+bar.get_height()/2,
                      f'{cnt}  ({pct:.1f}%)', va='center', color=C_INK,
                      fontsize=10, fontweight='bold', fontfamily=FONT_MONO)
        ax1b.set_yticks(y_pos)
        ax1b.set_yticklabels(labels, fontsize=10, color=C_INK)
        ax1b.set_xlim(0, max(counts)*1.25)
        ax1b.invert_yaxis()
        ax1b.set_title('Cantidad de imágenes por clase', color=C_INK,
                       fontsize=12, fontfamily=FONT_MONO, pad=12)
        ax1b.tick_params(colors=C_INK_FAINT, labelsize=9)
        ax1b.grid(axis='x', alpha=0.06, color='white', linestyle='--')
        for sp in ax1b.spines.values(): sp.set_edgecolor(C_BORDER)
        ax1b.spines['top'].set_visible(False); ax1b.spines['right'].set_visible(False)

        c1 = FigureCanvas(fig1); c1.setStyleSheet(f'background:{C_BG};')
        c1.setMinimumHeight(330)
        dl.addWidget(c1); c1.draw(); plt.close(fig1)

        # ── Gráfica 2: F1 por clase + PCA varianza ────────────
        fig2 = plt.figure(figsize=(13, 4.5)); fig2.patch.set_facecolor(C_BG)
        gs2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.22,
                                left=0.06, right=0.97, top=0.86, bottom=0.16)

        ax2 = fig2.add_subplot(gs2[0]); ax2.set_facecolor('#0a0a12')
        if mc:
            clases_idx = sorted(mc.keys())
            f1s = [mc[c]['f1'] for c in clases_idx]
            precs = [mc[c]['precision'] for c in clases_idx]
            recs = [mc[c]['recall'] for c in clases_idx]
            x = np.arange(len(clases_idx)); w = 0.26
            b1 = ax2.bar(x-w, precs, w, label='Precisión', color=C_SKY, alpha=0.95)
            b2 = ax2.bar(x, recs, w, label='Recall', color=C_ACCENT, alpha=0.95)
            b3 = ax2.bar(x+w, f1s, w, label='F1-Score', color='#4ade80', alpha=0.95)
            for bars in [b1,b2,b3]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 5:
                        ax2.text(bar.get_x()+bar.get_width()/2, h+1.5,
                                 f'{h:.0f}', ha='center', color=C_INK_DIM,
                                 fontsize=6)
            ax2.set_xticks(x)
            ax2.set_xticklabels([self.nombres_clase.get(c,f'C{c}')
                                 for c in clases_idx], rotation=15,
                                fontsize=9, color=C_INK)
            ax2.legend(fontsize=9, facecolor=C_PANEL, labelcolor=C_INK,
                       edgecolor=C_BORDER, loc='lower right')
            ax2.set_ylim(0, 118)
        ax2.set_title('Métricas de calidad por clase (%)', color=C_INK,
                      fontsize=12, fontfamily=FONT_MONO, pad=12)
        ax2.set_ylabel('Porcentaje', color=C_INK_DIM, fontsize=10)
        ax2.tick_params(colors=C_INK_FAINT, labelsize=9)
        ax2.grid(axis='y', alpha=0.06, color='white', linestyle='--')
        for sp in ax2.spines.values(): sp.set_edgecolor(C_BORDER)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

        ax3 = fig2.add_subplot(gs2[1]); ax3.set_facecolor('#0a0a12')
        try:
            var_exp, var_acum = varianza_explicada_pca(self.resultado['X'], 10)
            xc = range(1, len(var_exp)+1)
            ax3.bar(xc, var_exp, color=C_SKY, alpha=0.6, label='Por componente')
            ax3.plot(xc, var_acum, 'o-', color=C_SUN, markersize=6,
                     linewidth=2, label='Acumulada')
            ax3.axhline(90, color=C_STORM, linestyle='--', alpha=0.6, linewidth=1.5)
            ax3.text(len(var_exp)*0.5, 92, 'umbral 90%', color=C_STORM,
                     fontsize=8, fontfamily=FONT_MONO)
            ax3.legend(fontsize=9, facecolor=C_PANEL, labelcolor=C_INK,
                       edgecolor=C_BORDER, loc='center right')
            ax3.set_ylim(0, 105)
        except Exception: pass
        ax3.set_title('Varianza explicada por PCA', color=C_INK, fontsize=12,
                      fontfamily=FONT_MONO, pad=12)
        ax3.set_xlabel('Componente principal', color=C_INK_DIM, fontsize=10)
        ax3.set_ylabel('% varianza', color=C_INK_DIM, fontsize=10)
        ax3.tick_params(colors=C_INK_FAINT, labelsize=9)
        ax3.grid(axis='y', alpha=0.06, color='white', linestyle='--')
        for sp in ax3.spines.values(): sp.set_edgecolor(C_BORDER)
        ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

        c2 = FigureCanvas(fig2); c2.setStyleSheet(f'background:{C_BG};')
        c2.setMinimumHeight(330)
        dl.addWidget(c2); c2.draw(); plt.close(fig2)

        # ── Interpretación textual ────────────────────────────
        interp = QFrame()
        interp.setStyleSheet(
            f"QFrame{{background:{C_SURFACE};border:1px solid {C_BORDER};"
            f"border-left:4px solid {C_ACCENT};border-radius:8px;}}")
        il = QVBoxLayout(interp); il.setContentsMargins(18,14,18,14); il.setSpacing(6)
        ih = QLabel('INTERPRETACIÓN AUTOMÁTICA')
        ih.setStyleSheet(f"color:{C_ACCENT};font-size:10px;letter-spacing:2px;"
                         f"font-family:'{FONT_MONO}';border:none;")
        il.addWidget(ih)

        clase_may = labels[int(np.argmax(counts))]
        txt = (f'El análisis procesó <b>{self.resultado["n_ok"]} imágenes</b> '
               f'agrupándolas en <b>{k} clusters</b>. La clase más frecuente es '
               f'<b>{clase_may}</b> con {max(counts)} imágenes '
               f'({max(counts)/sum(counts)*100:.0f}%). ')
        txt += (f'El mejor clasificador fue <b>{mejor_nombre}</b> con '
                f'<b>{mejor_acc:.1f}%</b> de precisión. ')
        txt += (f'La calidad de separación de clusters es <b>{calidad.lower()}</b> '
                f'(Silhouette={sil:.3f}). ')
        if f1_prom > 0:
            txt += (f'El F1 promedio de <b>{f1_prom:.0f}%</b> indica un '
                    f'{"buen" if f1_prom>=70 else "aceptable" if f1_prom>=50 else "débil"} '
                    f'balance entre precisión y exhaustividad.')
        lbl_txt = QLabel(txt)
        lbl_txt.setStyleSheet(f"color:{C_INK};font-size:12px;border:none;line-height:1.5;")
        lbl_txt.setWordWrap(True)
        il.addWidget(lbl_txt)
        dl.addWidget(interp)

        dl.addStretch()
        scroll.setWidget(cont)
        lay.addWidget(scroll)

    def _show_analisis(self):
        """Matriz de correlación + importancia de features."""
        self._limpiar(self.tab_anal)
        lay = self.tab_anal.layout()
        X = self.resultado['X']

        # Features clave para correlación (las más interpretables)
        feat_idx = [54,56,57,58,72,73,65,67,76,99,94,95]
        feat_nombres = ['azul','blanco','verde','oscuro','sc_glob','sc_sup',
                        'grad_sup','grad_inf','lbp_ent','sat_sup',
                        'az_hsv','bl_hsv']
        feat_idx = [i for i in feat_idx if i < X.shape[1]]
        feat_nombres = feat_nombres[:len(feat_idx)]

        fig = plt.figure(figsize=(14, 6)); fig.patch.set_facecolor(C_BG)
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3,
                               left=0.10, right=0.97, top=0.90, bottom=0.20)

        # Matriz de correlación
        ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor('#0a0a12')
        corr = correlacion_features(X, feat_idx)
        im = ax1.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(len(feat_nombres)))
        ax1.set_yticks(range(len(feat_nombres)))
        ax1.set_xticklabels(feat_nombres, rotation=45, ha='right',
                            fontsize=7, color=C_INK)
        ax1.set_yticklabels(feat_nombres, fontsize=7, color=C_INK)
        for i in range(len(feat_nombres)):
            for j in range(len(feat_nombres)):
                v = corr[i,j]
                tc = 'white' if abs(v) > 0.5 else C_INK_DIM
                ax1.text(j, i, f'{v:.2f}', ha='center', va='center',
                         color=tc, fontsize=6)
        ax1.set_title('Matriz de correlación de features',
                      color=C_INK, fontsize=10, fontfamily=FONT_MONO, pad=8)
        ax1.tick_params(colors=C_INK_FAINT)
        plt.colorbar(im, ax=ax1, shrink=0.7).ax.tick_params(
            colors=C_INK_DIM, labelcolor=C_INK_DIM)

        # Importancia de features (top 20)
        ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor('#0a0a12')
        fi = getattr(self, 'feat_importance', np.zeros(X.shape[1]))
        nombres_todos = self._nombres_features()
        top_idx = np.argsort(fi)[::-1][:18]
        top_vals = fi[top_idx]
        top_nombres = [nombres_todos[i] if i < len(nombres_todos) else f'f{i}'
                       for i in top_idx]
        y_pos = np.arange(len(top_idx))
        ax2.barh(y_pos, top_vals[::-1], color=C_ACCENT, alpha=0.85)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([top_nombres[::-1][i] for i in range(len(top_idx))],
                            fontsize=7, color=C_INK)
        ax2.set_title('Importancia de features (F-score ANOVA)',
                      color=C_INK, fontsize=10, fontfamily=FONT_MONO, pad=8)
        ax2.set_xlabel('Poder discriminante', color=C_INK_DIM, fontsize=8)
        ax2.tick_params(colors=C_INK_FAINT, labelsize=7)
        ax2.grid(axis='x', alpha=0.06, color='white', linestyle='--')
        for sp in ax2.spines.values(): sp.set_edgecolor(C_BORDER)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

        canvas = FigureCanvas(fig); canvas.setStyleSheet(f'background:{C_BG};')
        lay.addWidget(canvas, 1); canvas.draw(); plt.close(fig)

        info = QLabel(
            'La matriz de correlación muestra qué features varían juntas '
            '(rojo=correlación positiva, azul=negativa). La importancia mide '
            'cuánto discrimina cada feature entre las clases (F-score tipo ANOVA: '
            'varianza inter-clase / intra-clase).')
        info.setStyleSheet(
            f"color:{C_INK_FAINT};font-size:9px;background:{C_SURFACE};"
            f"border:1px solid {C_BORDER};border-radius:6px;padding:8px 12px;")
        info.setWordWrap(True)
        lay.addWidget(info)

    def _show_estadisticas(self):
        """Distribución (boxplots) de features clave por clase."""
        self._limpiar(self.tab_stats)
        lay = self.tab_stats.layout()
        X = self.resultado['X']
        etqs = self.etqs_map
        k = self.resultado['k']
        cols = plt.cm.plasma(np.linspace(0.1,0.9,k))

        # Features a graficar
        feats_plot = [
            (54, 'Pct Azul'), (56, 'Pct Blanco'), (73, 'Score Cielo Sup'),
            (99, 'Saturación Sup'), (65, 'Grad Sup'), (76, 'LBP Entropía'),
        ]
        feats_plot = [(i,n) for i,n in feats_plot if i < X.shape[1]]

        fig = plt.figure(figsize=(14, 7)); fig.patch.set_facecolor(C_BG)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3,
                               left=0.06, right=0.97, top=0.93, bottom=0.08)
        labels = [self.nombres_clase.get(i,f'C{i}') for i in range(k)]

        for plot_idx, (fidx, fname) in enumerate(feats_plot):
            ax = fig.add_subplot(gs[plot_idx//3, plot_idx%3])
            ax.set_facecolor('#0a0a12')
            datos = [X[etqs==i, fidx] for i in range(k)]
            datos = [d if len(d)>0 else np.array([0]) for d in datos]
            bp = ax.boxplot(datos, patch_artist=True, widths=0.6,
                            medianprops=dict(color='white', linewidth=1.5),
                            flierprops=dict(marker='o', markersize=2,
                                            markerfacecolor=C_INK_FAINT,
                                            markeredgecolor='none', alpha=0.4))
            for patch, color in zip(bp['boxes'], cols):
                patch.set_facecolor(color); patch.set_alpha(0.7)
                patch.set_edgecolor(color)
            for w in bp['whiskers']: w.set_color(C_INK_FAINT)
            for c in bp['caps']: c.set_color(C_INK_FAINT)
            ax.set_xticklabels([l[:6] for l in labels], rotation=20,
                               fontsize=7, color=C_INK)
            ax.set_title(fname, color=C_INK, fontsize=9,
                         fontfamily=FONT_MONO, pad=6)
            ax.tick_params(colors=C_INK_FAINT, labelsize=7)
            ax.grid(axis='y', alpha=0.06, color='white', linestyle='--')
            for sp in ax.spines.values(): sp.set_edgecolor(C_BORDER)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        fig.suptitle('Distribución de features por clase (box plots)',
                     color=C_INK, fontsize=11, fontfamily=FONT_MONO)
        canvas = FigureCanvas(fig); canvas.setStyleSheet(f'background:{C_BG};')
        lay.addWidget(canvas, 1); canvas.draw(); plt.close(fig)

        info = QLabel(
            'Cada caja muestra el rango intercuartílico (25%-75%) de una feature '
            'para cada clase. La línea blanca es la mediana. Cajas bien separadas '
            'verticalmente indican que esa feature discrimina bien entre clases.')
        info.setStyleSheet(
            f"color:{C_INK_FAINT};font-size:9px;background:{C_SURFACE};"
            f"border:1px solid {C_BORDER};border-radius:6px;padding:8px 12px;")
        info.setWordWrap(True)
        lay.addWidget(info)

    def _nombres_features(self):
        """Lista de nombres de las 100 features."""
        nombres = []
        for canal in ['R','G','B']:
            for bi in range(16): nombres.append(f'hist_{canal}{bi}')
        for canal in ['R','G','B']: nombres.append(f'media_{canal}')
        for canal in ['R','G','B']: nombres.append(f'std_{canal}')
        nombres += ['pct_azul','pct_gris','pct_blanco','pct_verde','pct_oscuro',
                    'sat_rgb','contraste']
        nombres += ['paz_sup','pbl_sup','pgr_sup']
        nombres += ['grad_glob','grad_sup','grad_med','grad_inf','grad_var',
                    'ratio_tex','suav_B','ratio_HV']
        nombres += ['score_cielo_g','score_cielo_s']
        nombres += ['lbp_mean','lbp_std','lbp_entr','lbp_unif']
        nombres += ['entr_lum','pct_bril_s','pct_med_s','pct_osc_s']
        nombres += ['hue_mg','hue_sg','sat_mg','sat_sg','val_mg','val_sg']
        nombres += ['hue_ms','hue_ss','sat_ms','val_ms','sat_ss','val_ss']
        nombres += ['az_hsv_s','bl_hsv_s','gr_hsv_s','b_rel_s','unif_s','sat_s']
        return nombres

    def _show_clusters(self):
        self._limpiar(self.tab_cl)
        X    = self.resultado['X']; etqs = self.etqs_map; k = self.resultado['k']
        cols = plt.cm.plasma(np.linspace(0.1,0.9,k))

        fig = plt.figure(figsize=(14,5.5)); fig.patch.set_facecolor(C_BG)
        Xc   = X - X.mean(axis=0)
        _, vv = np.linalg.eigh(np.cov(Xc.T))
        X3d  = Xc @ vv[:,-3:]

        ax3d = fig.add_subplot(121, projection='3d')
        ax3d.set_facecolor('#0a0a12')
        ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor(C_BORDER)
        ax3d.yaxis.pane.set_edgecolor(C_BORDER)
        ax3d.zaxis.pane.set_edgecolor(C_BORDER)
        ax3d.grid(True, alpha=0.07, color='white')
        for i in range(k):
            mask = etqs==i
            if not mask.any(): continue
            ax3d.scatter(X3d[mask,0],X3d[mask,1],X3d[mask,2],
                         color=cols[i], s=12, alpha=0.6,
                         label=self.nombres_clase.get(i,f'C{i}'),
                         depthshade=True)
        ax3d.set_title('PCA 3D', color=C_INK, fontsize=10,
                       fontfamily=FONT_MONO, pad=8)
        ax3d.tick_params(colors=C_INK_FAINT, labelsize=7)
        ax3d.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_BORDER,
                    labelcolor=C_INK, framealpha=0.9)
        ax3d.view_init(elev=25, azim=45)

        ax2 = fig.add_subplot(122); ax2.set_facecolor('#0a0a12')
        counts = [int(np.sum(etqs==i)) for i in range(k)]
        bars   = ax2.bar(range(k), counts,
                         color=[cols[i] for i in range(k)],
                         edgecolor='none', width=0.6)
        for bar,cnt in zip(bars,counts):
            ax2.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.5, str(cnt),
                     ha='center', color=C_INK, fontsize=9, fontweight='bold')
        ax2.set_xticks(range(k))
        ax2.set_xticklabels([self.nombres_clase.get(i,f'C{i}') for i in range(k)],
                             rotation=15, fontsize=8, color=C_INK)
        ax2.tick_params(colors=C_INK_FAINT, labelsize=8)
        ax2.set_ylabel('Imágenes', color=C_INK_DIM, fontsize=9)
        ax2.set_title('Distribución', color=C_INK, fontsize=10,
                      fontfamily=FONT_MONO, pad=8)
        ax2.grid(axis='y', alpha=0.06, color='white', linestyle='--')
        for sp in ax2.spines.values(): sp.set_edgecolor(C_BORDER)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

        fig.tight_layout(pad=2)
        canvas = FigureCanvas(fig); canvas.setStyleSheet(f'background:{C_BG};')
        self.tab_cl.layout().addWidget(canvas)
        canvas.draw(); plt.close(fig)
        tip = QLabel('💡  Arrastra la gráfica 3D para rotar')
        tip.setStyleSheet(
            f"color:{C_INK_FAINT};font-size:9px;font-family:'{FONT_MONO}';")
        self.tab_cl.layout().addWidget(tip)

    def _show_seg(self):
        self._limpiar(self.tab_sg)
        lay = self.tab_sg.layout()

        # ── Controles ─────────────────────────────────────────
        ctrl = QFrame()
        ctrl.setStyleSheet(
            f'background:{C_PANEL};border-radius:8px;border:1px solid {C_BORDER};')
        cl = QHBoxLayout(ctrl); cl.setContentsMargins(14,8,14,8); cl.setSpacing(10)
        cl.addWidget(QLabel('Imagen:'))
        self.combo_seg = QComboBox()
        self.combo_seg.setStyleSheet(
            f'QComboBox{{{style_input()}}}'
            f'QComboBox::drop-down{{border:none;}}'
            f'QComboBox QAbstractItemView{{background:{C_CARD};color:{C_INK};'
            f'selection-background-color:{C_ACCENT};}}')
        self.combo_seg.setMinimumWidth(240)
        k = self.resultado['k']
        for i in range(k):
            idx_c = np.where(self.etqs_raw==i)[0]
            if len(idx_c)>0:
                path = self.archivos[idx_c[0]]
                self.combo_seg.addItem(
                    f'{self.asignacion.get(i,f"C{i}")} — {os.path.basename(path)}',
                    path)
        cl.addWidget(self.combo_seg,1)
        blib = QPushButton('Abrir imagen externa…')
        blib.setStyleSheet(style_btn(C_PANEL,C_INK,'6px 12px'))
        blib.setStyleSheet(blib.styleSheet()+f'QPushButton{{border:1px solid {C_BORDER_LT};}}')
        blib.clicked.connect(self._seg_elegir); cl.addWidget(blib)
        bseg = QPushButton('Analizar ▶')
        bseg.setStyleSheet(style_btn(C_ACCENT))
        bseg.clicked.connect(self._seg_update); cl.addWidget(bseg)
        lay.addWidget(ctrl)

        # ── Resultado de clasificación ────────────────────────
        self.seg_res = QFrame()
        self.seg_res.setStyleSheet(
            f'background:{C_PANEL};border-radius:8px;border:1px solid {C_BORDER};')
        srl = QHBoxLayout(self.seg_res)
        srl.setContentsMargins(14,8,14,8); srl.setSpacing(14)
        lt = QLabel('RESULTADO:')
        lt.setStyleSheet(
            f"color:{C_INK_DIM};font-size:9px;letter-spacing:2px;"
            f"font-family:'{FONT_MONO}';border:none;")
        srl.addWidget(lt)
        self.seg_lbl_cl = QLabel('—')
        self.seg_lbl_cl.setStyleSheet(
            f"color:{C_SKY};font-size:19px;font-weight:bold;"
            f"font-family:'{FONT_MONO}';border:none;")
        self.seg_lbl_cf = QLabel('')
        self.seg_lbl_cf.setStyleSheet(f"color:{C_INK_DIM};font-size:11px;border:none;")
        self.seg_lbl_pc = QLabel('')
        self.seg_lbl_pc.setStyleSheet(f"color:{C_SKY};font-size:11px;border:none;")
        srl.addWidget(self.seg_lbl_cl); srl.addWidget(self.seg_lbl_cf)
        srl.addWidget(self.seg_lbl_pc); srl.addStretch()
        lay.addWidget(self.seg_res)

        # ── Canvas segmentación: 2 paneles ────────────────────
        self.fig_seg, self.axes_seg = plt.subplots(1, 2, figsize=(11, 3.4))
        self.fig_seg.patch.set_facecolor(C_BG)
        self.canvas_seg = FigureCanvas(self.fig_seg)
        self.canvas_seg.setStyleSheet(f'background:{C_BG};')
        self.canvas_seg.setMaximumHeight(290)
        lay.addWidget(self.canvas_seg)

        # ── Info de métodos usados ────────────────────────────
        self.seg_metodos = QLabel('')
        self.seg_metodos.setStyleSheet(
            f"color:{C_INK_FAINT};font-size:9px;font-family:'{FONT_MONO}';"
            f"background:{C_SURFACE};border:1px solid {C_BORDER};"
            f"border-radius:6px;padding:8px 12px;")
        self.seg_metodos.setWordWrap(True)
        lay.addWidget(self.seg_metodos)

        # ── Panel de imágenes similares ───────────────────────
        sim_hdr = QLabel('IMÁGENES SIMILARES DE LA MISMA CLASE')
        sim_hdr.setStyleSheet(
            f"color:{C_INK_DIM};font-size:9px;letter-spacing:2px;"
            f"font-family:'{FONT_MONO}';border:none;padding-top:4px;")
        lay.addWidget(sim_hdr)

        self.seg_sim_scroll = QScrollArea()
        self.seg_sim_scroll.setWidgetResizable(True)
        self.seg_sim_scroll.setStyleSheet(f'background:{C_BG};border:none;')
        self.seg_sim_scroll.setMaximumHeight(180)
        self.seg_sim_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.seg_sim_w = QWidget(); self.seg_sim_w.setStyleSheet(f'background:{C_BG};')
        self.seg_sim_lay = QHBoxLayout(self.seg_sim_w)
        self.seg_sim_lay.setSpacing(8); self.seg_sim_lay.setContentsMargins(4,4,4,4)
        self.seg_sim_scroll.setWidget(self.seg_sim_w)
        lay.addWidget(self.seg_sim_scroll)

        self._seg_update()

    def _seg_elegir(self):
        path,_ = QFileDialog.getOpenFileName(
            self,'Selecciona imagen',
            getattr(self,'carpeta',''),
            'Imágenes (*.jpg *.jpeg *.png *.bmp *.webp)')
        if path:
            for i in range(self.combo_seg.count()):
                if self.combo_seg.itemData(i)==path:
                    self.combo_seg.setCurrentIndex(i)
                    self._seg_update(); return
            self.combo_seg.addItem(f'EXTERNA: {os.path.basename(path)}', path)
            self.combo_seg.setCurrentIndex(self.combo_seg.count()-1)
            self._seg_update()

    def _seg_update(self):
        path = self.combo_seg.currentData()
        if not path: return

        for ax in self.axes_seg:
            ax.clear(); ax.set_facecolor('#0a0a12')

        arr, mask, overlay, score, grad = segmentar_cielo(path)
        feat = extraer_features(path)

        cl_final  = None
        es_outlier = False
        pct_cielo  = 0.0

        # ── Clasificación con detección de outlier ────────────
        if self.medias_clf and feat is not None:
            nc = len(self.medias_clf)
            umbral_m = getattr(self, 'umbral_mahal', None)
            cl_final, conf, es_outlier = clasificar_con_outlier(
                feat, self.medias_clf, self.inv_covs_clf,
                self.det_covs_clf, 2, nc, umbral_mahal=umbral_m)

            nombre_c = self.nombres_clase.get(cl_final, 'No relacionado'
                        if cl_final == 3 else f'C{cl_final}')
            if cl_final == 3:
                nombre_c = 'No relacionado'
                color_c  = C_STORM
            else:
                color_c  = C_CLASSES.get(cl_final, C_INK_DIM)

            self.seg_lbl_cl.setText(nombre_c)
            self.seg_lbl_cl.setStyleSheet(
                f"color:{color_c};font-size:19px;font-weight:bold;"
                f"font-family:'{FONT_MONO}';border:none;")
            if es_outlier:
                self.seg_lbl_cf.setText('⚠ No coincide con ninguna clase de cielo')
            else:
                self.seg_lbl_cf.setText(f'Confianza: {conf:.1f}%')
            if mask is not None:
                pct_cielo = float(mask.mean()) * 100
                self.seg_lbl_pc.setText(f'Cielo: {pct_cielo:.1f}%')

        if arr is None:
            self.canvas_seg.draw(); return

        # ── Panel 1: original + contorno ──────────────────────
        ax0 = self.axes_seg[0]
        ax0.imshow(arr, aspect='auto')
        if mask is not None:
            ax0.contour(mask.astype(float), levels=[0.5],
                        colors=['#00e5ff'], linewidths=[2.5], alpha=1.0)
        ax0.set_title('Original + límite del cielo',
                      color=C_INK, fontsize=10, fontfamily=FONT_MONO, pad=6)
        ax0.axis('off')

        # ── Panel 2: cielo sombreado + contorno ───────────────
        ax1 = self.axes_seg[1]
        ax1.imshow(overlay, aspect='auto')
        if mask is not None:
            ax1.contour(mask.astype(float), levels=[0.5],
                        colors=['#ffffff'], linewidths=[2.0], alpha=1.0)
            pct = float(mask.mean()) * 100
            ax1.text(0.5, 0.08, f'CIELO  {pct:.1f}%',
                     transform=ax1.transAxes, color='white',
                     fontsize=11, fontweight='bold', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.4',
                               facecolor='#1a5fff', alpha=0.80, edgecolor='none'))
            if pct < 95:
                ax1.text(0.5, 0.95, f'NO CIELO  {100-pct:.1f}%',
                         transform=ax1.transAxes, color='white',
                         fontsize=10, fontweight='bold', ha='center', va='top',
                         bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='#333355', alpha=0.75, edgecolor='none'))
        ax1.set_title('Segmentación cielo / no-cielo',
                      color=C_SKY, fontsize=10, fontfamily=FONT_MONO, pad=6)
        ax1.axis('off')

        nombre_sel = self.combo_seg.currentText().split('—')[0].strip()
        self.fig_seg.suptitle(nombre_sel, color=C_INK_DIM,
                              fontsize=10, fontfamily=FONT_MONO, y=1.02)
        self.fig_seg.tight_layout(pad=0.8)
        self.canvas_seg.draw()

        # ── Info de métodos ───────────────────────────────────
        metodos_txt = (
            'MÉTODOS UTILIZADOS:  '
            '[1] Segmentación: detección multi-cue (RGB diferencial + HSV + '
            'gradiente paso-altas + posición vertical) + morfología '
            '(fill-holes, dilatación, erosión).   '
            '[2] Clasificación: Bayes (máxima verosimilitud gaussiana).   '
            '[3] Detección no-relacionado: distancia de Mahalanobis vs '
            f'umbral={getattr(self,"umbral_mahal",0) or 0:.1f}.   '
            '[4] Similitud: coseno (50%) + euclidiana exponencial (50%).')
        self.seg_metodos.setText(metodos_txt)

        # ── Imágenes similares ────────────────────────────────
        self._mostrar_similares(feat, cl_final, es_outlier)

    def _mostrar_similares(self, feat, cl_final, es_outlier):
        # Limpiar
        while self.seg_sim_lay.count():
            item = self.seg_sim_lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        if feat is None or es_outlier or cl_final is None or cl_final == 3:
            lbl = QLabel('Imagen no relacionada: sin imágenes similares de cielo.'
                         if es_outlier else 'Sin datos de similitud.')
            lbl.setStyleSheet(f"color:{C_INK_FAINT};font-size:10px;border:none;")
            self.seg_sim_lay.addWidget(lbl); self.seg_sim_lay.addStretch()
            return

        # Buscar imágenes de la misma clase asignada
        X = self.resultado['X']
        indices_clase = [i for i in range(len(self.archivos))
                         if self.etqs_map[i] == cl_final]

        # Calcular similitud de cada una a la imagen actual
        sims = []
        for i in indices_clase:
            s_cos = similitud_coseno(feat, X[i])
            s_euc = similitud_euclidiana(feat, X[i])
            s_tot = s_cos * 0.5 + s_euc * 0.5
            sims.append((s_tot, s_cos, s_euc, i))
        sims.sort(reverse=True)

        nombre_clase = self.nombres_clase.get(cl_final, f'C{cl_final}')

        # Mostrar las 5 más similares
        top5 = sims[:5]
        if not top5:
            lbl = QLabel('No hay otras imágenes en esta clase.')
            lbl.setStyleSheet(f"color:{C_INK_FAINT};font-size:10px;border:none;")
            self.seg_sim_lay.addWidget(lbl); self.seg_sim_lay.addStretch()
            return

        for rank, (s_tot, s_cos, s_euc, idx) in enumerate(top5, 1):
            card = QFrame()
            col = C_CLASSES.get(cl_final, C_INK_DIM)
            card.setStyleSheet(
                f"QFrame{{background:{C_CARD};border:1px solid {C_BORDER};"
                f"border-radius:8px;}}")
            card.setFixedSize(140, 150)
            fl = QVBoxLayout(card); fl.setContentsMargins(6,6,6,6); fl.setSpacing(3)

            # Thumbnail
            li = QLabel(); li.setAlignment(Qt.AlignCenter)
            li.setFixedHeight(80); li.setStyleSheet('border:none;')
            try:
                img = Image.open(self.archivos[idx]).convert('RGB').resize((126,78))
                a = np.array(img).copy()
                qi = QImage(a.data, 126, 78, 378, QImage.Format_RGB888)
                li.setPixmap(QPixmap.fromImage(qi.copy()))
            except Exception: pass
            fl.addWidget(li)

            # Nombre
            fn = QLabel(f'#{rank}  {os.path.basename(self.archivos[idx])[:12]}')
            fn.setStyleSheet(f"color:{C_INK_DIM};font-size:8px;border:none;")
            fn.setAlignment(Qt.AlignCenter)
            fl.addWidget(fn)

            # Barra de similitud
            sim_lbl = QLabel(f'{s_tot:.1f}% similar')
            sim_color = '#4ade80' if s_tot >= 70 else (C_SUN if s_tot >= 50 else C_STORM)
            sim_lbl.setStyleSheet(
                f"color:{sim_color};font-size:10px;font-weight:bold;"
                f"font-family:'{FONT_MONO}';border:none;")
            sim_lbl.setAlignment(Qt.AlignCenter)
            fl.addWidget(sim_lbl)

            # Detalle coseno/euclidiana
            det = QLabel(f'cos {s_cos:.0f}% · euc {s_euc:.0f}%')
            det.setStyleSheet(f"color:{C_INK_FAINT};font-size:7px;border:none;")
            det.setAlignment(Qt.AlignCenter)
            fl.addWidget(det)

            self.seg_sim_lay.addWidget(card)

        # Resumen de similitud media a la clase
        sim_media = np.mean([s[0] for s in top5])
        resumen = QFrame()
        resumen.setStyleSheet(
            f"QFrame{{background:{C_PANEL};border:1px solid {C_BORDER_LT};"
            f"border-radius:8px;}}")
        resumen.setFixedSize(150, 150)
        rl = QVBoxLayout(resumen); rl.setContentsMargins(10,10,10,10)
        r_t = QLabel('SIMILITUD MEDIA')
        r_t.setStyleSheet(
            f"color:{C_INK_DIM};font-size:8px;letter-spacing:1px;"
            f"font-family:'{FONT_MONO}';border:none;")
        r_t.setAlignment(Qt.AlignCenter)
        r_v = QLabel(f'{sim_media:.1f}%')
        col_m = '#4ade80' if sim_media >= 70 else (C_SUN if sim_media >= 50 else C_STORM)
        r_v.setStyleSheet(
            f"color:{col_m};font-size:26px;font-weight:bold;"
            f"font-family:'{FONT_MONO}';border:none;")
        r_v.setAlignment(Qt.AlignCenter)
        r_c = QLabel(f'a "{nombre_clase}"')
        r_c.setStyleSheet(f"color:{C_INK_DIM};font-size:9px;border:none;")
        r_c.setAlignment(Qt.AlignCenter)
        rl.addStretch(); rl.addWidget(r_t); rl.addWidget(r_v)
        rl.addWidget(r_c); rl.addStretch()
        self.seg_sim_lay.addWidget(resumen)
        self.seg_sim_lay.addStretch()

    def _show_eval(self):
        self._limpiar(self.tab_ev)
        if not self.accs_mat: return
        k           = self.resultado['k']
        nmet        = ['Euclidiana','Mahalanobis','Bayes']
        nmodo       = ['Robust','Cross-Val','LOO']
        labels_cl   = [self.nombres_clase.get(i,f'C{i}') for i in range(k)]
        lay         = self.tab_ev.layout()

        # ── Gráfica de barras de precisión ──────────────────
        fig1, ax_b = plt.subplots(figsize=(12,3.5))
        fig1.patch.set_facecolor(C_BG)
        ax_b.set_facecolor('#0a0a12')
        x  = np.arange(3); w = 0.24
        cm = [C_SKY, C_ACCENT, '#4ade80']
        for mo in range(3):
            vals = [self.accs_mat[mi][mo] for mi in range(3)]
            bars = ax_b.bar(x+mo*w, vals, w, label=nmodo[mo],
                            color=cm[mo], edgecolor='none', alpha=0.85)
            for bar,val in zip(bars,vals):
                ax_b.text(bar.get_x()+bar.get_width()/2,
                          bar.get_height()+0.5, f'{val:.0f}%',
                          ha='center', color=C_INK, fontsize=8)
        ax_b.set_xticks(x+w)
        ax_b.set_xticklabels(nmet, color=C_INK, fontsize=11)
        ax_b.set_ylim(0, 115)
        ax_b.legend(fontsize=9, facecolor=C_PANEL,
                    edgecolor=C_BORDER, labelcolor=C_INK)
        ax_b.set_title('Precisión por Clasificador y Modo de Evaluación',
                       color=C_INK, fontsize=11, fontfamily=FONT_MONO, pad=10)
        ax_b.set_ylabel('Precisión (%)', color=C_INK_DIM, fontsize=9)
        ax_b.tick_params(colors=C_INK_FAINT, labelsize=8)
        ax_b.grid(axis='y', alpha=0.06, color='white', linestyle='--')
        for sp in ax_b.spines.values(): sp.set_edgecolor(C_BORDER)
        ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
        fig1.tight_layout(pad=1.5)
        c1 = FigureCanvas(fig1); c1.setStyleSheet(f'background:{C_BG};')
        lay.addWidget(c1); c1.draw(); plt.close(fig1)

        # ── Matrices de confusión (3 — una por clasificador, modo Robust) ──
        fig2, axes = plt.subplots(1, 3, figsize=(14,4))
        fig2.patch.set_facecolor(C_BG)
        for mi in range(3):
            ax = axes[mi]; ax.set_facecolor('#0a0a12')
            mc = self.mcs_mat[mi][0]  # modo Robust
            im = ax.imshow(mc, cmap='viridis', aspect='auto')
            ax.set_xticks(range(k)); ax.set_yticks(range(k))
            ax.set_xticklabels(labels_cl, rotation=30, ha='right',
                               fontsize=8, color=C_INK)
            ax.set_yticklabels(labels_cl, fontsize=8, color=C_INK)
            mx = mc.max() if mc.max() > 0 else 1
            for i in range(k):
                for j in range(k):
                    tc = 'white' if mc[i,j] < mx*0.6 else 'black'
                    ax.text(j, i, str(mc[i,j]),
                            ha='center', va='center',
                            color=tc, fontsize=9, fontweight='bold')
            ax.set_title(f'MC — {nmet[mi]} (Robust)',
                         color=C_INK_DIM, fontsize=9,
                         fontfamily=FONT_MONO, pad=6)
            ax.set_xlabel('Predicho', color=C_INK_DIM, fontsize=8)
            ax.set_ylabel('Real', color=C_INK_DIM, fontsize=8)
            ax.tick_params(colors=C_INK_FAINT)
            for sp in ax.spines.values(): sp.set_edgecolor(C_BORDER)
            plt.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(
                colors=C_INK_DIM, labelcolor=C_INK_DIM)
        fig2.suptitle('Matrices de Confusión',
                      color=C_INK, fontsize=11, fontfamily=FONT_MONO)
        fig2.tight_layout(pad=1.5)
        c2 = FigureCanvas(fig2); c2.setStyleSheet(f'background:{C_BG};')
        lay.addWidget(c2); c2.draw(); plt.close(fig2)

        # ── Tabla de resultados ───────────────────────────────
        tabla = QTableWidget(9, 3)
        tabla.setHorizontalHeaderLabels(['Clasificador','Modo','Precisión'])
        tabla.setMaximumHeight(210)
        tabla.setStyleSheet(
            f"QTableWidget{{background:{C_SURFACE};color:{C_INK};"
            f"gridline-color:{C_BORDER};font-size:10px;"
            f"font-family:'{FONT_MONO}';border:1px solid {C_BORDER};"
            f"border-radius:8px;}}"
            f"QHeaderView::section{{background:{C_PANEL};color:{C_INK_DIM};"
            f"font-size:9px;letter-spacing:1px;padding:5px;border:none;"
            f"border-bottom:1px solid {C_BORDER};}}"
            f"QTableWidget::item:selected{{background:{C_ACCENT}33;}}")
        tabla.horizontalHeader().setStretchLastSection(True)
        tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        tabla.verticalHeader().setVisible(False)
        row = 0
        for mi,mn in enumerate(nmet):
            for mo,mon in enumerate(nmodo):
                acc = self.accs_mat[mi][mo]
                for col,val in enumerate([mn,mon,f'{acc:.1f}%']):
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setBackground(QColor(C_SURFACE if row%2==0 else C_CARD))
                    if col == 2:
                        if acc >= 70: item.setForeground(QColor('#4ade80'))
                        elif acc >= 50: item.setForeground(QColor(C_SUN))
                        else: item.setForeground(QColor(C_STORM))
                    tabla.setItem(row, col, item)
                row += 1
        lay.addWidget(tabla)

    def _show_galeria(self):
        self._limpiar(self.tab_gal)
        if not self.archivos: return
        k   = self.resultado['k']
        lay = self.tab_gal.layout()

        # Controles filtro
        ctrl = QFrame()
        ctrl.setStyleSheet(
            f'background:{C_PANEL};border-radius:8px;border:1px solid {C_BORDER};')
        cl = QHBoxLayout(ctrl); cl.setContentsMargins(14,8,14,8); cl.setSpacing(12)
        cl.addWidget(QLabel('Filtrar:'))
        self.combo_gal = QComboBox()
        self.combo_gal.setStyleSheet(
            f'QComboBox{{{style_input()}}}'
            f'QComboBox::drop-down{{border:none;}}'
            f'QComboBox QAbstractItemView{{background:{C_CARD};color:{C_INK};'
            f'selection-background-color:{C_ACCENT};}}')
        self.combo_gal.setMinimumWidth(180)
        self.combo_gal.addItem('Todas', -1)
        for i in range(k):
            nombre = self.asignacion.get(i, self.nombres_clase.get(i,f'C{i}'))
            self.combo_gal.addItem(nombre, i)
        self.combo_gal.currentIndexChanged.connect(self._filtrar_gal)
        cl.addWidget(self.combo_gal)
        self.lbl_gal_cnt = QLabel(f'{len(self.archivos)} imágenes')
        self.lbl_gal_cnt.setStyleSheet(
            f"color:{C_ACCENT};font-weight:bold;font-family:'{FONT_MONO}';border:none;")
        cl.addWidget(self.lbl_gal_cnt); cl.addStretch()
        lay.addWidget(ctrl)

        # Scroll area — TODAS las imágenes
        self.scroll_gal = QScrollArea()
        self.scroll_gal.setWidgetResizable(True)
        self.scroll_gal.setStyleSheet(f'background:{C_BG};border:none;')
        self.scroll_gal.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_gal.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gal_w    = QWidget(); self.gal_w.setStyleSheet(f'background:{C_BG};')
        self.gal_grid = QGridLayout(self.gal_w)
        self.gal_grid.setSpacing(8); self.gal_grid.setContentsMargins(4,4,4,4)
        self.scroll_gal.setWidget(self.gal_w)
        lay.addWidget(self.scroll_gal, 1)
        self._filtrar_gal()

    def _filtrar_gal(self):
        while self.gal_grid.count():
            item = self.gal_grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        filtro = self.combo_gal.currentData()
        cols   = 9; row_g = 0; col_g = 0; total = 0

        for i, path in enumerate(self.archivos):
            raw = int(self.etqs_raw[i])
            if filtro != -1 and raw != filtro: continue
            nombre_c = self.asignacion.get(raw, self.nombres_clase.get(raw,f'C{raw}'))
            try: idx_n = list(self.nombres_clase.values()).index(nombre_c)
            except ValueError: idx_n = raw

            card = ImageCard(path, idx_n, nombre_c)
            card.clicked.connect(self._abrir_visor)
            self.gal_grid.addWidget(card, row_g, col_g)
            col_g += 1; total += 1
            if col_g >= cols: col_g = 0; row_g += 1

        self.lbl_gal_cnt.setText(f'{total} imágenes')

    def _abrir_visor(self, img_path, clase_idx):
        nombre_c = self.nombres_clase.get(clase_idx, f'C{clase_idx}')
        dlg = ImageViewer(
            img_path, clase_idx, nombre_c,
            medias    = self.medias_clf,
            inv_covs  = self.inv_covs_clf,
            det_covs  = self.det_covs_clf,
            nombres_clase = self.nombres_clase,
            parent = self)
        dlg.exec_()

    # ── GUARDAR ───────────────────────────────────────────────

    def guardar_txt(self):
        """
        Exporta CSV completo con TODAS las features (100) + derivadas +
        similitudes + clasificación por 3 métodos + detección outlier.
        Una fila por imagen, separador ';', utf-8-sig para Excel.
        """
        if not self.resultado:
            QMessageBox.warning(self,'Aviso','Ejecuta primero.'); return
        path,_ = QFileDialog.getSaveFileName(
            self, 'Guardar tabla CSV',
            'sky_classifier_completo.csv', 'CSV (*.csv)')
        if not path: return

        k = self.resultado['k']
        X = self.resultado['X']

        # ── Nombres ENTENDIBLES de las 100 columnas ───────────
        col_names = []
        # Histogramas RGB → "Cantidad de pixeles ROJO nivel N"
        for canal, nom in [('R','Rojo'),('G','Verde'),('B','Azul')]:
            for bi in range(16):
                rango_ini = bi*16; rango_fin = (bi+1)*16-1
                col_names.append(f'Pixeles {nom} ({rango_ini}-{rango_fin})')
        # Medias
        for canal, nom in [('R','Rojo'),('G','Verde'),('B','Azul')]:
            col_names.append(f'Brillo promedio {nom}')
        for canal, nom in [('R','Rojo'),('G','Verde'),('B','Azul')]:
            col_names.append(f'Variacion {nom}')
        # Ratios de color globales
        col_names += [
            'Porcentaje Azul (toda la imagen)',
            'Porcentaje Gris (toda la imagen)',
            'Porcentaje Blanco (toda la imagen)',
            'Porcentaje Verde (toda la imagen)',
            'Porcentaje Oscuro (toda la imagen)',
            'Saturacion de color promedio',
            'Contraste general']
        # Zona superior
        col_names += [
            'Porcentaje Azul (zona del cielo)',
            'Porcentaje Blanco (zona del cielo)',
            'Porcentaje Gris (zona del cielo)']
        # Gradiente/textura
        col_names += [
            'Nivel de detalle general',
            'Nivel de detalle (zona superior)',
            'Nivel de detalle (zona media)',
            'Nivel de detalle (zona inferior)',
            'Variacion del detalle',
            'Relacion textura arriba/abajo',
            'Suavidad del cielo',
            'Relacion bordes horiz/vert']
        # Score cielo
        col_names += [
            'Probabilidad de cielo (general)',
            'Probabilidad de cielo (zona superior)']
        # LBP
        col_names += [
            'Textura promedio (LBP)',
            'Variacion de textura (LBP)',
            'Complejidad de textura (LBP)',
            'Uniformidad de textura (LBP)']
        # Luminancia sup
        col_names += [
            'Complejidad de brillo (cielo)',
            'Porcentaje muy brillante (cielo)',
            'Porcentaje brillo medio (cielo)',
            'Porcentaje oscuro (cielo)']
        # HSV global
        col_names += [
            'Matiz de color promedio',
            'Variacion del matiz',
            'Saturacion promedio (HSV)',
            'Variacion saturacion (HSV)',
            'Luminosidad promedio (HSV)',
            'Variacion luminosidad (HSV)']
        # HSV sup
        col_names += [
            'Matiz del cielo',
            'Variacion matiz del cielo',
            'Saturacion del cielo',
            'Luminosidad del cielo',
            'Variacion saturacion cielo',
            'Variacion luminosidad cielo']
        # Cues
        col_names += [
            'Deteccion azul-cielo (HSV)',
            'Deteccion nube-blanca (HSV)',
            'Deteccion nube-gris (HSV)',
            'Dominancia canal azul (cielo)',
            'Uniformidad del cielo',
            'Intensidad color del cielo']

        # ── Columnas derivadas con nombres claros ─────────────
        derivadas = [
            'Relacion Azul/Rojo',
            'Relacion Azul/Verde',
            'Relacion Azul/(Rojo+Verde)',
            'Diferencia detalle arriba-abajo',
            'Relacion detalle abajo/arriba',
            'Proporcion de nubes en cielo',
            'Pureza del azul del cielo',
            'Indice de brillo del cielo',
            'Indice de textura combinada',
            'Contraste vertical cielo-suelo',
            'Pureza azul (saturado)',
            'Dispersion general de color',
            'PUNTAJE clase Despejado',
            'PUNTAJE clase Semi-Nublado',
            'PUNTAJE clase Nublado',
            'PUNTAJE clase No-relacionado',
            'CLASE con mayor puntaje',
            'Margen de confianza (1ro-2do)',
            'Distancia estadistica minima',
            'Es imagen externa (1=si)',
            'Similitud a su clase (%)',
            'Confianza metodo Euclidiano (%)',
            'Confianza metodo Mahalanobis (%)',
            'Confianza metodo Bayes (%)',
            'Prediccion metodo Euclidiano',
            'Prediccion metodo Mahalanobis',
            'Prediccion metodo Bayes',
            'Metodos que coinciden (de 3)']

        id_cols = ['Nombre de archivo', 'Numero de cluster',
                   'Clase asignada', 'Indice']
        header  = id_cols + col_names + derivadas

        umbral_m = getattr(self, 'umbral_mahal', None)

        with open(path,'w',encoding='utf-8-sig') as f:
            f.write(';'.join(header) + '\n')
            for i, p in enumerate(self.archivos):
                cid    = int(self.etqs_raw[i])
                nombre = self.asignacion.get(cid, f'C{cid}')
                fn     = os.path.basename(p)
                ft     = X[i]

                # Derivadas básicas
                mr,mg,mb = ft[48],ft[49],ft[50]
                gs,gi    = ft[65],ft[67]
                ratio_BR = mb/(mr+1e-6)
                ratio_BG = mb/(mg+1e-6)
                ratio_BRG= mb/(mr+mg+1e-6)
                diff_grad= gs-gi
                ratio_gis= gi/(gs+1e-6)

                # HSV derivadas (si 100 features)
                if len(ft) >= 100:
                    az_s = ft[94]; bl_s = ft[95]; gr_s = ft[96]
                    val_s = ft[91]; sat_s = ft[99]
                else:
                    az_s = ft[61]; bl_s = ft[62]; gr_s = ft[63]
                    val_s = mb; sat_s = ft[59]
                nube_ratio = bl_s/(az_s+bl_s+1e-6)
                azuleza    = az_s/(az_s+bl_s+gr_s+1e-6)
                idx_brillo = val_s
                idx_textura= ft[76]*ft[68]
                contr_vert = abs(val_s - mb)
                pureza_az  = sat_s * az_s
                disp_color = (ft[51]+ft[52]+ft[53])/3.0

                # Scores de clase
                sc = score_clase(ft)
                sd,ss,sn,sr = sc.get(0,0),sc.get(1,0),sc.get(2,0),sc.get(3,0)
                scores_ord = sorted([sd,ss,sn,sr], reverse=True)
                ganador = ['Despejado','Semi-Nublado','Nublado','No relacionado'][
                    np.argmax([sd,ss,sn,sr])]
                margen = scores_ord[0]-scores_ord[1]

                # Distancia Mahalanobis y outlier
                dist_min = 0.0; es_out = 0
                sim_clase = 0.0
                if self.medias_clf:
                    eo, _, dist_min = detectar_outlier(
                        ft, self.medias_clf, self.inv_covs_clf,
                        self.det_covs_clf, umbral_mahal=umbral_m)
                    es_out = 1 if eo else 0
                    # similitud a su clase asignada
                    cl_idx = int(self.etqs_map[i])
                    sim_clase = similitud_a_clase(ft, self.medias_clf, cl_idx)

                # Clasificación 3 métodos
                preds=[]; confs=[]
                if self.medias_clf:
                    nc=len(self.medias_clf)
                    for mi in range(3):
                        try:
                            cl,cf = clasificar_punto(
                                ft,self.medias_clf,self.inv_covs_clf,
                                self.det_covs_clf,mi,nc)
                            preds.append(self.nombres_clase.get(cl,f'C{cl}'))
                            confs.append(cf)
                        except:
                            preds.append('-'); confs.append(0.0)
                else:
                    preds=['-','-','-']; confs=[0,0,0]
                # concordancia
                concord = len(set(preds)) == 1 and 3 or (
                    3 - len(set(preds)) + 1)
                concord = sum(1 for x in preds if x == max(set(preds), key=preds.count))

                row  = [fn, str(cid), nombre, str(i)]
                row += [f'{v:.6f}' for v in ft]
                row += [f'{ratio_BR:.4f}',f'{ratio_BG:.4f}',f'{ratio_BRG:.4f}',
                        f'{diff_grad:.6f}',f'{ratio_gis:.4f}',
                        f'{nube_ratio:.4f}',f'{azuleza:.4f}',
                        f'{idx_brillo:.4f}',f'{idx_textura:.6f}',
                        f'{contr_vert:.4f}',f'{pureza_az:.4f}',
                        f'{disp_color:.4f}',
                        f'{sd:.4f}',f'{ss:.4f}',f'{sn:.4f}',f'{sr:.4f}',
                        ganador,f'{margen:.4f}',
                        f'{dist_min:.4f}',str(es_out),f'{sim_clase:.2f}',
                        f'{confs[0]:.1f}',f'{confs[1]:.1f}',f'{confs[2]:.1f}',
                        preds[0],preds[1],preds[2],str(concord)]
                f.write(';'.join(row) + '\n')

        n_cols=len(header); n_rows=len(self.archivos)
        self.log(f'💾 CSV completo: {path}  ({n_rows} × {n_cols} cols)')
        QMessageBox.information(
            self,'Guardado',
            f'Tabla completa guardada:\n{path}\n\n'
            f'{n_rows} imágenes × {n_cols} columnas\n\n'
            f'Incluye: 100 features + 30 derivadas\n'
            f'(scores, similitudes, distancias, predicciones)\n\n'
            f'→ Excel: Datos → Desde texto/CSV → separador ";"')

    def guardar_pdf(self):
        if not self.resultado:
            QMessageBox.warning(self,'Aviso','Ejecuta primero.'); return
        path,_ = QFileDialog.getSaveFileName(
            self,'Guardar PDF','sky_classifier.pdf','PDF (*.pdf)')
        if not path: return
        k           = self.resultado['k']
        X           = self.resultado['X']
        etqs        = self.etqs_map
        nmet        = ['Euclidiana','Mahalanobis','Bayes']
        nmodo       = ['Robust','Cross-Val','LOO']
        labels_cl   = [self.nombres_clase.get(i,f'C{i}') for i in range(k)]
        cols_plt    = plt.cm.plasma(np.linspace(0.1,0.9,k))

        with PdfPages(path) as pdf:
            # Portada
            fig=plt.figure(figsize=(16,9)); fig.patch.set_facecolor(C_BG)
            ax=fig.add_axes([0,0,1,1]); ax.set_facecolor(C_BG); ax.axis('off')
            for i,a in enumerate(np.linspace(0,0.15,40)):
                ax.add_patch(plt.Rectangle((0,i/40),1,1/40,
                    transform=ax.transAxes,facecolor=C_ACCENT,alpha=a))
            fig.text(0.5,0.76,'SKY CLASSIFIER',ha='center',fontsize=30,
                     fontweight='bold',color=C_INK,fontfamily=FONT_MONO)
            fig.text(0.5,0.66,
                     'K-Means++ · LBP · Score Cielo · Euclid · Mahal · Bayes',
                     ha='center',fontsize=12,color=C_INK_DIM)
            stats=[('IMÁGENES',str(self.resultado['n_ok'])),
                   ('CLUSTERS',str(k)),
                   ('INERCIA',f'{self.resultado["inercia"]:.0f}')]
            for idx,(k2,v) in enumerate(stats):
                x=0.25+idx*0.25
                fig.add_artist(plt.Rectangle((x-0.08,0.35),0.16,0.15,
                    transform=fig.transFigure,facecolor=C_PANEL,
                    edgecolor=C_ACCENT,linewidth=1.5,zorder=2))
                fig.text(x,0.46,v,ha='center',fontsize=20,fontweight='bold',
                         color=C_SKY,zorder=3,fontfamily=FONT_MONO)
                fig.text(x,0.37,k2,ha='center',fontsize=9,
                         color=C_INK_DIM,zorder=3,fontfamily=FONT_MONO)
            pdf.savefig(fig,facecolor=C_BG,bbox_inches='tight'); plt.close(fig)

            # PÁG 1: Clusters 3D
            fig=plt.figure(figsize=(18,8)); fig.patch.set_facecolor(C_BG)
            Xc=X-X.mean(axis=0); _,vv=np.linalg.eigh(np.cov(Xc.T)); X3d=Xc@vv[:,-3:]
            ax3=fig.add_subplot(121,projection='3d'); ax3.set_facecolor('#0a0a12')
            ax3.xaxis.pane.fill=ax3.yaxis.pane.fill=ax3.zaxis.pane.fill=False
            for i in range(k):
                mask=etqs==i
                if not mask.any(): continue
                ax3.scatter(X3d[mask,0],X3d[mask,1],X3d[mask,2],
                            color=cols_plt[i],s=12,alpha=0.6,
                            label=self.nombres_clase.get(i,f'C{i}'))
            ax3.set_title('PCA 3D',color=C_INK,fontsize=10)
            ax3.legend(fontsize=8,facecolor=C_PANEL,labelcolor=C_INK)
            ax3.view_init(elev=25,azim=45)
            ax2=fig.add_subplot(122); ax2.set_facecolor('#0a0a12')
            counts=[int(np.sum(etqs==i)) for i in range(k)]
            ax2.bar(range(k),counts,color=[cols_plt[i] for i in range(k)],edgecolor='none')
            for i,cnt in enumerate(counts):
                ax2.text(i,cnt+0.5,str(cnt),ha='center',color=C_INK,fontsize=9,fontweight='bold')
            ax2.set_xticks(range(k)); ax2.set_xticklabels(labels_cl,rotation=20,color=C_INK)
            ax2.tick_params(colors=C_INK_FAINT); ax2.set_facecolor('#0a0a12')
            for sp in ax2.spines.values(): sp.set_edgecolor(C_BORDER)
            fig.suptitle('Análisis de Clusters',color=C_INK,fontsize=14,fontweight='bold')
            fig.tight_layout(pad=2)
            pdf.savefig(fig,facecolor=C_BG,bbox_inches='tight'); plt.close(fig)

            # PÁG 2: Segmentación
            muestras=[(i,self.archivos[np.where(self.etqs_raw==i)[0][0]])
                      for i in range(k) if np.any(self.etqs_raw==i)]
            fig,axes_seg=plt.subplots(len(muestras),2,
                                       figsize=(12,4*len(muestras)))
            fig.patch.set_facecolor(C_BG)
            if len(muestras)==1: axes_seg=[axes_seg]
            for ri,(cid,ip) in enumerate(muestras):
                nombre_c=self.asignacion.get(cid,f'C{cid}')
                arr_s,mask,overlay,score,_=segmentar_cielo(ip)
                if arr_s is None: continue
                axes_seg[ri][0].imshow(arr_s); axes_seg[ri][0].axis('off')
                axes_seg[ri][0].set_title(f'Original — {nombre_c}',
                    color=C_INK,fontsize=10)
                axes_seg[ri][1].imshow(overlay); axes_seg[ri][1].axis('off')
                pct=float(mask.mean())*100 if mask is not None else 0
                axes_seg[ri][1].set_title(f'Cielo detectado: {pct:.1f}%',
                    color=C_SKY,fontsize=10)
            fig.suptitle('Segmentación del Cielo',color=C_INK,
                         fontsize=14,fontweight='bold')
            fig.tight_layout(pad=2)
            pdf.savefig(fig,facecolor=C_BG,bbox_inches='tight'); plt.close(fig)

            # PÁG 3: Precisiones
            fig,ax_b=plt.subplots(figsize=(14,6)); fig.patch.set_facecolor(C_BG)
            ax_b.set_facecolor('#0a0a12')
            x=np.arange(3); w=0.24; cm=[C_SKY,C_ACCENT,'#4ade80']
            for mo in range(3):
                vals=[self.accs_mat[mi][mo] for mi in range(3)]
                bars=ax_b.bar(x+mo*w,vals,w,label=nmodo[mo],
                              color=cm[mo],edgecolor='none',alpha=0.85)
                for bar,val in zip(bars,vals):
                    ax_b.text(bar.get_x()+bar.get_width()/2,
                              bar.get_height()+0.5,f'{val:.0f}%',
                              ha='center',color=C_INK,fontsize=9)
            ax_b.set_xticks(x+w); ax_b.set_xticklabels(nmet,color=C_INK,fontsize=12)
            ax_b.set_ylim(0,115); ax_b.legend(fontsize=10,facecolor=C_PANEL,labelcolor=C_INK)
            ax_b.set_title('Precisión por Clasificador',color=C_INK,fontsize=13,fontweight='bold')
            ax_b.tick_params(colors=C_INK_FAINT)
            ax_b.grid(axis='y',alpha=0.06,color='white',linestyle='--')
            for sp in ax_b.spines.values(): sp.set_edgecolor(C_BORDER)
            fig.tight_layout(pad=2)
            pdf.savefig(fig,facecolor=C_BG,bbox_inches='tight'); plt.close(fig)

            # PÁG 4: Matrices de confusión
            fig,axes=plt.subplots(1,3,figsize=(18,6)); fig.patch.set_facecolor(C_BG)
            for mi in range(3):
                axm=axes[mi]; axm.set_facecolor('#0a0a12')
                mc=self.mcs_mat[mi][0]
                im=axm.imshow(mc,cmap='viridis',aspect='auto')
                axm.set_xticks(range(k)); axm.set_yticks(range(k))
                axm.set_xticklabels(labels_cl,rotation=30,ha='right',fontsize=9,color=C_INK)
                axm.set_yticklabels(labels_cl,fontsize=9,color=C_INK)
                mxv=mc.max() if mc.max()>0 else 1
                for i in range(k):
                    for j in range(k):
                        tc='white' if mc[i,j]<mxv*0.6 else 'black'
                        axm.text(j,i,str(mc[i,j]),ha='center',va='center',
                                 color=tc,fontsize=10,fontweight='bold')
                axm.set_title(f'MC — {nmet[mi]}',color=C_INK_DIM,fontsize=11)
                axm.tick_params(colors=C_INK_FAINT)
                plt.colorbar(im,ax=axm,shrink=0.8).ax.tick_params(
                    colors=C_INK_DIM,labelcolor=C_INK_DIM)
            fig.suptitle('Matrices de Confusión',color=C_INK,fontsize=14,fontweight='bold')
            fig.tight_layout(pad=2)
            pdf.savefig(fig,facecolor=C_BG,bbox_inches='tight'); plt.close(fig)

        self.log(f'📄 PDF: {path}')
        QMessageBox.information(self,'Guardado',f'PDF guardado:\n{path}')

    def guardar_html(self):
        """Genera un reporte HTML profesional completo y autocontenido."""
        if not self.resultado:
            QMessageBox.warning(self,'Aviso','Ejecuta primero.'); return
        path,_ = QFileDialog.getSaveFileName(
            self,'Guardar reporte HTML','sky_classifier_reporte.html','HTML (*.html)')
        if not path: return

        import base64, io, datetime
        k = self.resultado['k']
        X = self.resultado['X']
        etqs = self.etqs_map
        fecha = datetime.datetime.now().strftime('%d/%m/%Y a las %H:%M')
        total = len(self.archivos)

        # ── Función para incrustar figuras matplotlib como base64 ──
        def fig_a_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', facecolor=C_BG,
                        bbox_inches='tight', dpi=90)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return f'data:image/png;base64,{b64}'

        # ── Generar gráficas ──────────────────────────────────
        cols = plt.cm.plasma(np.linspace(0.1,0.9,k))
        counts = [int(np.sum(etqs==i)) for i in range(k)]
        labels = [self.nombres_clase.get(i,f'C{i}') for i in range(k)]

        # Gráfica distribución
        fig_d = plt.figure(figsize=(9,4)); fig_d.patch.set_facecolor(C_BG)
        axd = fig_d.add_subplot(111); axd.set_facecolor('#0a0a12')
        bars = axd.bar(range(k), counts, color=[cols[i] for i in range(k)],
                       edgecolor='none')
        for i,(b,c) in enumerate(zip(bars,counts)):
            axd.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                     f'{c}\n{c/total*100:.0f}%', ha='center', color=C_INK,
                     fontsize=9, fontweight='bold')
        axd.set_xticks(range(k)); axd.set_xticklabels(labels, color=C_INK, fontsize=9)
        axd.tick_params(colors=C_INK_FAINT); axd.set_facecolor('#0a0a12')
        for sp in axd.spines.values(): sp.set_edgecolor(C_BORDER)
        axd.spines['top'].set_visible(False); axd.spines['right'].set_visible(False)
        axd.set_title('Distribución de imágenes por clase', color=C_INK, fontsize=11)
        img_dist = fig_a_base64(fig_d)

        # Gráfica clusters 3D (proyección 2D para HTML)
        fig_c = plt.figure(figsize=(9,4)); fig_c.patch.set_facecolor(C_BG)
        axc = fig_c.add_subplot(111); axc.set_facecolor('#0a0a12')
        Xc = X - X.mean(axis=0)
        _, vv = np.linalg.eigh(np.cov(Xc.T))
        X2d = Xc @ vv[:, -2:]
        for i in range(k):
            m = etqs==i
            if m.any():
                axc.scatter(X2d[m,0], X2d[m,1], color=cols[i], s=14,
                            alpha=0.6, label=labels[i])
        axc.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_INK, edgecolor=C_BORDER)
        axc.set_xlabel('Componente 1', color=C_INK_DIM, fontsize=9)
        axc.set_ylabel('Componente 2', color=C_INK_DIM, fontsize=9)
        axc.tick_params(colors=C_INK_FAINT)
        for sp in axc.spines.values(): sp.set_edgecolor(C_BORDER)
        axc.set_title('Separación de clases (PCA 2D)', color=C_INK, fontsize=11)
        img_pca = fig_a_base64(fig_c)

        # Gráfica precisiones
        img_prec = ''
        if self.accs_mat:
            fig_p = plt.figure(figsize=(9,4)); fig_p.patch.set_facecolor(C_BG)
            axp = fig_p.add_subplot(111); axp.set_facecolor('#0a0a12')
            nmet=['Euclidiana','Mahalanobis','Bayes']; nmodo=['Robust','Cross-Val','LOO']
            x=np.arange(3); w=0.25; cm=[C_SKY,C_ACCENT,'#4ade80']
            for mo in range(3):
                vals=[self.accs_mat[mi][mo] for mi in range(3)]
                axp.bar(x+mo*w, vals, w, label=nmodo[mo], color=cm[mo], alpha=0.9)
            axp.set_xticks(x+w); axp.set_xticklabels(nmet, color=C_INK, fontsize=10)
            axp.legend(fontsize=9, facecolor=C_PANEL, labelcolor=C_INK, edgecolor=C_BORDER)
            axp.set_ylim(0,115); axp.tick_params(colors=C_INK_FAINT)
            for sp in axp.spines.values(): sp.set_edgecolor(C_BORDER)
            axp.spines['top'].set_visible(False); axp.spines['right'].set_visible(False)
            axp.set_title('Precisión por clasificador y método', color=C_INK, fontsize=11)
            img_prec = fig_a_base64(fig_p)

        # ── Tablas HTML ───────────────────────────────────────
        filas_dist = ''
        for i in range(k):
            cnt = counts[i]; nombre = labels[i]; pct = cnt/total*100 if total else 0
            col = C_CLASSES.get(i, '#888')
            filas_dist += (
                f'<tr><td><span style="color:{col};font-size:18px">●</span> {nombre}</td>'
                f'<td>{cnt}</td><td>{pct:.1f}%</td>'
                f'<td><div class="bar" style="background:{col};width:{pct*2.5}px"></div></td></tr>')

        filas_prec = ''
        mejor_acc = 0; mejor_combo = ''
        if self.accs_mat:
            nmet=['Euclidiana','Mahalanobis','Bayes']; nmodo=['Robust','Cross-Val','LOO']
            for mi,mn in enumerate(nmet):
                for mo,mon in enumerate(nmodo):
                    acc=self.accs_mat[mi][mo]
                    if acc>mejor_acc: mejor_acc=acc; mejor_combo=f'{mn} + {mon}'
                    col='#4ade80' if acc>=70 else ('#ffd54f' if acc>=50 else '#ef5350')
                    estrella = ' ⭐' if acc==max(self.accs_mat[a][b] for a in range(3) for b in range(3)) else ''
                    filas_prec += (f'<tr><td>{mn}</td><td>{mon}</td>'
                                   f'<td style="color:{col};font-weight:bold">{acc:.1f}%{estrella}</td></tr>')

        filas_f1 = ''
        mc = getattr(self,'metricas_clase',{})
        for ci in sorted(mc.keys()):
            m = mc[ci]; nombre = labels[ci] if ci < len(labels) else f'C{ci}'
            col_f1 = '#4ade80' if m['f1']>=70 else ('#ffd54f' if m['f1']>=50 else '#ef5350')
            filas_f1 += (f'<tr><td>{nombre}</td><td>{m["precision"]:.1f}%</td>'
                         f'<td>{m["recall"]:.1f}%</td>'
                         f'<td style="color:{col_f1};font-weight:bold">{m["f1"]:.1f}%</td>'
                         f'<td>{m["soporte"]}</td></tr>')

        # Tabla importancia features (top 12)
        filas_imp = ''
        fi = getattr(self,'feat_importance',None)
        if fi is not None:
            nombres_f = self._nombres_features_legibles()
            top = np.argsort(fi)[::-1][:12]
            max_imp = fi[top[0]] if len(top)>0 else 1
            for rank, idx in enumerate(top, 1):
                nom = nombres_f[idx] if idx < len(nombres_f) else f'feature {idx}'
                barw = fi[idx]/max_imp*200
                filas_imp += (f'<tr><td>{rank}</td><td>{nom}</td>'
                              f'<td>{fi[idx]:.1f}</td>'
                              f'<td><div class="bar" style="background:#7c6af4;width:{barw}px"></div></td></tr>')

        sil = getattr(self,'m_silhouette',0)
        calidad = ('Excelente' if sil>0.5 else 'Buena' if sil>0.35
                   else 'Razonable' if sil>0.2 else 'Débil')
        f1_prom = np.mean([mc[c]['f1'] for c in mc]) if mc else 0
        clase_may = labels[int(np.argmax(counts))]

        html = f'''<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sky Classifier — Reporte de Análisis</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0a0a0f;
       color:#e8e8f0; padding:40px; line-height:1.6; }}
.container {{ max-width:1100px; margin:0 auto; }}
header {{ border-bottom:3px solid #7c6af4; padding-bottom:24px; margin-bottom:32px; }}
h1 {{ font-size:36px; letter-spacing:4px; color:#e8e8f0;
     font-family:'Courier New',monospace; }}
.sub {{ color:#7070a0; font-size:14px; margin-top:8px; }}
.badge {{ display:inline-block; background:#1c1c28; border:1px solid #2a2a3a;
         border-radius:20px; padding:4px 14px; font-size:11px; color:#4fc3f7;
         margin-top:12px; margin-right:8px; font-family:'Courier New',monospace; }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:28px 0; }}
.card {{ background:#1c1c28; border:1px solid #2a2a3a; border-top:4px solid #4fc3f7;
        border-radius:12px; padding:20px; }}
.card .v {{ font-size:32px; font-weight:bold; color:#4fc3f7;
           font-family:'Courier New',monospace; }}
.card .l {{ font-size:10px; color:#7070a0; letter-spacing:1px; margin-top:6px; }}
.card .d {{ font-size:11px; color:#404060; margin-top:2px; }}
h2 {{ font-size:16px; letter-spacing:2px; color:#e8e8f0; margin:40px 0 16px;
     font-family:'Courier New',monospace; border-left:4px solid #7c6af4;
     padding-left:14px; }}
h3 {{ font-size:13px; color:#7070a0; margin:20px 0 10px;
     font-family:'Courier New',monospace; letter-spacing:1px; }}
table {{ width:100%; border-collapse:collapse; background:#161620;
        border-radius:10px; overflow:hidden; margin-bottom:20px; }}
th {{ background:#1c1c28; color:#7070a0; padding:12px 16px; text-align:left;
     font-size:11px; letter-spacing:1px; }}
td {{ padding:11px 16px; border-top:1px solid #2a2a3a; font-size:13px; }}
.bar {{ height:16px; border-radius:4px; }}
.methods {{ background:#161620; border:1px solid #2a2a3a; border-radius:10px;
           padding:20px; font-size:13px; color:#b0bec5; }}
.methods b {{ color:#4fc3f7; }}
.methods p {{ margin-bottom:12px; }}
.interp {{ background:#161620; border:1px solid #2a2a3a; border-left:4px solid #4ade80;
          border-radius:10px; padding:20px; font-size:14px; margin-bottom:20px; }}
.interp b {{ color:#4fc3f7; }}
.imgbox {{ background:#161620; border:1px solid #2a2a3a; border-radius:10px;
          padding:16px; margin-bottom:20px; text-align:center; }}
.imgbox img {{ max-width:100%; border-radius:6px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
.note {{ font-size:11px; color:#404060; font-style:italic; margin-top:8px; }}
footer {{ margin-top:50px; color:#404060; font-size:11px; text-align:center;
         border-top:1px solid #2a2a3a; padding-top:24px; }}
.tag {{ display:inline-block; padding:2px 10px; border-radius:6px; font-size:11px;
       font-weight:bold; }}
@media(max-width:800px){{ .cards,.grid2{{grid-template-columns:1fr 1fr;}} }}
</style></head><body><div class="container">

<header>
<h1>☁ SKY CLASSIFIER</h1>
<div class="sub">Reporte completo de análisis y clasificación de cielos</div>
<div class="badge">Generado: {fecha}</div>
<div class="badge">{self.resultado['n_ok']} imágenes</div>
<div class="badge">{k} clusters</div>
<div class="badge">100 características por imagen</div>
</header>

<div class="interp">
<b>Resumen:</b> Se analizaron <b>{self.resultado['n_ok']}</b> imágenes
(de {self.resultado['n_found']} encontradas) agrupándolas automáticamente en
<b>{k} clases</b> mediante K-Means. La clase predominante es <b>{clase_may}</b>
({max(counts)} imágenes, {max(counts)/total*100:.0f}%). El clasificador más preciso
fue <b>{mejor_combo}</b> alcanzando <b>{mejor_acc:.1f}%</b> de precisión. La calidad
de agrupamiento es <b>{calidad.lower()}</b> (Silhouette = {sil:.3f}), con un
F1-Score promedio de <b>{f1_prom:.1f}%</b>.
</div>

<h2>MÉTRICAS PRINCIPALES</h2>
<div class="cards">
<div class="card"><div class="v">{self.resultado['n_ok']}</div>
<div class="l">IMÁGENES</div><div class="d">de {self.resultado['n_found']} encontradas</div></div>
<div class="card" style="border-top-color:#4ade80"><div class="v" style="color:#4ade80">{mejor_acc:.1f}%</div>
<div class="l">PRECISIÓN MÁXIMA</div><div class="d">{mejor_combo}</div></div>
<div class="card" style="border-top-color:#7c6af4"><div class="v" style="color:#7c6af4">{sil:.3f}</div>
<div class="l">SILHOUETTE</div><div class="d">calidad {calidad.lower()}</div></div>
<div class="card" style="border-top-color:#ffd54f"><div class="v" style="color:#ffd54f">{f1_prom:.0f}%</div>
<div class="l">F1 PROMEDIO</div><div class="d">balance prec/recall</div></div>
</div>

<h2>DISTRIBUCIÓN DE CLASES</h2>
<div class="imgbox"><img src="{img_dist}" alt="distribución"></div>
<table><tr><th>Clase</th><th>Cantidad</th><th>Porcentaje</th><th>Visual</th></tr>
{filas_dist}</table>

<h2>SEPARACIÓN DE CLASES (PCA)</h2>
<div class="imgbox"><img src="{img_pca}" alt="pca">
<div class="note">Proyección de las 100 características a 2 dimensiones mediante
Análisis de Componentes Principales. Clusters bien separados indican clases distinguibles.</div></div>

<h2>RENDIMIENTO DE CLASIFICADORES</h2>
{'<div class="imgbox"><img src="'+img_prec+'" alt="precisiones"></div>' if img_prec else ''}
<table><tr><th>Método</th><th>Validación</th><th>Precisión</th></tr>
{filas_prec}</table>
<h3>EXPLICACIÓN DE LOS MÉTODOS DE EVALUACIÓN</h3>
<div class="methods">
<p><b>Robust (70/30):</b> entrena con 70% de las imágenes y prueba con el 30% restante.</p>
<p><b>Cross-Validation (k-fold):</b> divide los datos en k partes, rota cuál se usa
para probar, y promedia. Más robusto ante el azar.</p>
<p><b>Leave-One-Out (LOO):</b> entrena con todas las imágenes menos una, prueba con esa,
y repite para cada imagen. El más exhaustivo.</p>
</div>

<h2>CALIDAD POR CLASE</h2>
<table><tr><th>Clase</th><th>Precisión</th><th>Recall</th><th>F1-Score</th><th>Imágenes</th></tr>
{filas_f1}</table>
<div class="methods">
<p><b>Precisión:</b> de todas las imágenes que el sistema dijo que eran de esta clase,
qué porcentaje realmente lo eran.</p>
<p><b>Recall:</b> de todas las imágenes que realmente son de esta clase, qué porcentaje
detectó correctamente el sistema.</p>
<p><b>F1-Score:</b> media armónica entre precisión y recall. Un valor alto significa que
el sistema acierta y no se le escapan imágenes.</p>
</div>

<h2>CARACTERÍSTICAS MÁS IMPORTANTES</h2>
<p style="color:#7070a0;font-size:12px;margin-bottom:12px">
Las características que más ayudan a distinguir entre las clases, ordenadas por su
poder discriminante (calculado con análisis de varianza tipo ANOVA):</p>
<table><tr><th>#</th><th>Característica</th><th>Poder</th><th>Visual</th></tr>
{filas_imp}</table>

<h2>MÉTRICAS DE AGRUPAMIENTO (CLUSTERING)</h2>
<table>
<tr><th>Métrica</th><th>Valor</th><th>Interpretación</th></tr>
<tr><td>Silhouette Score</td><td><b>{sil:.3f}</b></td>
<td>Mide qué tan separados están los grupos (de -1 a 1). Mayor es mejor.
Resultado: <span class="tag" style="background:{'#14532d' if sil>0.35 else '#3d2a00'};color:{'#4ade80' if sil>0.35 else '#ffd54f'}">{calidad}</span></td></tr>
<tr><td>Davies-Bouldin</td><td><b>{getattr(self,'m_davies',0):.3f}</b></td>
<td>Relación entre dispersión interna y separación de clusters. Menor es mejor.</td></tr>
<tr><td>Calinski-Harabasz</td><td><b>{getattr(self,'m_calinski',0):.1f}</b></td>
<td>Razón de varianza entre grupos vs dentro de grupos. Mayor es mejor.</td></tr>
<tr><td>Inercia K-Means</td><td><b>{self.resultado['inercia']:.1f}</b></td>
<td>Suma de distancias de cada imagen a su centroide. Menor = grupos más compactos.</td></tr>
</table>

<h2>METODOLOGÍA COMPLETA</h2>
<div class="methods">
<p><b>1. Segmentación del cielo:</b> Se detecta qué parte de la imagen es cielo
combinando múltiples señales: análisis de color en espacio RGB (diferencia entre
canales), espacio HSV (matiz azul entre 180-270°, nubes blancas con saturación
baja), detección de bordes mediante filtro de paso-altas (el cielo es liso, los
edificios tienen mucho detalle), dominancia del canal azul, y posición vertical
(el cielo suele estar arriba). La máscara resultante se limpia con operaciones
morfológicas (rellenado de huecos, dilatación y erosión).</p>

<p><b>2. Extracción de características:</b> De cada imagen se calculan <b>100
descriptores numéricos</b>: 48 valores de histograma de color RGB, estadísticas de
brillo por canal, porcentajes de azul/blanco/gris/verde, medidas de textura (gradiente
y Local Binary Patterns), probabilidad de cielo, y 18 características del espacio de
color HSV (matiz, saturación y luminosidad por zonas).</p>

<p><b>3. Agrupamiento (clustering):</b> Las imágenes se agrupan con el algoritmo
<b>K-Means</b>, usando inicialización inteligente <b>K-Means++</b> con 10 reinicios
para encontrar la mejor solución. Cada grupo se asigna automáticamente a una clase
de cielo (Despejado, Semi-Nublado, Nublado) según el análisis de sus características.</p>

<p><b>4. Clasificación:</b> Se entrenan y comparan <b>tres clasificadores
estadísticos</b>: distancia Euclidiana (la más simple), distancia de Mahalanobis
(considera la forma de cada clase), y clasificador Bayesiano (probabilístico
gaussiano). Cada uno se evalúa con tres métodos de validación.</p>

<p><b>5. Detección de imágenes externas:</b> Si se carga una imagen que no es un
cielo, el sistema la marca como <b>"No relacionada"</b> midiendo su distancia
estadística de Mahalanobis. Si supera el umbral del percentil 97 del dataset, se
considera que no pertenece a ninguna clase de cielo.</p>

<p><b>6. Medición de similitud:</b> Entre imágenes se calcula similitud combinando
distancia coseno (orientación de los vectores de características) y distancia
euclidiana exponencial (cercanía absoluta).</p>
</div>

<footer>
Sky Classifier · Sistema de clasificación de cielos por Visión Artificial<br>
ESCOM — Instituto Politécnico Nacional · {fecha}
</footer>
</div></body></html>'''

        with open(path,'w',encoding='utf-8') as f:
            f.write(html)
        self.log(f'🌐 HTML completo: {path}')
        QMessageBox.information(self,'Guardado',
            f'Reporte HTML completo guardado:\n{path}\n\n'
            f'Incluye gráficas incrustadas, tablas, métricas y metodología.\n'
            f'Ábrelo en cualquier navegador.')

    def _nombres_features_legibles(self):
        """Nombres legibles de las 100 features para el reporte."""
        nombres = []
        for canal in ['Rojo','Verde','Azul']:
            for bi in range(16):
                nombres.append(f'Histograma {canal} nivel {bi}')
        for canal in ['Rojo','Verde','Azul']: nombres.append(f'Brillo promedio {canal}')
        for canal in ['Rojo','Verde','Azul']: nombres.append(f'Variación {canal}')
        nombres += ['Porcentaje Azul','Porcentaje Gris','Porcentaje Blanco',
                    'Porcentaje Verde','Porcentaje Oscuro','Saturación','Contraste']
        nombres += ['Azul en el cielo','Blanco en el cielo','Gris en el cielo']
        nombres += ['Detalle general','Detalle zona superior','Detalle zona media',
                    'Detalle zona inferior','Variación de detalle',
                    'Relación textura arriba/abajo','Suavidad del cielo','Bordes H/V']
        nombres += ['Probabilidad de cielo','Probabilidad cielo (arriba)']
        nombres += ['Textura promedio','Variación textura','Complejidad textura',
                    'Uniformidad textura']
        nombres += ['Complejidad brillo','Muy brillante (cielo)','Brillo medio (cielo)',
                    'Oscuro (cielo)']
        nombres += ['Matiz promedio','Variación matiz','Saturación HSV',
                    'Variación saturación','Luminosidad HSV','Variación luminosidad']
        nombres += ['Matiz del cielo','Variación matiz cielo','Saturación cielo',
                    'Luminosidad cielo','Var. saturación cielo','Var. luminosidad cielo']
        nombres += ['Detección azul-cielo','Detección nube-blanca','Detección nube-gris',
                    'Dominancia azul','Uniformidad cielo','Intensidad color cielo']
        return nombres


class DialogAsignacion(QDialog):
    def __init__(self, k, archivos, etqs, asignacion, nombres_clase, parent=None):
        super().__init__(parent)
        self.k = k
        self.archivos = archivos
        self.etqs = etqs
        # asignacion es {cid(int): nombre_clase(str)}
        self.asignacion = asignacion
        self.nombres_clase = nombres_clase
        self.combos = []
        self.setWindowTitle('Corregir asignación')
        self.setMinimumSize(840, 540)
        self.setStyleSheet(STYLE_BASE + f'QDialog{{background:{C_BG};}}')
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(12)

        hdr = QLabel('ASIGNACIÓN DE CLUSTERS')
        hdr.setStyleSheet(
            f"color:{C_INK_DIM};font-family:'{FONT_MONO}';font-size:10px;"
            f"letter-spacing:3px;border:none;")
        lay.addWidget(hdr)

        info = QLabel(
            'Asignación automática con K-Means++ + LBP + score de cielo. '
            'Cada cluster muestra dos imágenes de ejemplo para ayudarte a identificarlo.')
        info.setStyleSheet(f'color:{C_INK_FAINT};font-size:10px;border:none;')
        info.setWordWrap(True); lay.addWidget(info)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f'background:{C_SURFACE};border:1px solid {C_BORDER};')
        cont = QWidget(); cont.setStyleSheet(f'background:{C_SURFACE};')
        grid = QGridLayout(cont); grid.setSpacing(14)

        # Headers
        for col, txt in enumerate(
                ['CLUSTER', 'MUESTRA 1', 'MUESTRA 2', 'CANTIDAD', 'CLASE ASIGNADA']):
            h = QLabel(txt)
            h.setStyleSheet(
                f"color:{C_INK_DIM};font-family:'{FONT_MONO}';font-size:9px;"
                f"letter-spacing:1px;padding:4px;border:none;")
            grid.addWidget(h, 0, col)

        # Opciones de clase: siempre incluir las del usuario + No relacionado
        opciones = []
        for i in range(self.k):
            nombre = self.nombres_clase.get(i, NOMBRES_DEFAULT[i]
                      if i < len(NOMBRES_DEFAULT) else f'Clase {i}')
            if nombre not in opciones:
                opciones.append(nombre)
        # Asegurar que "No relacionado" siempre esté disponible
        if 'No relacionado' not in opciones:
            opciones.append('No relacionado')

        for row in range(self.k):
            # ID cluster
            col_c = C_CLASSES.get(row, C_ACCENT)
            lbl_id = QLabel(f'  C{row:02d}')
            lbl_id.setStyleSheet(
                f"color:{col_c};font-weight:bold;font-family:'{FONT_MONO}';"
                f"font-size:12px;padding:4px;border:none;")
            grid.addWidget(lbl_id, row + 1, 0)

            # Dos muestras — con .copy() para evitar crash de memoria
            idx_c = np.where(self.etqs == row)[0]
            for sc in range(2):
                lbl_img = QLabel()
                lbl_img.setFixedSize(104, 74)
                lbl_img.setStyleSheet(
                    f'border:1px solid {C_BORDER};border-radius:5px;'
                    f'background:{C_CARD};')
                lbl_img.setAlignment(Qt.AlignCenter)
                if len(idx_c) > sc:
                    try:
                        img = Image.open(
                            self.archivos[idx_c[sc]]
                        ).convert('RGB').resize((104, 74), Image.LANCZOS)
                        arr = np.array(img).copy()   # .copy() evita el crash
                        qi  = QImage(arr.data, 104, 74, 312,
                                     QImage.Format_RGB888)
                        lbl_img.setPixmap(
                            QPixmap.fromImage(qi.copy()))   # .copy() en QImage
                    except Exception:
                        lbl_img.setText('—')
                grid.addWidget(lbl_img, row + 1, sc + 1)

            # Cantidad
            lbl_cnt = QLabel(f'{len(idx_c)} imgs')
            lbl_cnt.setStyleSheet(
                f"color:{C_INK_DIM};font-family:'{FONT_MONO}';"
                f"font-size:10px;padding:4px;border:none;")
            grid.addWidget(lbl_cnt, row + 1, 3)

            # ComboBox
            combo = QComboBox()
            combo.addItems(opciones)
            combo.setStyleSheet(
                f'QComboBox{{{style_input()}}}'
                f'QComboBox::drop-down{{border:none;}}'
                f'QComboBox QAbstractItemView{{background:{C_CARD};'
                f'color:{C_INK};selection-background-color:{C_ACCENT};}}')

            # Pre-seleccionar la asignación actual
            # asignacion[row] es un string (nombre de clase)
            nombre_actual = self.asignacion.get(row, opciones[0])
            if nombre_actual in opciones:
                combo.setCurrentIndex(opciones.index(nombre_actual))
            else:
                combo.setCurrentIndex(0)

            self.combos.append(combo)
            grid.addWidget(combo, row + 1, 4)

        scroll.setWidget(cont); lay.addWidget(scroll, 1)

        # Botones
        fr = QWidget(); fr.setStyleSheet('background:transparent;')
        bl = QHBoxLayout(fr); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(10)
        bc = QPushButton('Cancelar')
        bc.setStyleSheet(style_btn(C_PANEL, C_INK_DIM))
        bc.setStyleSheet(
            bc.styleSheet() + f'QPushButton{{border:1px solid {C_BORDER};}}')
        bc.clicked.connect(self.reject)
        bok = QPushButton('✔  Confirmar asignación')
        bok.setStyleSheet(style_btn(C_ACCENT))
        bok.clicked.connect(self.accept)
        bl.addStretch(); bl.addWidget(bc); bl.addWidget(bok)
        lay.addWidget(fr)

    def get_asignacion(self):
        """Retorna {cid(int): clase_idx(int)}"""
        res = {}
        for i in range(self.k):
            nombre = self.combos[i].currentText()
            # Buscar en nombres_clase primero
            found = False
            for idx, nm in self.nombres_clase.items():
                if nm == nombre:
                    res[i] = idx; found = True; break
            if not found:
                # Buscar en NOMBRES_DEFAULT
                try:
                    res[i] = NOMBRES_DEFAULT.index(nombre)
                except ValueError:
                    res[i] = i
        return res


# ═══════════════════════════════════════════════════════════════
if __name__=='__main__':
    app=QApplication(sys.argv); app.setStyle('Fusion')
    pal=QPalette()
    pal.setColor(QPalette.Window,        QColor(C_BG))
    pal.setColor(QPalette.WindowText,    QColor(C_INK))
    pal.setColor(QPalette.Base,          QColor(C_SURFACE))
    pal.setColor(QPalette.AlternateBase, QColor(C_PANEL))
    pal.setColor(QPalette.Text,          QColor(C_INK))
    pal.setColor(QPalette.Button,        QColor(C_PANEL))
    pal.setColor(QPalette.ButtonText,    QColor(C_INK))
    pal.setColor(QPalette.Highlight,     QColor(C_ACCENT))
    app.setPalette(pal)
    win=MainWindow(); win.show(); sys.exit(app.exec_())