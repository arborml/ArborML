"""
ArborML 1.0
Works on macOS, Windows & Linux (tested on Ubuntu)
GPL-3.0 license
https://github.com/arborml/arborml
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import font as tkfont

import customtkinter as ctk
import numpy as np
import pandas as pd

import math


import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import joblib

from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import Table, TableStyle

APP_NAME = "ArborML"
APP_VERSION = "1.0"
ACCENT = "#2E7D5B"
ACCENT_HOVER = "#256B4C"
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".arborml_config.json")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

logger = logging.getLogger(APP_NAME)


def asset(*parts: str) -> str:
    path = os.path.join(ASSETS_DIR, *parts)
    if not os.path.exists(path):
        fallback = os.path.join(BASE_DIR, *parts)
        if os.path.exists(fallback):
            return fallback
    return path


IS_WINDOWS = sys.platform.startswith("win")
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

ICON_STYLE = "auto"

ICON_SPECS: dict[str, tuple[str, str, str]] = {
    "load":    ("\U0001F4C2", "\u25A4", "[+]"),
    "save":    ("\U0001F4BE", "\u21E7", "->"),
    "import":  ("\U0001F4E5", "\u21E9", "<-"),
    "hist":    ("\U0001F4CA", "\u25A5", "|||"),
    "heat":    ("\U0001F525", "\u25A6", "##"),
    "tree":    ("\U0001F332", "\u2756", "*"),
    "linear":  ("\U0001F4C8", "\u2197", "/"),
    "predict": ("\U0001F3AF", "\u25CE", "()"),
    "export":  ("\U0001F4E4", "\u21E7", "->"),
    "report":  ("\U0001F4C4", "\u25A4", "="),
    "lang":    ("\U0001F310", "\u2295", "@"),
    "sun":     ("\u2600",     "\u2600", "O"),
    "moon":    ("\U0001F319", "\u263E", "C"),
    "ok":      ("\u2714",     "\u2714", "OK"),
    "warn":    ("\u26A0",     "\u26A0", "!"),
}


def _tk_patchlevel(root) -> tuple[int, int, int]:
    try:
        parts = str(root.tk.call("info", "patchlevel")).split(".")
    except Exception:
        return (0, 0, 0)
    out = []
    for part in parts[:3]:
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)


class _GlyphProbe:
    def __init__(self, root):
        self.font = None
        self.missing = None
        try:
            self.font = tkfont.Font(root=root, font="TkDefaultFont")
            w1 = self.font.measure("\uFFFF")
            w2 = self.font.measure("\uE0FF")
            if w1 == w2 and w1 > 0:
                self.missing = w1
        except Exception:
            self.font = None

    def ok(self, ch: str) -> bool:
        if not ch:
            return False
        if self.font is None:
            return max(ord(c) for c in ch) <= 0xFFFF
        try:
            width = self.font.measure(ch)
        except tk.TclError:
            return False
        if width <= 0:
            return False
        return self.missing is None or width != self.missing


class IconTheme:
    def __init__(self) -> None:
        self.mode = "text"
        self.astral_ok = False
        self._icons = {name: spec[2] for name, spec in ICON_SPECS.items()}

    def configure(self, root, preferred: str = "auto") -> None:
        version = _tk_patchlevel(root)
        probe = _GlyphProbe(root)
        self.astral_ok = version >= (8, 6, 10) and probe.ok("\U0001F332")

        families = set()
        try:
            families = {f.lower() for f in tkfont.families(root)}
        except Exception:
            pass

        mode = preferred if preferred in ("emoji", "symbols", "text") \
            else self._auto_mode(version, families)
        self.mode = mode

        icons = {}
        for name, (emoji, symbol, plain) in ICON_SPECS.items():
            chosen = plain
            if mode == "emoji" and self.astral_ok and probe.ok(emoji):
                chosen = emoji
            elif mode in ("emoji", "symbols") and probe.ok(symbol):
                chosen = symbol
            icons[name] = chosen
        self._icons = icons
        logger.debug("Icones : mode=%s, Tk=%s, hors-BMP=%s",
                     mode, version, self.astral_ok)

    def _auto_mode(self, version, families) -> str:
        if not self.astral_ok:
            return "symbols"
        if IS_MACOS:
            return "emoji"
        if IS_WINDOWS and version >= (8, 6, 12) and "segoe ui emoji" in families:
            return "emoji"
        return "symbols"

    def __call__(self, name: str) -> str:
        return self._icons.get(name, "")

    def label(self, name: str, text: str) -> str:
        icon = self(name)
        return f"{icon}  {text}" if icon else text


ICONS = IconTheme()


def tk_safe(text: str) -> str:
    if not text or ICONS.astral_ok:
        return text
    return "".join(c if ord(c) <= 0xFFFF else "\uFFFD" for c in text)


LOGO_LEAF = ACCENT
LOGO_TRUNK = "#6B4A2F"
_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS",
                    Image.BICUBIC)


def render_logo(size: int = 256) -> Image.Image:
    side = max(int(size), 16)
    ss = 4
    px = side * ss
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    u = px / 64.0
    draw.rectangle([28 * u, 42 * u, 36 * u, 60 * u], fill=LOGO_TRUNK)
    for top, half, base in ((3, 14, 26), (13, 21, 41), (24, 27, 55)):
        draw.polygon([(32 * u, top * u),
                      ((32 - half) * u, base * u),
                      ((32 + half) * u, base * u)], fill=LOGO_LEAF)
    return img.resize((side, side), _RESAMPLE)


def app_logo_ctk(size: tuple[int, int] = (30, 30)):
    try:
        source = render_logo(max(size) * 8)
        return ctk.CTkImage(light_image=source, dark_image=source, size=size)
    except Exception as exc:
        logger.debug("Logo indisponible : %s", exc)
        return None


def set_window_icon(win) -> None:
    try:
        png = os.path.join(tempfile.gettempdir(), "arborml_icon.png")
        render_logo(256).save(png)
        photo = tk.PhotoImage(master=win, file=png)
        win.iconphoto(True, photo)
        win._app_icon = photo
    except Exception as exc:
        logger.debug("iconphoto indisponible : %s", exc)
    if IS_WINDOWS:
        try:
            ico = os.path.join(tempfile.gettempdir(), "arborml_icon.ico")
            render_logo(256).save(ico, sizes=[(16, 16), (32, 32), (48, 48),
                                              (64, 64), (128, 128), (256, 256)])
            win.iconbitmap(default=ico)
        except Exception as exc:
            logger.debug("iconbitmap indisponible : %s", exc)


def windows_taskbar_identity() -> None:
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            f"{APP_NAME}.{APP_NAME}.{APP_VERSION}")
    except Exception as exc:
        logger.debug("AppUserModelID indisponible : %s", exc)


MONO_CANDIDATES = ("Menlo", "SF Mono", "Consolas", "Cascadia Mono",
                   "DejaVu Sans Mono", "Liberation Mono", "Ubuntu Mono",
                   "Courier New", "Courier")

UI_FONT_CANDIDATES = {
    "zh": ["Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC",
           "Hiragino Sans GB", "Noto Sans CJK SC", "Source Han Sans SC",
           "WenQuanYi Micro Hei", "SimHei"],
    "ja": ["Yu Gothic UI", "Meiryo UI", "Hiragino Sans",
           "Hiragino Kaku Gothic Pro", "Noto Sans CJK JP",
           "Source Han Sans JP", "TakaoPGothic", "IPAPGothic"],
    "ko": ["Malgun Gothic", "Apple SD Gothic Neo", "AppleGothic",
           "Noto Sans CJK KR", "Source Han Sans KR", "NanumGothic"],
}
_DEFAULT_UI_FAMILY: str | None = None


def _available_families(root) -> dict[str, str]:
    try:
        return {f.lower(): f for f in tkfont.families(root)}
    except Exception:
        return {}


def mono_font(root, size: int = 12) -> tuple[str, int]:
    families = _available_families(root)
    for candidate in MONO_CANDIDATES:
        if candidate.lower() in families:
            return (families[candidate.lower()], size)
    try:
        return (tkfont.nametofont("TkFixedFont").actual("family"), size)
    except Exception:
        return ("Courier", size)


def _ctk_font_nodes() -> list[dict]:
    node = ctk.ThemeManager.theme.get("CTkFont")
    if not isinstance(node, dict):
        return []
    if "family" in node:
        return [node]
    return [v for v in node.values() if isinstance(v, dict) and "family" in v]


def apply_ui_font(root) -> None:
    global _DEFAULT_UI_FAMILY
    nodes = _ctk_font_nodes()
    if not nodes:
        return
    if _DEFAULT_UI_FAMILY is None:
        _DEFAULT_UI_FAMILY = nodes[0].get("family")
    family = _DEFAULT_UI_FAMILY
    candidates = UI_FONT_CANDIDATES.get(CURRENT_LANG)
    if candidates:
        families = _available_families(root)
        for candidate in candidates:
            if candidate.lower() in families:
                family = families[candidate.lower()]
                break
    for node in nodes:
        node["family"] = family


LANGUAGES = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
}
DEFAULT_LANG = "en"
CURRENT_LANG = DEFAULT_LANG

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "error": "Error", "success": "Success", "auto": "auto",
        "language": "Language", "theme": "Theme",
        "sec_data": "DATA", "sec_explore": "EXPLORE", "sec_model": "MODELING",
        "load_file": "Load a file", "export_json": "Export to JSON",
        "import_json": "Import JSON", "histograms": "Histograms",
        "heatmap": "Correlation heatmap", "simulate": "Simulate results",
        "export_model": "Export model", "import_model": "Import model",
        "pdf_report": "PDF report",
        "status_none": "No data loaded. Start with \u201cLoad a file\u201d.",
        "status_loaded": "{file} \u2014 {rows} rows, {cols} columns",
        "last_model": "last model: {name}",
        "rows_hidden": "\u2026 ({n} rows hidden)",
        "rename_cols": "Rename columns:", "new_name": "new name",
        "apply_names": "Apply names",
        "dup_names": "Two columns cannot have the same name.",
        "sep_title": "Separator",
        "sep_msg": "File separator (e.g. ; , tab):",
        "bad_format": "Unsupported format: {ext}",
        "no_numeric": "No usable numeric data found.",
        "data_cleaned": "{r} row(s) and {c} column(s) removed on load "
                        "(missing or non-numeric values)",
        "load_err": "Loading error",
        "data_saved": "Data saved:\n{path}",
        "hyperparams": "Hyperparameters", "options": "Analysis options",
        "cv_check": "Cross-validation (5 folds)", "shap_check": "SHAP analysis",
        "shap_missing": "SHAP analysis (shap module not installed)",
        "x_vars": "Input variables (X)", "y_vars": "Output variables (y)",
        "need_xy": "Select at least one X column and one y column.",
        "xy_overlap": "The same column cannot be both X and y.",
        "bad_params": "Invalid parameters", "train_err": "Training error",
        "train_done": "Training complete",
        "cv_line": "Cross-validation (5 folds)",
        "results": "Results \u2014 {name}",
        "train": "Train the model", "close": "Close",
        "sim_title": "Simulation ({name})", "in_values": "Input values",
        "out_values": "Predicted values", "predict": "Predict",
        "all_numeric": "All inputs must be numeric.",
        "model_saved": "Model exported:\n{path}",
        "model_loaded_title": "Model imported",
        "model_loaded_msg": "Inputs: {x}\nOutputs: {y}\n\n"
                            "You can now run \u201cSimulate results\u201d.",
        "bad_model": "Invalid model file:\n{err}",
        "report_saved": "Report generated:\n{path}",
        "report_err": "Could not generate the PDF:\n{err}",
        "tab_chart": "Chart {i}", "tab_dist": "Distributions",
        "tab_heat": "Heatmap", "tab_imp": "Importances", "tab_y": "y: {t}",
        "real": "(actual)", "predicted": "(predicted)",
        "ideal_line": "Ideal line", "count": "Count",
        "corr_title": "Correlation matrix (Pearson)",
        "imp_title": "Feature importance", "imp_axis": "Importance",
        "shap_title": "SHAP analysis \u2014 {t}",
        "pdf_summary": "Data summary", "pdf_source": "Source file: {file}",
        "pdf_dims": "{r} rows \u00d7 {c} columns",
        "pdf_variable": "Variable", "pdf_min": "Min", "pdf_max": "Max",
        "pdf_mean": "Mean", "pdf_std": "Std dev",
        "pdf_corr": "Correlations", "pdf_model": "Model: {name}",
        "pdf_hparam": "Hyperparameter", "pdf_value": "Value",
        "pdf_none": "None", "pdf_inputs": "Inputs (X)",
        "pdf_outputs": "Outputs (y)", "pdf_target": "Target",
        "pdf_pred_page": "Predictions vs actual values",
        "ft_all": "All formats", "ft_text": "Text",
        "ft_model": "ArborML model", "ft_allfiles": "All files",
        "p_test_size": "Test size", "p_random_state": "Random seed",
        "p_max_depth": "Max depth", "p_min_split": "Min samples split",
        "p_min_leaf": "Min samples leaf", "p_n_estimators": "Number of trees",
        "p_max_features": "Max features (0\u20131)",
        "p_bootstrap": "Bootstrap (yes/no)",
        "p_learning_rate": "Learning rate", "p_subsample": "Subsample",
        "p_fit_intercept": "Fit intercept (yes/no)",
        "p_positive": "Positive coefficients (yes/no)",
        "p_colsample": "Colsample bytree", "p_reg_alpha": "Alpha (L1)",
        "p_reg_lambda": "Lambda (L2)",
        "val_invalid": "\u201c{label}\u201d: invalid value ({raw}).",
        "val_min": "\u201c{label}\u201d must be \u2265 {v}.",
        "val_max": "\u201c{label}\u201d must be \u2264 {v}.",
        "pdf_sims": "Simulations run", "pdf_sim_n": "#",
        "pdf_sim_time": "Time",
        "pdf_sim_ask": "Include the {n} simulation(s) run in this session "
                       "in the report?",
        "sim_count": "{n} simulation(s) recorded",
        "sim_clear": "Clear the simulation history",
        "sim_cleared": "Simulation history cleared.",
        "prog_read": "Reading the file",
        "prog_clean": "Filtering numeric data",
        "prog_display": "Updating the display",
        "prog_write": "Writing the file",
        "prog_prepare": "Preparing the data",
        "prog_fit": "Fitting the model",
        "prog_figure": "Chart: {t}",
        "prog_figures": "Building the charts",
        "prog_importance": "Feature importance",
        "prog_cv": "Cross-validation",
        "prog_shap": "SHAP: {t}",
        "prog_compute": "Computing",
        "prog_pdf_data": "Data pages",
        "prog_pdf_model": "Model pages",
        "prog_pdf_sim": "Simulation pages",
        "prog_pdf_save": "Finalizing the PDF",
        "prog_save_model": "Saving the model",
        "prog_load_model": "Loading the model",
        "prog_done": "Done",
    },
    "es": {
        "error": "Error", "success": "Éxito", "auto": "auto",
        "language": "Idioma", "theme": "Tema",
        "sec_data": "DATOS", "sec_explore": "EXPLORACIÓN",
        "sec_model": "MODELIZACIÓN",
        "load_file": "Cargar un archivo", "export_json": "Exportar a JSON",
        "import_json": "Importar JSON", "histograms": "Histogramas",
        "heatmap": "Mapa de correlaciones",
        "simulate": "Simular resultados",
        "export_model": "Exportar el modelo",
        "import_model": "Importar un modelo",
        "pdf_report": "Informe PDF",
        "status_none": "Ningún dato cargado. Empiece por "
                       "«Cargar un archivo».",
        "status_loaded": "{file} \u2014 {rows} filas, {cols} columnas",
        "last_model": "último modelo: {name}",
        "rows_hidden": "\u2026 ({n} filas ocultas)",
        "rename_cols": "Renombrar columnas:", "new_name": "nuevo nombre",
        "apply_names": "Aplicar nombres",
        "dup_names": "Dos columnas no pueden tener el mismo nombre.",
        "sep_title": "Separador",
        "sep_msg": "Separador del archivo (ej. ; , tab):",
        "bad_format": "Formato no compatible: {ext}",
        "no_numeric": "No se encontraron datos numéricos utilizables.",
        "data_cleaned": "{r} fila(s) y {c} columna(s) eliminadas al "
                        "cargar (valores ausentes o no numéricos)",
        "load_err": "Error de carga",
        "data_saved": "Datos guardados:\n{path}",
        "hyperparams": "Hiperparámetros", "options": "Opciones de análisis",
        "cv_check": "Validación cruzada (5 pliegues)",
        "shap_check": "Análisis SHAP",
        "shap_missing": "Análisis SHAP (módulo shap no instalado)",
        "x_vars": "Variables de entrada (X)",
        "y_vars": "Variables de salida (y)",
        "need_xy": "Seleccione al menos una columna X y una columna y.",
        "xy_overlap": "Una misma columna no puede ser X e y a la vez.",
        "bad_params": "Parámetros no válidos",
        "train_err": "Error de entrenamiento",
        "train_done": "Entrenamiento terminado",
        "cv_line": "Validación cruzada (5 pliegues)",
        "results": "Resultados \u2014 {name}",
        "train": "Entrenar el modelo", "close": "Cerrar",
        "sim_title": "Simulación ({name})", "in_values": "Valores de entrada",
        "out_values": "Valores predichos", "predict": "Predecir",
        "all_numeric": "Todas las entradas deben ser numéricas.",
        "model_saved": "Modelo exportado:\n{path}",
        "model_loaded_title": "Modelo importado",
        "model_loaded_msg": "Entradas: {x}\nSalidas: {y}\n\n"
                            "Puede lanzar «Simular resultados».",
        "bad_model": "Archivo de modelo no válido:\n{err}",
        "report_saved": "Informe generado:\n{path}",
        "report_err": "No se pudo generar el PDF:\n{err}",
        "tab_chart": "Gráfico {i}", "tab_dist": "Distribuciones",
        "tab_heat": "Mapa de calor", "tab_imp": "Importancias",
        "tab_y": "y: {t}",
        "real": "(real)", "predicted": "(predicho)",
        "ideal_line": "Línea ideal", "count": "Frecuencia",
        "corr_title": "Matriz de correlación (Pearson)",
        "imp_title": "Importancia de las variables", "imp_axis": "Importancia",
        "shap_title": "Análisis SHAP \u2014 {t}",
        "pdf_summary": "Síntesis de los datos",
        "pdf_source": "Archivo fuente: {file}",
        "pdf_dims": "{r} filas \u00d7 {c} columnas",
        "pdf_variable": "Variable", "pdf_min": "Mín", "pdf_max": "Máx",
        "pdf_mean": "Media", "pdf_std": "Desv. típica",
        "pdf_corr": "Correlaciones", "pdf_model": "Modelo: {name}",
        "pdf_hparam": "Hiperparámetro", "pdf_value": "Valor",
        "pdf_none": "Ninguno", "pdf_inputs": "Entradas (X)",
        "pdf_outputs": "Salidas (y)", "pdf_target": "Objetivo",
        "pdf_pred_page": "Predicciones vs valores reales",
        "ft_all": "Todos los formatos", "ft_text": "Texto",
        "ft_model": "Modelo ArborML", "ft_allfiles": "Todos los archivos",
        "p_test_size": "Tamaño del test", "p_random_state": "Semilla aleatoria",
        "p_max_depth": "Profundidad máx",
        "p_min_split": "Mín. muestras split",
        "p_min_leaf": "Mín. muestras hoja",
        "p_n_estimators": "Número de árboles",
        "p_max_features": "Máx. features (0\u20131)",
        "p_bootstrap": "Bootstrap (sí/no)",
        "p_learning_rate": "Tasa de aprendizaje", "p_subsample": "Submuestreo",
        "p_fit_intercept": "Ajustar intercepto (sí/no)",
        "p_positive": "Coeficientes positivos (sí/no)",
        "p_colsample": "Colsample bytree", "p_reg_alpha": "Alfa (L1)",
        "p_reg_lambda": "Lambda (L2)",
        "val_invalid": "«{label}»: valor no válido ({raw}).",
        "val_min": "«{label}» debe ser \u2265 {v}.",
        "val_max": "«{label}» debe ser \u2264 {v}.",
        "pdf_sims": "Simulaciones realizadas", "pdf_sim_n": "N.º",
        "pdf_sim_time": "Hora",
        "pdf_sim_ask": "¿Incluir en el informe las {n} simulación(es) "
                       "de esta sesión?",
        "sim_count": "{n} simulación(es) registrada(s)",
        "sim_clear": "Borrar el historial de simulaciones",
        "sim_cleared": "Historial de simulaciones borrado.",
        "prog_read": "Lectura del archivo",
        "prog_clean": "Filtrado de datos numéricos",
        "prog_display": "Actualización de la vista",
        "prog_write": "Escritura del archivo",
        "prog_prepare": "Preparación de los datos",
        "prog_fit": "Ajuste del modelo",
        "prog_figure": "Gráfico: {t}",
        "prog_figures": "Creación de los gráficos",
        "prog_importance": "Importancia de las variables",
        "prog_cv": "Validación cruzada",
        "prog_shap": "SHAP: {t}",
        "prog_compute": "Calculando",
        "prog_pdf_data": "Páginas de datos",
        "prog_pdf_model": "Páginas del modelo",
        "prog_pdf_sim": "Páginas de simulaciones",
        "prog_pdf_save": "Finalización del PDF",
        "prog_save_model": "Guardando el modelo",
        "prog_load_model": "Cargando el modelo",
        "prog_done": "Terminado",
    },
    "fr": {
        "error": "Erreur", "success": "Succès", "auto": "auto",
        "language": "Langue", "theme": "Thème",
        "sec_data": "DONNÉES", "sec_explore": "EXPLORATION",
        "sec_model": "MODÉLISATION",
        "load_file": "Charger un fichier", "export_json": "Exporter en JSON",
        "import_json": "Importer un JSON", "histograms": "Histogrammes",
        "heatmap": "Heatmap corrélations",
        "simulate": "Simuler des résultats",
        "export_model": "Exporter le modèle",
        "import_model": "Importer un modèle",
        "pdf_report": "Rapport PDF",
        "status_none": "Aucune donnée chargée. Commencez par "
                       "« Charger un fichier ».",
        "status_loaded": "{file} \u2014 {rows} lignes, {cols} colonnes",
        "last_model": "dernier modèle : {name}",
        "rows_hidden": "\u2026 ({n} lignes masquées)",
        "rename_cols": "Renommer les colonnes :", "new_name": "nouveau nom",
        "apply_names": "Appliquer les noms",
        "dup_names": "Deux colonnes ne peuvent pas porter le même nom.",
        "sep_title": "Séparateur",
        "sep_msg": "Séparateur du fichier (ex. ; , tab) :",
        "bad_format": "Format non pris en charge : {ext}",
        "no_numeric": "Aucune donnée numérique exploitable trouvée.",
        "data_cleaned": "{r} ligne(s) et {c} colonne(s) écartée(s) au "
                        "chargement (valeurs manquantes ou non "
                        "numériques)",
        "load_err": "Erreur de chargement",
        "data_saved": "Données enregistrées :\n{path}",
        "hyperparams": "Hyperparamètres", "options": "Options d'analyse",
        "cv_check": "Validation croisée (5 folds)",
        "shap_check": "Analyse SHAP",
        "shap_missing": "Analyse SHAP (module shap non installé)",
        "x_vars": "Variables d'entrée (X)",
        "y_vars": "Variables de sortie (y)",
        "need_xy": "Sélectionnez au moins une colonne X et une colonne y.",
        "xy_overlap": "Une même colonne ne peut pas être à la fois X et y.",
        "bad_params": "Paramètres invalides",
        "train_err": "Erreur d'entraînement",
        "train_done": "Entraînement terminé",
        "cv_line": "Validation croisée (5 folds)",
        "results": "Résultats \u2014 {name}",
        "train": "Entraîner le modèle", "close": "Fermer",
        "sim_title": "Simulation ({name})", "in_values": "Valeurs d'entrée",
        "out_values": "Valeurs prédites", "predict": "Prédire",
        "all_numeric": "Toutes les entrées doivent être numériques.",
        "model_saved": "Modèle exporté :\n{path}",
        "model_loaded_title": "Modèle importé",
        "model_loaded_msg": "Entrées : {x}\nSorties : {y}\n\n"
                            "Vous pouvez lancer « Simuler des résultats ».",
        "bad_model": "Fichier de modèle invalide :\n{err}",
        "report_saved": "Rapport généré :\n{path}",
        "report_err": "Génération du PDF impossible :\n{err}",
        "tab_chart": "Graphique {i}", "tab_dist": "Distributions",
        "tab_heat": "Heatmap", "tab_imp": "Importances", "tab_y": "y : {t}",
        "real": "(réel)", "predicted": "(prédit)",
        "ideal_line": "Droite idéale", "count": "Effectif",
        "corr_title": "Matrice de corrélation (Pearson)",
        "imp_title": "Importance des variables", "imp_axis": "Importance",
        "shap_title": "Analyse SHAP \u2014 {t}",
        "pdf_summary": "Synthèse des données",
        "pdf_source": "Fichier source : {file}",
        "pdf_dims": "{r} lignes \u00d7 {c} colonnes",
        "pdf_variable": "Variable", "pdf_min": "Min", "pdf_max": "Max",
        "pdf_mean": "Moyenne", "pdf_std": "Écart-type",
        "pdf_corr": "Corrélations", "pdf_model": "Modèle : {name}",
        "pdf_hparam": "Hyperparamètre", "pdf_value": "Valeur",
        "pdf_none": "Aucune", "pdf_inputs": "Entrées (X)",
        "pdf_outputs": "Sorties (y)", "pdf_target": "Cible",
        "pdf_pred_page": "Prédictions vs valeurs réelles",
        "ft_all": "Tous les formats", "ft_text": "Texte",
        "ft_model": "Modèle ArborML", "ft_allfiles": "Tous les fichiers",
        "p_test_size": "Taille du test", "p_random_state": "Random state",
        "p_max_depth": "Profondeur max",
        "p_min_split": "Min samples split",
        "p_min_leaf": "Min samples leaf",
        "p_n_estimators": "Nombre d'arbres",
        "p_max_features": "Max features (0\u20131)",
        "p_bootstrap": "Bootstrap (oui/non)",
        "p_learning_rate": "Learning rate", "p_subsample": "Subsample",
        "p_fit_intercept": "Ordonnée à l'origine (oui/non)",
        "p_positive": "Coefficients positifs (oui/non)",
        "p_colsample": "Colsample bytree", "p_reg_alpha": "Alpha (L1)",
        "p_reg_lambda": "Lambda (L2)",
        "val_invalid": "« {label} » : valeur invalide ({raw}).",
        "val_min": "« {label} » doit être \u2265 {v}.",
        "val_max": "« {label} » doit être \u2264 {v}.",
        "pdf_sims": "Simulations réalisées", "pdf_sim_n": "N°",
        "pdf_sim_time": "Heure",
        "pdf_sim_ask": "Inclure dans le rapport les {n} simulation(s) "
                       "réalisée(s) pendant la session ?",
        "sim_count": "{n} simulation(s) enregistrée(s)",
        "sim_clear": "Vider l'historique des simulations",
        "sim_cleared": "Historique des simulations vidé.",
        "prog_read": "Lecture du fichier",
        "prog_clean": "Filtrage des données numériques",
        "prog_display": "Mise à jour de l'affichage",
        "prog_write": "Écriture du fichier",
        "prog_prepare": "Préparation des données",
        "prog_fit": "Ajustement du modèle",
        "prog_figure": "Graphique : {t}",
        "prog_figures": "Construction des graphiques",
        "prog_importance": "Importance des variables",
        "prog_cv": "Validation croisée",
        "prog_shap": "SHAP : {t}",
        "prog_compute": "Calcul en cours",
        "prog_pdf_data": "Pages de données",
        "prog_pdf_model": "Pages du modèle",
        "prog_pdf_sim": "Pages de simulations",
        "prog_pdf_save": "Finalisation du PDF",
        "prog_save_model": "Enregistrement du modèle",
        "prog_load_model": "Chargement du modèle",
        "prog_done": "Terminé",
    },
    "de": {
        "error": "Fehler", "success": "Erfolg", "auto": "auto",
        "language": "Sprache", "theme": "Design",
        "sec_data": "DATEN", "sec_explore": "ERKUNDUNG",
        "sec_model": "MODELLIERUNG",
        "load_file": "Datei laden", "export_json": "Als JSON exportieren",
        "import_json": "JSON importieren", "histograms": "Histogramme",
        "heatmap": "Korrelations-Heatmap",
        "simulate": "Ergebnisse simulieren",
        "export_model": "Modell exportieren",
        "import_model": "Modell importieren",
        "pdf_report": "PDF-Bericht",
        "status_none": "Keine Daten geladen. Beginnen Sie mit "
                       "\u201eDatei laden\u201c.",
        "status_loaded": "{file} \u2014 {rows} Zeilen, {cols} Spalten",
        "last_model": "letztes Modell: {name}",
        "rows_hidden": "\u2026 ({n} Zeilen ausgeblendet)",
        "rename_cols": "Spalten umbenennen:", "new_name": "neuer Name",
        "apply_names": "Namen übernehmen",
        "dup_names": "Zwei Spalten dürfen nicht denselben Namen tragen.",
        "sep_title": "Trennzeichen",
        "sep_msg": "Trennzeichen der Datei (z. B. ; , tab):",
        "bad_format": "Nicht unterstütztes Format: {ext}",
        "no_numeric": "Keine verwertbaren numerischen Daten gefunden.",
        "data_cleaned": "{r} Zeile(n) und {c} Spalte(n) beim Laden "
                        "entfernt (fehlende oder nicht numerische "
                        "Werte)",
        "load_err": "Fehler beim Laden",
        "data_saved": "Daten gespeichert:\n{path}",
        "hyperparams": "Hyperparameter", "options": "Analyseoptionen",
        "cv_check": "Kreuzvalidierung (5 Folds)",
        "shap_check": "SHAP-Analyse",
        "shap_missing": "SHAP-Analyse (Modul shap nicht installiert)",
        "x_vars": "Eingabevariablen (X)",
        "y_vars": "Ausgabevariablen (y)",
        "need_xy": "Wählen Sie mindestens eine X- und eine y-Spalte.",
        "xy_overlap": "Eine Spalte kann nicht gleichzeitig X und y sein.",
        "bad_params": "Ungültige Parameter", "train_err": "Trainingsfehler",
        "train_done": "Training abgeschlossen",
        "cv_line": "Kreuzvalidierung (5 Folds)",
        "results": "Ergebnisse \u2014 {name}",
        "train": "Modell trainieren", "close": "Schließen",
        "sim_title": "Simulation ({name})", "in_values": "Eingabewerte",
        "out_values": "Vorhergesagte Werte", "predict": "Vorhersagen",
        "all_numeric": "Alle Eingaben müssen numerisch sein.",
        "model_saved": "Modell exportiert:\n{path}",
        "model_loaded_title": "Modell importiert",
        "model_loaded_msg": "Eingaben: {x}\nAusgaben: {y}\n\n"
                            "Sie können \u201eErgebnisse simulieren\u201c "
                            "starten.",
        "bad_model": "Ungültige Modelldatei:\n{err}",
        "report_saved": "Bericht erstellt:\n{path}",
        "report_err": "PDF konnte nicht erstellt werden:\n{err}",
        "tab_chart": "Diagramm {i}", "tab_dist": "Verteilungen",
        "tab_heat": "Heatmap", "tab_imp": "Wichtigkeiten", "tab_y": "y: {t}",
        "real": "(tatsächlich)", "predicted": "(vorhergesagt)",
        "ideal_line": "Ideallinie", "count": "Häufigkeit",
        "corr_title": "Korrelationsmatrix (Pearson)",
        "imp_title": "Variablenwichtigkeit", "imp_axis": "Wichtigkeit",
        "shap_title": "SHAP-Analyse \u2014 {t}",
        "pdf_summary": "Datenübersicht",
        "pdf_source": "Quelldatei: {file}",
        "pdf_dims": "{r} Zeilen \u00d7 {c} Spalten",
        "pdf_variable": "Variable", "pdf_min": "Min", "pdf_max": "Max",
        "pdf_mean": "Mittelwert", "pdf_std": "Std.-Abw.",
        "pdf_corr": "Korrelationen", "pdf_model": "Modell: {name}",
        "pdf_hparam": "Hyperparameter", "pdf_value": "Wert",
        "pdf_none": "Keine", "pdf_inputs": "Eingaben (X)",
        "pdf_outputs": "Ausgaben (y)", "pdf_target": "Ziel",
        "pdf_pred_page": "Vorhersagen vs. tatsächliche Werte",
        "ft_all": "Alle Formate", "ft_text": "Text",
        "ft_model": "ArborML-Modell", "ft_allfiles": "Alle Dateien",
        "p_test_size": "Testanteil", "p_random_state": "Zufalls-Seed",
        "p_max_depth": "Max. Tiefe",
        "p_min_split": "Min. Samples Split",
        "p_min_leaf": "Min. Samples Leaf",
        "p_n_estimators": "Anzahl Bäume",
        "p_max_features": "Max. Features (0\u20131)",
        "p_bootstrap": "Bootstrap (ja/nein)",
        "p_learning_rate": "Lernrate", "p_subsample": "Subsample",
        "p_fit_intercept": "Achsenabschnitt (ja/nein)",
        "p_positive": "Positive Koeffizienten (ja/nein)",
        "p_colsample": "Colsample bytree", "p_reg_alpha": "Alpha (L1)",
        "p_reg_lambda": "Lambda (L2)",
        "val_invalid": "\u201e{label}\u201c: ungültiger Wert ({raw}).",
        "val_min": "\u201e{label}\u201c muss \u2265 {v} sein.",
        "val_max": "\u201e{label}\u201c muss \u2264 {v} sein.",
        "pdf_sims": "Durchgeführte Simulationen", "pdf_sim_n": "Nr.",
        "pdf_sim_time": "Zeit",
        "pdf_sim_ask": "Die {n} Simulation(en) dieser Sitzung in den "
                       "Bericht aufnehmen?",
        "sim_count": "{n} Simulation(en) gespeichert",
        "sim_clear": "Simulationsverlauf löschen",
        "sim_cleared": "Simulationsverlauf gelöscht.",
        "prog_read": "Datei wird gelesen",
        "prog_clean": "Numerische Daten werden gefiltert",
        "prog_display": "Anzeige wird aktualisiert",
        "prog_write": "Datei wird geschrieben",
        "prog_prepare": "Daten werden vorbereitet",
        "prog_fit": "Modell wird angepasst",
        "prog_figure": "Diagramm: {t}",
        "prog_figures": "Diagramme werden erstellt",
        "prog_importance": "Variablenwichtigkeit",
        "prog_cv": "Kreuzvalidierung",
        "prog_shap": "SHAP: {t}",
        "prog_compute": "Berechnung läuft",
        "prog_pdf_data": "Datenseiten",
        "prog_pdf_model": "Modellseiten",
        "prog_pdf_sim": "Simulationsseiten",
        "prog_pdf_save": "PDF wird finalisiert",
        "prog_save_model": "Modell wird gespeichert",
        "prog_load_model": "Modell wird geladen",
        "prog_done": "Fertig",
    },
    "zh": {
        "error": "错误", "success": "成功", "auto": "自动",
        "language": "语言", "theme": "主题",
        "sec_data": "数据", "sec_explore": "数据探索", "sec_model": "建模",
        "load_file": "加载文件", "export_json": "导出为 JSON",
        "import_json": "导入 JSON", "histograms": "直方图",
        "heatmap": "相关性热图", "simulate": "结果模拟",
        "export_model": "导出模型", "import_model": "导入模型",
        "pdf_report": "PDF 报告",
        "status_none": "尚未加载数据。请先使用“加载文件”。",
        "status_loaded": "{file} \u2014 {rows} 行 \u00d7 {cols} 列",
        "last_model": "最近模型：{name}",
        "rows_hidden": "\u2026（隐藏 {n} 行）",
        "rename_cols": "重命名列：", "new_name": "新名称",
        "apply_names": "应用名称",
        "dup_names": "两列不能使用相同的名称。",
        "sep_title": "分隔符",
        "sep_msg": "文件分隔符（如 ; , tab）：",
        "bad_format": "不支持的格式：{ext}",
        "no_numeric": "未找到可用的数值数据。",
        "data_cleaned": "加载时已剔除 {r} 行、{c} 列"
                        "（缺失值或非数值）",
        "load_err": "加载错误",
        "data_saved": "数据已保存：\n{path}",
        "hyperparams": "超参数", "options": "分析选项",
        "cv_check": "交叉验证（5 折）", "shap_check": "SHAP 分析",
        "shap_missing": "SHAP 分析（未安装 shap 模块）",
        "x_vars": "输入变量 (X)", "y_vars": "输出变量 (y)",
        "need_xy": "请至少选择一列 X 和一列 y。",
        "xy_overlap": "同一列不能同时作为 X 和 y。",
        "bad_params": "参数无效", "train_err": "训练错误",
        "train_done": "训练完成",
        "cv_line": "交叉验证（5 折）",
        "results": "结果 \u2014 {name}",
        "train": "训练模型", "close": "关闭",
        "sim_title": "模拟（{name}）", "in_values": "输入值",
        "out_values": "预测值", "predict": "预测",
        "all_numeric": "所有输入必须为数值。",
        "model_saved": "模型已导出：\n{path}",
        "model_loaded_title": "模型已导入",
        "model_loaded_msg": "输入：{x}\n输出：{y}\n\n现在可以使用“结果模拟”。",
        "bad_model": "无效的模型文件：\n{err}",
        "report_saved": "报告已生成：\n{path}",
        "report_err": "无法生成 PDF：\n{err}",
        "tab_chart": "图 {i}", "tab_dist": "分布",
        "tab_heat": "热图", "tab_imp": "重要性", "tab_y": "y：{t}",
        "real": "（实际）", "predicted": "（预测）",
        "ideal_line": "理想线", "count": "频数",
        "corr_title": "相关矩阵（Pearson）",
        "imp_title": "变量重要性", "imp_axis": "重要性",
        "shap_title": "SHAP 分析 \u2014 {t}",
        "pdf_summary": "数据概览", "pdf_source": "源文件：{file}",
        "pdf_dims": "{r} 行 \u00d7 {c} 列",
        "pdf_variable": "变量", "pdf_min": "最小值", "pdf_max": "最大值",
        "pdf_mean": "均值", "pdf_std": "标准差",
        "pdf_corr": "相关性", "pdf_model": "模型：{name}",
        "pdf_hparam": "超参数", "pdf_value": "值",
        "pdf_none": "无", "pdf_inputs": "输入 (X)",
        "pdf_outputs": "输出 (y)", "pdf_target": "目标",
        "pdf_pred_page": "预测值 vs 实际值",
        "ft_all": "所有格式", "ft_text": "文本",
        "ft_model": "ArborML 模型", "ft_allfiles": "所有文件",
        "p_test_size": "测试集比例", "p_random_state": "随机种子",
        "p_max_depth": "最大深度",
        "p_min_split": "最小分裂样本数",
        "p_min_leaf": "最小叶样本数",
        "p_n_estimators": "树的数量",
        "p_max_features": "最大特征比例 (0\u20131)",
        "p_bootstrap": "Bootstrap（是/否）",
        "p_learning_rate": "学习率", "p_subsample": "子采样",
        "p_fit_intercept": "拟合截距（是/否）",
        "p_positive": "系数为正（是/否）",
        "p_colsample": "Colsample bytree", "p_reg_alpha": "Alpha (L1)",
        "p_reg_lambda": "Lambda (L2)",
        "val_invalid": "“{label}”：无效值（{raw}）。",
        "val_min": "“{label}”必须 \u2265 {v}。",
        "val_max": "“{label}”必须 \u2264 {v}。",
        "pdf_sims": "已执行的模拟", "pdf_sim_n": "序号",
        "pdf_sim_time": "时间",
        "pdf_sim_ask": "是否将本次会话中的 {n} 次模拟写入报告？",
        "sim_count": "已记录 {n} 次模拟",
        "sim_clear": "清空模拟记录",
        "sim_cleared": "模拟记录已清空。",
        "prog_read": "正在读取文件",
        "prog_clean": "正在筛选数值数据",
        "prog_display": "正在更新显示",
        "prog_write": "正在写入文件",
        "prog_prepare": "正在准备数据",
        "prog_fit": "正在拟合模型",
        "prog_figure": "图表：{t}",
        "prog_figures": "正在生成图表",
        "prog_importance": "变量重要性",
        "prog_cv": "交叉验证",
        "prog_shap": "SHAP：{t}",
        "prog_compute": "正在计算",
        "prog_pdf_data": "数据页",
        "prog_pdf_model": "模型页",
        "prog_pdf_sim": "模拟页",
        "prog_pdf_save": "正在生成 PDF",
        "prog_save_model": "正在保存模型",
        "prog_load_model": "正在加载模型",
        "prog_done": "完成",
    },
    "ja": {
        "error": "エラー", "success": "成功", "auto": "自動",
        "language": "言語", "theme": "テーマ",
        "sec_data": "データ", "sec_explore": "探索", "sec_model": "モデリング",
        "load_file": "ファイルを読み込む", "export_json": "JSON にエクスポート",
        "import_json": "JSON をインポート", "histograms": "ヒストグラム",
        "heatmap": "相関ヒートマップ", "simulate": "結果をシミュレート",
        "export_model": "モデルをエクスポート",
        "import_model": "モデルをインポート",
        "pdf_report": "PDF レポート",
        "status_none": "データが読み込まれていません。"
                       "「ファイルを読み込む」から始めてください。",
        "status_loaded": "{file} \u2014 {rows} 行 \u00d7 {cols} 列",
        "last_model": "最新モデル：{name}",
        "rows_hidden": "\u2026（{n} 行を省略）",
        "rename_cols": "列名の変更：", "new_name": "新しい名前",
        "apply_names": "名前を適用",
        "dup_names": "同じ名前の列を 2 つ作ることはできません。",
        "sep_title": "区切り文字",
        "sep_msg": "ファイルの区切り文字（例：; , tab）：",
        "bad_format": "未対応の形式：{ext}",
        "no_numeric": "利用可能な数値データが見つかりません。",
        "data_cleaned": "読み込み時に {r} 行・{c} 列を除外"
                        "（欠損値または非数値）",
        "load_err": "読み込みエラー",
        "data_saved": "データを保存しました：\n{path}",
        "hyperparams": "ハイパーパラメータ", "options": "分析オプション",
        "cv_check": "交差検証（5 分割）", "shap_check": "SHAP 分析",
        "shap_missing": "SHAP 分析（shap モジュール未インストール）",
        "x_vars": "入力変数 (X)", "y_vars": "出力変数 (y)",
        "need_xy": "X 列と y 列を少なくとも 1 つずつ選択してください。",
        "xy_overlap": "同じ列を X と y の両方にはできません。",
        "bad_params": "無効なパラメータ", "train_err": "学習エラー",
        "train_done": "学習完了",
        "cv_line": "交差検証（5 分割）",
        "results": "結果 \u2014 {name}",
        "train": "モデルを学習", "close": "閉じる",
        "sim_title": "シミュレーション（{name}）", "in_values": "入力値",
        "out_values": "予測値", "predict": "予測",
        "all_numeric": "すべての入力は数値である必要があります。",
        "model_saved": "モデルをエクスポートしました：\n{path}",
        "model_loaded_title": "モデルをインポートしました",
        "model_loaded_msg": "入力：{x}\n出力：{y}\n\n"
                            "「結果をシミュレート」を実行できます。",
        "bad_model": "無効なモデルファイル：\n{err}",
        "report_saved": "レポートを生成しました：\n{path}",
        "report_err": "PDF を生成できません：\n{err}",
        "tab_chart": "グラフ {i}", "tab_dist": "分布",
        "tab_heat": "ヒートマップ", "tab_imp": "重要度", "tab_y": "y：{t}",
        "real": "（実測）", "predicted": "（予測）",
        "ideal_line": "理想直線", "count": "度数",
        "corr_title": "相関行列（Pearson）",
        "imp_title": "変数の重要度", "imp_axis": "重要度",
        "shap_title": "SHAP 分析 \u2014 {t}",
        "pdf_summary": "データ概要", "pdf_source": "ソースファイル：{file}",
        "pdf_dims": "{r} 行 \u00d7 {c} 列",
        "pdf_variable": "変数", "pdf_min": "最小", "pdf_max": "最大",
        "pdf_mean": "平均", "pdf_std": "標準偏差",
        "pdf_corr": "相関", "pdf_model": "モデル：{name}",
        "pdf_hparam": "ハイパーパラメータ", "pdf_value": "値",
        "pdf_none": "なし", "pdf_inputs": "入力 (X)",
        "pdf_outputs": "出力 (y)", "pdf_target": "目的変数",
        "pdf_pred_page": "予測値 vs 実測値",
        "ft_all": "すべての形式", "ft_text": "テキスト",
        "ft_model": "ArborML モデル", "ft_allfiles": "すべてのファイル",
        "p_test_size": "テスト割合", "p_random_state": "乱数シード",
        "p_max_depth": "最大深さ",
        "p_min_split": "分割最小サンプル数",
        "p_min_leaf": "葉の最小サンプル数",
        "p_n_estimators": "木の本数",
        "p_max_features": "最大特徴量割合 (0\u20131)",
        "p_bootstrap": "ブートストラップ（はい/いいえ）",
        "p_learning_rate": "学習率", "p_subsample": "サブサンプル",
        "p_fit_intercept": "切片を推定（はい/いいえ）",
        "p_positive": "係数を非負に（はい/いいえ）",
        "p_colsample": "Colsample bytree", "p_reg_alpha": "Alpha (L1)",
        "p_reg_lambda": "Lambda (L2)",
        "val_invalid": "「{label}」：無効な値（{raw}）。",
        "val_min": "「{label}」は {v} 以上にしてください。",
        "val_max": "「{label}」は {v} 以下にしてください。",
        "pdf_sims": "実行したシミュレーション", "pdf_sim_n": "番号",
        "pdf_sim_time": "時刻",
        "pdf_sim_ask": "このセッションの {n} 件のシミュレーションを"
                       "レポートに含めますか？",
        "sim_count": "{n} 件のシミュレーションを記録",
        "sim_clear": "シミュレーション履歴を消去",
        "sim_cleared": "シミュレーション履歴を消去しました。",
        "prog_read": "ファイルを読み込み中",
        "prog_clean": "数値データを抽出中",
        "prog_display": "表示を更新中",
        "prog_write": "ファイルを書き込み中",
        "prog_prepare": "データを準備中",
        "prog_fit": "モデルを学習中",
        "prog_figure": "グラフ：{t}",
        "prog_figures": "グラフを作成中",
        "prog_importance": "変数の重要度",
        "prog_cv": "交差検証",
        "prog_shap": "SHAP：{t}",
        "prog_compute": "計算中",
        "prog_pdf_data": "データページ",
        "prog_pdf_model": "モデルページ",
        "prog_pdf_sim": "シミュレーションのページ",
        "prog_pdf_save": "PDF を仕上げ中",
        "prog_save_model": "モデルを保存中",
        "prog_load_model": "モデルを読み込み中",
        "prog_done": "完了",
    },
    "ko": {
        "error": "오류", "success": "성공", "auto": "자동",
        "language": "언어", "theme": "테마",
        "sec_data": "데이터", "sec_explore": "탐색", "sec_model": "모델링",
        "load_file": "파일 불러오기", "export_json": "JSON으로 내보내기",
        "import_json": "JSON 가져오기", "histograms": "히스토그램",
        "heatmap": "상관관계 히트맵", "simulate": "결과 시뮬레이션",
        "export_model": "모델 내보내기", "import_model": "모델 가져오기",
        "pdf_report": "PDF 보고서",
        "status_none": "불러온 데이터가 없습니다. "
                       "\"파일 불러오기\"부터 시작하세요.",
        "status_loaded": "{file} \u2014 {rows}행 \u00d7 {cols}열",
        "last_model": "최근 모델: {name}",
        "rows_hidden": "\u2026 ({n}행 숨김)",
        "rename_cols": "열 이름 바꾸기:", "new_name": "새 이름",
        "apply_names": "이름 적용",
        "dup_names": "두 열이 같은 이름을 가질 수 없습니다.",
        "sep_title": "구분자",
        "sep_msg": "파일 구분자 (예: ; , tab):",
        "bad_format": "지원하지 않는 형식: {ext}",
        "no_numeric": "사용 가능한 숫자 데이터가 없습니다.",
        "data_cleaned": "불러오는 중 {r}행 · {c}열 제외"
                        "(결측치 또는 비숫자 값)",
        "load_err": "불러오기 오류",
        "data_saved": "데이터 저장 완료:\n{path}",
        "hyperparams": "하이퍼파라미터", "options": "분석 옵션",
        "cv_check": "교차 검증 (5-폴드)", "shap_check": "SHAP 분석",
        "shap_missing": "SHAP 분석 (shap 모듈 미설치)",
        "x_vars": "입력 변수 (X)", "y_vars": "출력 변수 (y)",
        "need_xy": "X 열과 y 열을 각각 최소 하나 선택하세요.",
        "xy_overlap": "같은 열을 X와 y로 동시에 사용할 수 없습니다.",
        "bad_params": "잘못된 파라미터", "train_err": "학습 오류",
        "train_done": "학습 완료",
        "cv_line": "교차 검증 (5-폴드)",
        "results": "결과 \u2014 {name}",
        "train": "모델 학습", "close": "닫기",
        "sim_title": "시뮬레이션 ({name})", "in_values": "입력값",
        "out_values": "예측값", "predict": "예측",
        "all_numeric": "모든 입력은 숫자여야 합니다.",
        "model_saved": "모델 내보내기 완료:\n{path}",
        "model_loaded_title": "모델 가져오기 완료",
        "model_loaded_msg": "입력: {x}\n출력: {y}\n\n"
                            "\"결과 시뮬레이션\"을 실행할 수 있습니다.",
        "bad_model": "잘못된 모델 파일:\n{err}",
        "report_saved": "보고서 생성 완료:\n{path}",
        "report_err": "PDF를 생성할 수 없습니다:\n{err}",
        "tab_chart": "그래프 {i}", "tab_dist": "분포",
        "tab_heat": "히트맵", "tab_imp": "중요도", "tab_y": "y: {t}",
        "real": "(실제)", "predicted": "(예측)",
        "ideal_line": "이상선", "count": "빈도",
        "corr_title": "상관 행렬 (Pearson)",
        "imp_title": "변수 중요도", "imp_axis": "중요도",
        "shap_title": "SHAP 분석 \u2014 {t}",
        "pdf_summary": "데이터 요약", "pdf_source": "원본 파일: {file}",
        "pdf_dims": "{r}행 \u00d7 {c}열",
        "pdf_variable": "변수", "pdf_min": "최소", "pdf_max": "최대",
        "pdf_mean": "평균", "pdf_std": "표준편차",
        "pdf_corr": "상관관계", "pdf_model": "모델: {name}",
        "pdf_hparam": "하이퍼파라미터", "pdf_value": "값",
        "pdf_none": "없음", "pdf_inputs": "입력 (X)",
        "pdf_outputs": "출력 (y)", "pdf_target": "목표",
        "pdf_pred_page": "예측값 vs 실제값",
        "ft_all": "모든 형식", "ft_text": "텍스트",
        "ft_model": "ArborML 모델", "ft_allfiles": "모든 파일",
        "p_test_size": "테스트 비율", "p_random_state": "랜덤 시드",
        "p_max_depth": "최대 깊이",
        "p_min_split": "최소 분할 샘플 수",
        "p_min_leaf": "최소 리프 샘플 수",
        "p_n_estimators": "트리 개수",
        "p_max_features": "최대 특성 비율 (0\u20131)",
        "p_bootstrap": "부트스트랩 (예/아니오)",
        "p_learning_rate": "학습률", "p_subsample": "서브샘플",
        "p_fit_intercept": "절편 적합 (예/아니오)",
        "p_positive": "양수 계수 (예/아니오)",
        "p_colsample": "Colsample bytree", "p_reg_alpha": "Alpha (L1)",
        "p_reg_lambda": "Lambda (L2)",
        "val_invalid": "\"{label}\": 잘못된 값 ({raw}).",
        "val_min": "\"{label}\"은(는) {v} 이상이어야 합니다.",
        "val_max": "\"{label}\"은(는) {v} 이하여야 합니다.",
        "pdf_sims": "실행한 시뮬레이션", "pdf_sim_n": "번호",
        "pdf_sim_time": "시각",
        "pdf_sim_ask": "이 세션의 시뮬레이션 {n}건을 보고서에 포함할까요?",
        "sim_count": "시뮬레이션 {n}건 기록됨",
        "sim_clear": "시뮬레이션 기록 지우기",
        "sim_cleared": "시뮬레이션 기록을 지웠습니다.",
        "prog_read": "파일을 읽는 중",
        "prog_clean": "숫자 데이터 선별 중",
        "prog_display": "화면 갱신 중",
        "prog_write": "파일 저장 중",
        "prog_prepare": "데이터 준비 중",
        "prog_fit": "모델 학습 중",
        "prog_figure": "그래프: {t}",
        "prog_figures": "그래프 생성 중",
        "prog_importance": "변수 중요도",
        "prog_cv": "교차 검증",
        "prog_shap": "SHAP: {t}",
        "prog_compute": "계산 중",
        "prog_pdf_data": "데이터 페이지",
        "prog_pdf_model": "모델 페이지",
        "prog_pdf_sim": "시뮬레이션 페이지",
        "prog_pdf_save": "PDF 마무리 중",
        "prog_save_model": "모델 저장 중",
        "prog_load_model": "모델 불러오는 중",
        "prog_done": "완료",
    },
}

TRUE_TOKENS = {"1", "true", "yes", "y", "oui", "o", "vrai", "sí", "si",
               "ja", "wahr", "是", "はい", "예"}

PDF_CID_FONTS = {"zh": "STSong-Light", "ja": "HeiseiMin-W3",
                 "ko": "HYSMyeongJo-Medium"}

MPL_CJK_FONTS = {
    "zh": ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"],
    "ja": ["Yu Gothic", "Meiryo", "Hiragino Sans", "Noto Sans CJK JP"],
    "ko": ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"],
}
_MPL_DEFAULT_SANS = list(matplotlib.rcParams["font.sans-serif"])


def tr(key: str, **kwargs) -> str:
    table = TRANSLATIONS.get(CURRENT_LANG, TRANSLATIONS["en"])
    text = table.get(key) or TRANSLATIONS["en"].get(key, key)
    return text.format(**kwargs) if kwargs else text


def apply_plot_fonts() -> None:
    matplotlib.rcParams["axes.unicode_minus"] = False
    fams = MPL_CJK_FONTS.get(CURRENT_LANG)
    if fams:
        matplotlib.rcParams["font.sans-serif"] = fams + _MPL_DEFAULT_SANS
    else:
        matplotlib.rcParams["font.sans-serif"] = _MPL_DEFAULT_SANS


def pdf_fonts() -> tuple[str, str]:
    cid = PDF_CID_FONTS.get(CURRENT_LANG)
    if cid:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(cid))
            return cid, cid
        except Exception:
            pass
    return "Helvetica", "Helvetica-Bold"


def set_language(code: str) -> None:
    global CURRENT_LANG
    if code in LANGUAGES:
        CURRENT_LANG = code
        apply_plot_fonts()


def load_config() -> None:
    global ICON_STYLE
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        set_language(data.get("language", DEFAULT_LANG))
        style = str(data.get("icon_style", "auto"))
        ICON_STYLE = style if style in ("auto", "emoji", "symbols",
                                        "text") else "auto"
    except Exception:
        set_language(DEFAULT_LANG)
        ICON_STYLE = "auto"


def save_config() -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"language": CURRENT_LANG,
                       "icon_style": ICON_STYLE}, f)
    except Exception:
        pass


class BaseProgress:
    def __init__(self, total: float = 1.0):
        self.total = max(float(total), 1.0)
        self.done = 0.0

    def _paint(self, fraction: float, message: str | None = None) -> None:
        raise NotImplementedError

    def _pump(self) -> None:
        pass

    def set(self, fraction: float, message: str | None = None) -> None:
        fraction = min(max(float(fraction), 0.0), 1.0)
        self.done = fraction * self.total
        self._paint(fraction, message)
        self._pump()

    def step(self, message: str | None = None, inc: float = 1.0) -> None:
        self.done = min(self.done + inc, self.total)
        self._paint(self.done / self.total, message)
        self._pump()

    def run(self, func, *args, **kwargs):
        box: dict[str, object] = {}

        def worker():
            try:
                box["value"] = func(*args, **kwargs)
            except BaseException as exc:
                box["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        start = self.done / self.total
        span = min(1.0 / self.total, max(1.0 - start, 0.0))
        t0 = time.monotonic()
        while thread.is_alive():
            elapsed = time.monotonic() - t0
            self._paint(start + span * (1.0 - 1.0 / (1.0 + elapsed / 3.0)))
            self._pump()
            time.sleep(0.05)
        thread.join()
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def close(self) -> None:
        pass


class NullProgress(BaseProgress):
    def _paint(self, fraction, message=None):
        pass

    def run(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class ProgressDialog(BaseProgress):
    def __init__(self, master, title: str, total: float = 1.0):
        super().__init__(total)
        self._closed = False
        self.win = ctk.CTkToplevel(master)
        self.win.title(f"{APP_NAME} \u2014 {title}")
        self.win.geometry("430x150")
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", lambda: None)

        ctk.CTkLabel(self.win, text=title,
                     font=ctk.CTkFont(size=15, weight="bold")
                     ).pack(anchor="w", padx=20, pady=(20, 2))
        self._msg = ctk.CTkLabel(self.win, text="", anchor="w",
                                 text_color="gray")
        self._msg.pack(fill="x", padx=20)
        self._bar = ctk.CTkProgressBar(self.win, height=14,
                                       progress_color=ACCENT)
        self._bar.pack(fill="x", padx=20, pady=(12, 4))
        self._bar.set(0)
        self._pct = ctk.CTkLabel(self.win, text="0 %", anchor="e",
                                 font=ctk.CTkFont(size=11))
        self._pct.pack(fill="x", padx=20)

        self.win.transient(master)
        self.win.update_idletasks()
        self._center_on(master)
        try:
            self.win.grab_set()
        except Exception:
            pass
        self.win.lift()
        self._pump()

    def _center_on(self, master):
        try:
            x = (master.winfo_rootx()
                 + (master.winfo_width() - self.win.winfo_width()) // 2)
            y = (master.winfo_rooty()
                 + (master.winfo_height() - self.win.winfo_height()) // 3)
            self.win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        except Exception:
            pass

    def _paint(self, fraction, message=None):
        if self._closed:
            return
        self._bar.set(fraction)
        self._pct.configure(text=f"{fraction * 100:.0f} %")
        if message is not None:
            self._msg.configure(text=message)

    def _pump(self):
        if self._closed:
            return
        try:
            self.win.update()
        except Exception:
            pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.win.grab_release()
        except Exception:
            pass
        self.win.destroy()


class InlineProgress(BaseProgress):
    def __init__(self, bar, label=None, total: float = 1.0):
        super().__init__(total)
        self.bar, self.label = bar, label
        self._paint(0.0)

    def _paint(self, fraction, message=None):
        try:
            self.bar.set(fraction)
            if message is not None and self.label is not None:
                self.label.configure(text=message)
        except Exception:
            pass

    def _pump(self):
        try:
            self.bar.update()
        except Exception:
            pass


@contextmanager
def progress_dialog(master, title: str, total: float = 1.0):
    dlg = ProgressDialog(master, title, total)
    try:
        yield dlg
    finally:
        dlg.close()


@dataclass
class Param:
    key: str
    label_key: str
    ptype: type
    default: object
    minv: float | None = None
    maxv: float | None = None
    allow_none: bool = False

    @property
    def label(self) -> str:
        return tr(self.label_key)

    def parse(self, raw: str):
        raw = (raw or "").strip().replace(",", ".")
        if raw == "":
            return None if self.allow_none else self.default
        if self.ptype is bool:
            return raw.lower() in TRUE_TOKENS
        try:
            value = self.ptype(float(raw)) if self.ptype is int else self.ptype(raw)
        except ValueError:
            raise ValueError(tr("val_invalid", label=self.label, raw=raw))
        if self.minv is not None and value < self.minv:
            raise ValueError(tr("val_min", label=self.label, v=self.minv))
        if self.maxv is not None and value > self.maxv:
            raise ValueError(tr("val_max", label=self.label, v=self.maxv))
        return value


COMMON_PARAMS = [
    Param("test_size", "p_test_size", float, 0.2, 0.05, 0.95),
    Param("random_state", "p_random_state", int, 42, 0, 2**32 - 1),
]


MODEL_SPECS: dict[str, dict] = {
    "Linear Regression": {
        "native_multioutput": False,
        "params": [
            Param("fit_intercept", "p_fit_intercept", bool, True),
            Param("positive", "p_positive", bool, False),
        ] + COMMON_PARAMS,
        "builder": lambda p: LinearRegression(
            fit_intercept=p["fit_intercept"],
            positive=p["positive"],
        ),
    },
    "Decision Tree": {
        "params": [
            Param("max_depth", "p_max_depth", int, None, 1, allow_none=True),
            Param("min_samples_split", "p_min_split", int, 2, 2),
            Param("min_samples_leaf", "p_min_leaf", int, 1, 1),
        ] + COMMON_PARAMS,
        "builder": lambda p: DecisionTreeRegressor(
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=p["random_state"],
        ),
    },
    "Random Forest": {
        "params": [
            Param("n_estimators", "p_n_estimators", int, 200, 1),
            Param("max_depth", "p_max_depth", int, None, 1, allow_none=True),
            Param("min_samples_split", "p_min_split", int, 2, 2),
            Param("min_samples_leaf", "p_min_leaf", int, 1, 1),
            Param("max_features", "p_max_features", float, 1.0, 0.01, 1.0),
            Param("bootstrap", "p_bootstrap", bool, True),
        ] + COMMON_PARAMS,
        "builder": lambda p: RandomForestRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"],
            bootstrap=p["bootstrap"],
            random_state=p["random_state"],
            n_jobs=-1,
        ),
    },
    "Gradient Boosting": {
        "native_multioutput": False,
        "params": [
            Param("n_estimators", "p_n_estimators", int, 200, 1),
            Param("learning_rate", "p_learning_rate", float, 0.1, 1e-4, 1.0),
            Param("max_depth", "p_max_depth", int, 3, 1),
            Param("min_samples_split", "p_min_split", int, 2, 2),
            Param("min_samples_leaf", "p_min_leaf", int, 1, 1),
            Param("subsample", "p_subsample", float, 1.0, 0.05, 1.0),
            Param("max_features", "p_max_features", float, 1.0, 0.01, 1.0),
        ] + COMMON_PARAMS,
        "builder": lambda p: GradientBoostingRegressor(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            subsample=p["subsample"],
            max_features=p["max_features"],
            random_state=p["random_state"],
        ),
    },
}

if XGB_AVAILABLE:
    MODEL_SPECS["XGBoost"] = {
        "native_multioutput": False,
        "params": [
            Param("n_estimators", "p_n_estimators", int, 200, 1),
            Param("max_depth", "p_max_depth", int, 6, 1),
            Param("learning_rate", "p_learning_rate", float, 0.05, 1e-4, 1.0),
            Param("subsample", "p_subsample", float, 0.8, 0.05, 1.0),
            Param("colsample_bytree", "p_colsample", float, 0.8, 0.05, 1.0),
            Param("reg_alpha", "p_reg_alpha", float, 0.0, 0.0),
            Param("reg_lambda", "p_reg_lambda", float, 1.0, 0.0),
        ] + COMMON_PARAMS,
        "builder": lambda p: XGBRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            reg_alpha=p["reg_alpha"],
            reg_lambda=p["reg_lambda"],
            random_state=p["random_state"],
            n_jobs=-1,
        ),
    }

class DataManager:
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.source_name: str = ""
        self.dropped_rows: int = 0
        self.dropped_cols: int = 0

    @property
    def loaded(self) -> bool:
        return self.df is not None and not self.df.empty

    @property
    def has_dropped(self) -> bool:
        return bool(self.dropped_rows or self.dropped_cols)

    @property
    def cleaning_note(self) -> str:
        if not self.has_dropped:
            return ""
        return tr("data_cleaned", r=self.dropped_rows, c=self.dropped_cols)

    @staticmethod
    def needs_separator(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in (".csv", ".dat")

    def load(self, path: str, parent=None, sep: str | None = None,
             progress: BaseProgress | None = None) -> None:
        prog = progress or NullProgress()
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".dat") and sep is None and parent is not None:
            sep = simpledialog.askstring(
                tr("sep_title"), tr("sep_msg"), parent=parent)

        def read_frame() -> pd.DataFrame:
            if ext in (".xlsx", ".xls"):
                return pd.read_excel(path)
            if ext == ".txt":
                return pd.read_csv(path, sep="\t")
            if ext in (".csv", ".dat"):
                s = sep or ","
                s = "\t" if s.lower() in ("tab", "\\t") else s
                return pd.read_csv(path, sep=s)
            raise ValueError(tr("bad_format", ext=ext))

        def to_numeric(frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.apply(pd.to_numeric, errors="coerce")
            return frame.dropna(axis=1, how="all")

        prog.step(tr("prog_read"))
        df = prog.run(read_frame)
        prog.step(tr("prog_clean"))
        n_rows_in, n_cols_in = df.shape
        df = prog.run(to_numeric, df)
        dropped_cols = n_cols_in - df.shape[1]
        df = df.dropna(axis=0, how="any")
        dropped_rows = n_rows_in - df.shape[0]
        if df.empty:
            raise ValueError(tr("no_numeric"))
        df.columns = [str(c) for c in df.columns]
        self.df = df.reset_index(drop=True)
        self.source_name = os.path.basename(path)
        self.dropped_rows, self.dropped_cols = dropped_rows, dropped_cols
        if self.has_dropped:
            logger.warning("%s : %s", self.source_name, self.cleaning_note)

    def rename(self, mapping: dict[str, str]) -> None:
        self.df = self.df.rename(columns=mapping)

    def to_json(self, path: str) -> None:
        payload = {
            "app": f"{APP_NAME} {APP_VERSION}",
            "source": self.source_name,
            "columns": list(self.df.columns),
            "values": self.df.to_dict(orient="list"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def from_json(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        self.df = pd.DataFrame(payload["values"])[payload["columns"]]
        self.source_name = payload.get("source", os.path.basename(path))
        self.dropped_rows = self.dropped_cols = 0


METRIC_COLS = ["R²", "RMSE", "MSE", "MAE"]
METRIC_FORMATS = {"R²": "{:.4f}", "RMSE": "{:.4g}",
                  "MSE": "{:.4g}", "MAE": "{:.4g}"}


def format_metric(name: str, value) -> str:
    return METRIC_FORMATS.get(name, "{:.4g}").format(float(value))


def metrics_caption(scores) -> str:
    return "\n".join(f"{name:<4} = {format_metric(name, scores[name])}"
                     for name in METRIC_COLS if name in scores)


def metrics_line(scores, sep: str = "   ") -> str:
    return sep.join(f"{name} = {format_metric(name, scores[name])}"
                    for name in METRIC_COLS if name in scores)


def ensure_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics.copy()
    if "RMSE" not in m.columns and "MSE" in m.columns:
        m["RMSE"] = np.sqrt(m["MSE"].astype(float))
    ordered = [c for c in METRIC_COLS if c in m.columns]
    return m[ordered + [c for c in m.columns if c not in ordered]]


@dataclass
class TrainedModel:
    name: str
    estimator: object
    scaler: StandardScaler
    x_cols: list[str]
    y_cols: list[str]
    params: dict
    metrics: pd.DataFrame
    pred_figs: list[plt.Figure] = field(default_factory=list)
    importance: pd.Series | None = None
    cv_mean: float | None = None
    cv_std: float | None = None
    cv_rmse_mean: float | None = None
    cv_rmse_std: float | None = None
    shap_figs: list[tuple[str, plt.Figure]] = field(default_factory=list)

    def save(self, path: str) -> None:
        joblib.dump({
            "app": f"{APP_NAME} {APP_VERSION}",
            "name": self.name,
            "estimator": self.estimator,
            "scaler": self.scaler,
            "x_cols": self.x_cols,
            "y_cols": self.y_cols,
            "params": self.params,
            "metrics": self.metrics,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "cv_rmse_mean": self.cv_rmse_mean,
            "cv_rmse_std": self.cv_rmse_std,
        }, path)

    @classmethod
    def load(cls, path: str) -> "TrainedModel":
        d = joblib.load(path)
        return cls(d["name"], d["estimator"], d["scaler"], d["x_cols"],
                   d["y_cols"], d["params"], ensure_metrics(d["metrics"]),
                   cv_mean=d.get("cv_mean"), cv_std=d.get("cv_std"),
                   cv_rmse_mean=d.get("cv_rmse_mean"),
                   cv_rmse_std=d.get("cv_rmse_std"))

    def predict(self, values: list[float]) -> np.ndarray:
        X = self.scaler.transform(pd.DataFrame([values], columns=self.x_cols))
        return np.atleast_2d(self.estimator.predict(X))[0]


class MLEngine:
    @staticmethod
    def _build_estimator(model_name: str, params: dict, n_targets: int):
        est = MODEL_SPECS[model_name]["builder"](params)
        if n_targets > 1 and not MODEL_SPECS[model_name].get(
                "native_multioutput", True):
            est = MultiOutputRegressor(est)
        return est

    @staticmethod
    def score(true, pred) -> dict[str, float]:
        mse = float(mean_squared_error(true, pred))
        return {"R²": float(r2_score(true, pred)),
                "RMSE": float(np.sqrt(mse)),
                "MSE": mse,
                "MAE": float(mean_absolute_error(true, pred))}

    @staticmethod
    def n_steps(y_cols: list[str], do_cv: bool = False,
                do_shap: bool = False) -> int:
        n = len(y_cols)
        return (3 + n + (1 if do_cv else 0)
                + (n if do_shap and SHAP_AVAILABLE else 0))

    @staticmethod
    def train(df: pd.DataFrame, x_cols: list[str], y_cols: list[str],
              model_name: str, params: dict, do_cv: bool = False,
              do_shap: bool = False,
              progress: BaseProgress | None = None) -> TrainedModel:
        prog = progress or NullProgress()
        prog.step(tr("prog_prepare"))
        X, y = df[x_cols], df[y_cols]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=params["test_size"],
            random_state=params["random_state"])

        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        est = MLEngine._build_estimator(model_name, params, len(y_cols))
        prog.step(tr("prog_fit"))
        prog.run(est.fit,
                 X_tr_s, y_tr.values if len(y_cols) > 1
                 else y_tr.values.ravel())

        y_pred = np.atleast_2d(est.predict(X_te_s))
        if y_pred.shape[0] == len(y_cols) and y_pred.shape[0] != len(y_te):
            y_pred = y_pred.T
        y_pred = y_pred.reshape(len(y_te), len(y_cols))

        rows, figs = [], []
        for j, target in enumerate(y_cols):
            prog.step(tr("prog_figure", t=target))
            true_j, pred_j = y_te.iloc[:, j].to_numpy(), y_pred[:, j]
            scores = MLEngine.score(true_j, pred_j)
            rows.append(scores)
            figs.append(MLEngine._pred_figure(true_j, pred_j, target,
                                              model_name, scores))

        metrics = pd.DataFrame(rows, index=y_cols)[METRIC_COLS]
        prog.step(tr("prog_importance"))
        importance = MLEngine._importance(est, x_cols)
        model = TrainedModel(model_name, est, scaler, x_cols, y_cols,
                             params, metrics, figs, importance)
        if do_cv:
            prog.step(tr("prog_cv"))
            (model.cv_mean, model.cv_std,
             model.cv_rmse_mean, model.cv_rmse_std) = prog.run(
                MLEngine._cross_validate, X, y, y_cols, model_name, params)
        if do_shap and SHAP_AVAILABLE:
            model.shap_figs = MLEngine._shap_figures(est, scaler, X,
                                                     x_cols, y_cols, prog)
        return model

    @staticmethod
    def _cross_validate(X, y, y_cols, model_name, params, n_splits: int = 5
                        ) -> tuple[float, float, float, float]:
        est = MLEngine._build_estimator(model_name, params, len(y_cols))
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
        cv = KFold(n_splits=min(n_splits, len(X)), shuffle=True,
                   random_state=params["random_state"])
        y_arr = y.values if len(y_cols) > 1 else y.values.ravel()
        res = cross_validate(pipe, X, y_arr, cv=cv,
                             scoring=("r2", "neg_root_mean_squared_error"))
        r2 = res["test_r2"]
        rmse = -res["test_neg_root_mean_squared_error"]
        return (float(r2.mean()), float(r2.std()),
                float(rmse.mean()), float(rmse.std()))

    @staticmethod
    def _shap_figures(est, scaler, X, x_cols, y_cols,
                      progress: BaseProgress | None = None
                      ) -> list[tuple[str, plt.Figure]]:
        prog = progress or NullProgress()
        figs = []
        sample = X.sample(min(len(X), 300), random_state=0)
        X_s = pd.DataFrame(scaler.transform(sample), columns=x_cols)
        estimators = (est.estimators_
                      if isinstance(est, MultiOutputRegressor) else [est])
        for j, target in enumerate(y_cols):
            fig = None
            prog.step(tr("prog_shap", t=target))
            try:
                sub = estimators[j] if len(estimators) > 1 else estimators[0]

                def shap_values(sub=sub):
                    explainer = (shap.TreeExplainer(sub)
                                 if hasattr(sub, "feature_importances_")
                                 else shap.LinearExplainer(sub, X_s))
                    return explainer.shap_values(X_s)
                sv = prog.run(shap_values)
                if isinstance(sv, list):
                    sv = sv[j]
                elif getattr(sv, "ndim", 2) == 3:
                    sv = sv[:, :, j]
                fig = plt.figure(figsize=(7.5, 0.45 * len(x_cols) + 2.2),
                                 dpi=110)
                shap.summary_plot(sv, X_s, show=False, plot_size=None)
                fig.suptitle(tr("shap_title", t=target), fontsize=13,
                             fontweight="bold")
                fig.tight_layout()
                figs.append((f"SHAP : {target}", fig))
            except Exception:
                logger.warning("Analyse SHAP impossible pour la cible %r ; "
                               "les autres sorties sont conservées.",
                               target, exc_info=True)
                if fig is not None:
                    plt.close(fig)
                continue
        return figs

    @staticmethod
    def _pred_figure(true, pred, target, model_name, scores) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=110)
        lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
        pad = 0.03 * (hi - lo or 1)
        ax.scatter(true, pred, s=38, c="#1b1b1b", marker="s", alpha=0.85)
        ax.plot([lo, hi], [lo, hi], c="#C0392B", lw=1.8,
                label=tr("ideal_line"))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"{target} {tr('real')}", fontsize=12,
                      fontweight="bold")
        ax.set_ylabel(f"{target} {tr('predicted')}", fontsize=12,
                      fontweight="bold")
        ax.set_title(f"{model_name} \u2014 R² = "
                     f"{format_metric('R²', scores['R²'])}", fontsize=13)
        ax.text(0.97, 0.03, metrics_caption(scores), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor="#9E9E9E", alpha=0.88))
        ax.legend(frameon=False, loc="upper left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    @staticmethod
    def _coef_importance(est, x_cols) -> np.ndarray | None:
        coef = getattr(est, "coef_", None)
        if coef is None:
            return None
        coef = np.abs(np.atleast_2d(coef).astype(float))
        sums = coef.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return (coef / sums).mean(axis=0)

    @staticmethod
    def _importance(est, x_cols) -> pd.Series | None:
        subs = (est.estimators_
                if isinstance(est, MultiOutputRegressor) else [est])
        arrays = []
        for s in subs:
            if hasattr(s, "feature_importances_"):
                arrays.append(np.asarray(s.feature_importances_,
                                         dtype=float))
            else:
                arr = MLEngine._coef_importance(s, x_cols)
                if arr is None:
                    return None
                arrays.append(arr)
        return pd.Series(np.mean(arrays, axis=0),
                         index=x_cols).sort_values(ascending=False)

@dataclass
class Simulation:
    model_name: str
    x_cols: list[str]
    y_cols: list[str]
    x: list[float]
    y: list[float]
    time: datetime = field(default_factory=datetime.now)

    def matches(self, model: TrainedModel | None) -> bool:
        return (model is not None
                and self.x_cols == list(model.x_cols)
                and self.y_cols == list(model.y_cols))


def fig_to_buffer(fig: plt.Figure, dpi=200) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf


def histograms_figure(df, ncols=2):
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    nrows = math.ceil(len(cols) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(11, 2.9 * nrows),
        layout="constrained",
    )

    fig.get_layout_engine().set(h_pad=0.10, w_pad=0.06, hspace=0.14, wspace=0.08)

    axes = axes.ravel()
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(cols)))

    for ax, col, color in zip(axes, cols, colors):
        ax.hist(df[col].dropna(), bins=20, color=color, edgecolor="none")
        ax.set_xlabel(col, fontsize=9, labelpad=3)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="both", alpha=0.25, linewidth=0.6)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    return fig

def heatmap_figure(df: pd.DataFrame) -> plt.Figure:
    corr = df.corr(method="pearson", numeric_only=True).round(2)
    n = len(corr)

    fig, ax = plt.subplots(figsize=(1.05 * n + 3.0, 0.95 * n + 2.2),
                           dpi=110, layout="constrained")

    sns.heatmap(corr, cmap="YlGnBu", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", linewidths=0.5,
                annot_kws={"size": 9}, square=True,
                cbar_kws={"shrink": 0.85}, ax=ax)

    ax.set_title(tr("corr_title"), fontsize=13, pad=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontweight="bold", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontweight="bold", fontsize=9)
    return fig

def importance_figure(imp: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(imp) + 2), dpi=110)
    bars = ax.barh(imp.index[::-1], imp.values[::-1],
                   color=sns.color_palette("crest", len(imp)),
                   edgecolor="black", linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width() * 100:.1f} %", va="center", fontsize=10)
    ax.set_xlabel(tr("imp_axis"), fontweight="bold")
    ax.set_title(tr("imp_title"), fontsize=13, fontweight="bold")
    ax.set_xlim(0, imp.max() * 1.18)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig

class ReportBuilder:
    SIM_ROWS_PER_PAGE = 24
    SIM_COLS_PER_TABLE = 8
    SIM_HEAD_MAX_CHARS = 14

    def __init__(self, dm: DataManager, model: TrainedModel | None,
                 simulations: list[Simulation] | None = None):
        self.dm, self.model = dm, model
        self.simulations = list(simulations or [])
        self.font, self.font_bold = pdf_fonts()

    def _table_style(self, font_size: int = 9) -> TableStyle:
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), self.font_bold),
            ("FONTNAME", (0, 1), (-1, -1), self.font),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F4F1EA")),
            ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ])

    def _header(self, c, title):
        w, h = letter
        c.setFillColor(colors.HexColor(ACCENT))
        c.rect(0, h - 0.55 * inch, w, 0.55 * inch, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(self.font_bold, 14)
        c.drawString(0.7 * inch, h - 0.38 * inch, title)
        c.setFont(self.font, 8)
        c.drawRightString(w - 0.5 * inch, h - 0.38 * inch,
                          f"{APP_NAME} {APP_VERSION} \u2014 "
                          f"{datetime.now():%d/%m/%Y %H:%M}")
        c.setFillColor(colors.black)
        return h - 0.9 * inch

    def _table(self, c, data, col_widths, y, font_size: int = 9):
        t = Table(data, colWidths=col_widths)
        t.setStyle(self._table_style(font_size))
        _, th = t.wrap(letter[0], letter[1])
        t.drawOn(c, 0.7 * inch, y - th)
        return y - th - 0.3 * inch

    def _image_page(self, c, fig, title):
        y = self._header(c, title)
        buf = fig_to_buffer(fig)
        img = ImageReader(buf)
        iw, ih = img.getSize()
        w, h = letter
        max_w, max_h = w - 1.4 * inch, y - 0.6 * inch
        scale = min(max_w / iw, max_h / ih)
        c.drawImage(img, (w - iw * scale) / 2, y - ih * scale,
                    width=iw * scale, height=ih * scale)
        c.showPage()

    @property
    def _sims(self) -> list[Simulation]:
        return [s for s in self.simulations if s.matches(self.model)]

    def _sim_layout(self) -> tuple[list[str], list[list[str]]]:
        m = self.model
        heads = list(m.x_cols) + [f"{t} {tr('predicted')}" for t in m.y_cols]
        heads = [h if len(h) <= self.SIM_HEAD_MAX_CHARS
                 else h[:self.SIM_HEAD_MAX_CHARS - 1] + "\u2026" for h in heads]
        rows = [[f"{v:.6g}" for v in list(s.x) + list(s.y)]
                for s in self._sims]
        return heads, rows

    def _sim_page_count(self) -> int:
        sims = self._sims
        if not sims:
            return 0
        n_cols = len(self.model.x_cols) + len(self.model.y_cols)
        n_tables = -(-n_cols // self.SIM_COLS_PER_TABLE)
        n_blocks = -(-len(sims) // self.SIM_ROWS_PER_PAGE)
        return n_tables * n_blocks

    def _simulation_pages(self, c, prog):
        sims = self._sims
        if not sims:
            return
        heads, rows = self._sim_layout()
        times = [f"{s.time:%H:%M:%S}" for s in sims]
        step = self.SIM_COLS_PER_TABLE
        for k in range(0, len(heads), step):
            sub = heads[k:k + step]
            for r0 in range(0, len(rows), self.SIM_ROWS_PER_PAGE):
                prog.step(tr("prog_pdf_sim"))
                y = self._header(c, tr("pdf_sims"))
                c.setFont(self.font, 10)
                c.drawString(0.7 * inch, y,
                             tr("pdf_model", name=self.model.name))
                data = [[tr("pdf_sim_n"), tr("pdf_sim_time")] + sub]
                for i in range(r0, min(r0 + self.SIM_ROWS_PER_PAGE,
                                       len(rows))):
                    data.append([str(i + 1), times[i]]
                                + rows[i][k:k + step])
                fixed = 0.45 * inch + 0.75 * inch
                w = (letter[0] - 1.4 * inch - fixed) / len(sub)
                self._table(c, data,
                            [0.45 * inch, 0.75 * inch] + [w] * len(sub),
                            y - 24, font_size=7)
                c.showPage()

    def n_steps(self) -> int:
        n = 4
        if self.model:
            n += 1 + len(self.model.pred_figs) + len(self.model.shap_figs)
            n += 1 if self.model.importance is not None else 0
            n += self._sim_page_count()
        return n

    def build(self, path: str,
              progress: BaseProgress | None = None) -> None:
        prog = progress or NullProgress()
        c = pdfcanvas.Canvas(path, pagesize=letter)
        df = self.dm.df

        prog.step(tr("prog_pdf_data"))
        y = self._header(c, tr("pdf_summary"))
        c.setFont(self.font, 10)
        c.drawString(0.7 * inch, y,
                     tr("pdf_source", file=self.dm.source_name))
        c.drawString(0.7 * inch, y - 16,
                     tr("pdf_dims", r=df.shape[0], c=df.shape[1]))
        y_stats = y - 40
        if self.dm.has_dropped:
            c.drawString(0.7 * inch, y - 32, self.dm.cleaning_note)
            y_stats = y - 56
        stats = [[tr("pdf_variable"), tr("pdf_min"), tr("pdf_max"),
                  tr("pdf_mean"), tr("pdf_std")]]
        for col in df.columns:
            s = df[col]
            stats.append([col, f"{s.min():.3g}", f"{s.max():.3g}",
                          f"{s.mean():.3g}", f"{s.std():.3g}"])
        self._table(c, stats, [1.9 * inch] + [1.15 * inch] * 4, y_stats)
        c.showPage()

        prog.step(tr("tab_dist"))
        self._image_page(c, histograms_figure(df), tr("tab_dist"))
        prog.step(tr("pdf_corr"))
        self._image_page(c, heatmap_figure(df), tr("pdf_corr"))

        if self.model:
            m = self.model
            prog.step(tr("prog_pdf_model"))
            y = self._header(c, tr("pdf_model", name=m.name))
            pdata = [[tr("pdf_hparam"), tr("pdf_value")]] + \
                    [[k, tr("pdf_none") if v is None else str(v)]
                     for k, v in m.params.items()]
            y = self._table(c, pdata, [2.4 * inch, 2.0 * inch], y)
            io_rows = max(len(m.x_cols), len(m.y_cols))
            iodata = [[tr("pdf_inputs"), tr("pdf_outputs")]] + [
                [m.x_cols[i] if i < len(m.x_cols) else "",
                 m.y_cols[i] if i < len(m.y_cols) else ""]
                for i in range(io_rows)]
            y = self._table(c, iodata, [2.4 * inch, 2.4 * inch], y)
            cols = [k for k in METRIC_COLS if k in m.metrics.columns]
            mdata = [[tr("pdf_target")] + cols] + [
                [t] + [format_metric(k, r[k]) for k in cols]
                for t, r in m.metrics.iterrows()]
            y = self._table(c, mdata,
                            [1.7 * inch] + [1.2 * inch] * len(cols), y)
            if m.cv_mean is not None:
                c.setFont(self.font_bold, 10)
                line = (f"{tr('cv_line')} : "
                        f"R² = {m.cv_mean:.4f} ± {m.cv_std:.4f}")
                if m.cv_rmse_mean is not None:
                    line += (f"    RMSE = {m.cv_rmse_mean:.4g} "
                             f"± {m.cv_rmse_std:.4g}")
                c.drawString(0.7 * inch, y, line)
            c.showPage()

            for fig in m.pred_figs:
                prog.step(tr("pdf_pred_page"))
                self._image_page(c, fig, tr("pdf_pred_page"))
            if m.importance is not None:
                prog.step(tr("imp_title"))
                self._image_page(c, importance_figure(m.importance),
                                 tr("imp_title"))
            for label, fig in m.shap_figs:
                prog.step(label)
                self._image_page(c, fig, label)
            self._simulation_pages(c, prog)
        prog.step(tr("prog_pdf_save"))
        prog.run(c.save)


class PlotWindow(ctk.CTkToplevel):

    def __init__(self, master, title: str, figs: list[plt.Figure],
                 labels: list[str] | None = None):
        super().__init__(master)
        self.title(f"{APP_NAME} \u2014 {title}")
        self.geometry("880x720")
        tabs = ctk.CTkTabview(self)
        tabs.pack(fill="both", expand=True, padx=8, pady=8)
        labels = labels or [tr("tab_chart", i=i + 1)
                            for i in range(len(figs))]
        for fig, label in zip(figs, labels):
            tab = tabs.add(label)
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        self.after(80, self.lift)


class ModelDialog(ctk.CTkToplevel):

    def __init__(self, app: "ArborMLApp", model_name: str):
        super().__init__(app)
        self.app, self.model_name = app, model_name
        self.title(f"{APP_NAME} \u2014 {model_name}")
        self.geometry("460x640")
        self.entries: dict[str, ctk.CTkEntry] = {}
        self.x_vars: dict[str, ctk.BooleanVar] = {}
        self.y_vars: dict[str, ctk.BooleanVar] = {}

        scroll = ctk.CTkScrollableFrame(self)
        scroll.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        ctk.CTkLabel(scroll, text=tr("hyperparams"),
                     font=ctk.CTkFont(size=15, weight="bold")
                     ).pack(anchor="w", pady=(4, 6))
        for p in MODEL_SPECS[model_name]["params"]:
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=p.label, width=170, anchor="w"
                         ).pack(side="left")
            default_txt = tr("auto") if p.default is None else str(p.default)
            e = ctk.CTkEntry(row, placeholder_text=default_txt, width=120)
            e.pack(side="left", padx=6)
            self.entries[p.key] = e

        ctk.CTkLabel(scroll, text=tr("options"),
                     font=ctk.CTkFont(size=15, weight="bold")
                     ).pack(anchor="w", pady=(14, 4))
        self.cv_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(scroll, text=tr("cv_check"),
                        variable=self.cv_var).pack(anchor="w", padx=12,
                                                   pady=1)
        self.shap_var = ctk.BooleanVar(value=SHAP_AVAILABLE)
        cb_shap = ctk.CTkCheckBox(scroll, text=tr("shap_check"),
                                  variable=self.shap_var)
        cb_shap.pack(anchor="w", padx=12, pady=1)
        if not SHAP_AVAILABLE:
            self.shap_var.set(False)
            cb_shap.configure(state="disabled", text=tr("shap_missing"))

        cols = list(app.dm.df.columns)
        for title, store in ((tr("x_vars"), self.x_vars),
                             (tr("y_vars"), self.y_vars)):
            ctk.CTkLabel(scroll, text=title,
                         font=ctk.CTkFont(size=15, weight="bold")
                         ).pack(anchor="w", pady=(14, 4))
            for col in cols:
                var = ctk.BooleanVar(value=False)
                ctk.CTkCheckBox(scroll, text=col, variable=var
                                ).pack(anchor="w", padx=12, pady=1)
                store[col] = var

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(btns, text=tr("train"),
                      fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      command=self.train).pack(side="left", expand=True,
                                               fill="x", padx=(0, 5))
        ctk.CTkButton(btns, text=tr("close"), fg_color="gray40",
                      command=self.destroy).pack(side="left", expand=True,
                                                 fill="x", padx=(5, 0))
        self.after(80, self.lift)

    def train(self):
        try:
            x_cols = [c for c, v in self.x_vars.items() if v.get()]
            y_cols = [c for c, v in self.y_vars.items() if v.get()]
            if not x_cols or not y_cols:
                raise ValueError(tr("need_xy"))
            if set(x_cols) & set(y_cols):
                raise ValueError(tr("xy_overlap"))
            params = {p.key: p.parse(self.entries[p.key].get())
                      for p in MODEL_SPECS[self.model_name]["params"]}
            do_cv, do_shap = self.cv_var.get(), self.shap_var.get()
            steps = MLEngine.n_steps(y_cols, do_cv, do_shap) + 1
            with progress_dialog(self, tr("train"), steps) as prog:
                model = MLEngine.train(self.app.dm.df, x_cols, y_cols,
                                       self.model_name, params,
                                       do_cv=do_cv, do_shap=do_shap,
                                       progress=prog)
                prog.step(tr("prog_figures"))
                figs = list(model.pred_figs)
                labels = [tr("tab_y", t=t) for t in model.y_cols]
                if model.importance is not None:
                    figs.append(importance_figure(model.importance))
                    labels.append(tr("tab_imp"))
                for label, fig in model.shap_figs:
                    figs.append(fig)
                    labels.append(label)
                prog.set(1.0, tr("prog_done"))
        except ValueError as exc:
            messagebox.showerror(tr("bad_params"), str(exc), parent=self)
            return
        except Exception as exc:
            messagebox.showerror(tr("train_err"), str(exc), parent=self)
            return

        self.app.set_model(model)
        PlotWindow(self.app, tr("results", name=self.model_name),
                   figs, labels)

        resume = "\n".join(f"{t} :  {metrics_line(r)}"
                           for t, r in model.metrics.iterrows())
        if model.cv_mean is not None:
            resume += (f"\n\n{tr('cv_line')} : "
                       f"R² = {model.cv_mean:.4f} ± {model.cv_std:.4f}")
            if model.cv_rmse_mean is not None:
                resume += (f"\nRMSE = {model.cv_rmse_mean:.4g} "
                           f"± {model.cv_rmse_std:.4g}")
        messagebox.showinfo(tr("train_done"), resume, parent=self)


class PredictDialog(ctk.CTkToplevel):

    def __init__(self, app: "ArborMLApp"):
        super().__init__(app)
        self.app, m = app, app.model
        self.title(f"{APP_NAME} \u2014 {tr('sim_title', name=m.name)}")
        self.geometry("520x560")
        frame = ctk.CTkScrollableFrame(self)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(frame, text=tr("in_values"),
                     font=ctk.CTkFont(size=15, weight="bold")
                     ).pack(anchor="w", pady=(2, 6))
        self.inputs: dict[str, ctk.CTkEntry] = {}
        for col in m.x_cols:
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=col, width=220, anchor="w").pack(side="left")
            e = ctk.CTkEntry(row, width=140)
            e.pack(side="left", padx=6)
            self.inputs[col] = e

        ctk.CTkButton(frame, text=tr("predict"), fg_color=ACCENT,
                      hover_color=ACCENT_HOVER, command=self.run
                      ).pack(pady=(12, 6))
        self.bar = ctk.CTkProgressBar(frame, height=10,
                                      progress_color=ACCENT)
        self.bar.pack(fill="x", padx=4, pady=(0, 8))
        self.bar.set(0)

        ctk.CTkLabel(frame, text=tr("out_values"),
                     font=ctk.CTkFont(size=15, weight="bold")
                     ).pack(anchor="w", pady=(8, 6))
        self.outputs: dict[str, ctk.CTkEntry] = {}
        for col in m.y_cols:
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=col, width=220, anchor="w").pack(side="left")
            e = ctk.CTkEntry(row, width=140, state="disabled")
            e.pack(side="left", padx=6)
            self.outputs[col] = e
        self.count_label = ctk.CTkLabel(frame, text="", anchor="w",
                                        text_color="gray",
                                        font=ctk.CTkFont(size=11))
        self.count_label.pack(fill="x", pady=(10, 2))
        self.btn_clear = ctk.CTkButton(frame, text=tr("sim_clear"),
                                       fg_color="gray40",
                                       command=self.clear_history)
        self.btn_clear.pack(fill="x", pady=(0, 4))
        self._refresh_count()
        self.after(80, self.lift)

    def _refresh_count(self):
        n = len(self.app.simulations)
        self.count_label.configure(text=tr("sim_count", n=n))
        self.btn_clear.configure(state="normal" if n else "disabled")

    def clear_history(self):
        self.app.clear_simulations()
        self._refresh_count()

    def run(self):
        try:
            values = [float(self.inputs[c].get().replace(",", "."))
                      for c in self.app.model.x_cols]
        except ValueError:
            self.bar.set(0)
            messagebox.showerror(tr("error"), tr("all_numeric"), parent=self)
            return
        prog = InlineProgress(self.bar, total=2)
        prog.step(tr("prog_compute"))
        try:
            preds = prog.run(self.app.model.predict, values)
        except Exception as exc:
            prog.set(0)
            messagebox.showerror(tr("error"), str(exc), parent=self)
            return
        for col, val in zip(self.app.model.y_cols, np.atleast_1d(preds)):
            e = self.outputs[col]
            e.configure(state="normal")
            e.delete(0, "end")
            e.insert(0, f"{val:.6g}")
            e.configure(state="disabled")
        self.app.record_simulation(values, preds)
        self._refresh_count()
        prog.set(1.0)


class ArborMLApp(ctk.CTk):
    SIM_HISTORY_MAX = 2000

    def __init__(self):
        super().__init__()
        load_config()
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("green")
        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1120x680")
        self.minsize(940, 560)

        windows_taskbar_identity()
        ICONS.configure(self, ICON_STYLE)
        set_window_icon(self)

        self.dm = DataManager()
        self.model: TrainedModel | None = None
        self.simulations: list[Simulation] = []
        self._logo_images: list[ctk.CTkImage] = []
        self._rename_entries: dict[str, ctk.CTkEntry] = {}

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_ui()

    def _build_ui(self):
        for w in self.winfo_children():
            w.destroy()
        self._logo_images.clear()
        apply_ui_font(self)
        self._build_sidebar()
        self._build_main()
        self._refresh_preview()
        self._refresh_state()

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=210, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsw")
        sb.grid_propagate(False)
        ctk.CTkButton(sb, text=f"{ICONS('sun')} / {ICONS('moon')}  "
                               f"{tr('theme')}",
                      fg_color="gray40",
                      command=self.toggle_theme).pack(side="bottom",
                                                      fill="x", padx=12,
                                                      pady=(4, 14))
        self.lang_menu = ctk.CTkOptionMenu(
            sb, values=list(LANGUAGES.values()),
            fg_color="gray40", button_color="gray30",
            command=self.change_language)
        self.lang_menu.set(LANGUAGES[CURRENT_LANG])
        self.lang_menu.pack(side="bottom", fill="x", padx=12, pady=(0, 4))
        ctk.CTkLabel(sb, text=ICONS.label('lang', tr('language')),
                     text_color="gray",
                     font=ctk.CTkFont(size=11)).pack(side="bottom",
                                                     anchor="w", padx=16)

        nav = ctk.CTkScrollableFrame(sb, fg_color="transparent",
                                     corner_radius=0)
        nav.pack(side="top", fill="both", expand=True)

        logo = app_logo_ctk((30, 30))
        if logo is not None:
            self._logo_images.append(logo)
        ctk.CTkLabel(nav, text="  ArborML", image=logo, compound="left",
                     font=ctk.CTkFont(size=22, weight="bold")
                     ).pack(pady=(14, 2))
        ctk.CTkLabel(nav, text=f"v{APP_VERSION}", text_color="gray"
                     ).pack(pady=(0, 10))

        def btn(text, cmd, **kw):
            b = ctk.CTkButton(nav, text=text, command=cmd, anchor="w",
                              fg_color="transparent", text_color=("gray10",
                              "gray90"), hover_color=("gray80", "gray25"), **kw)
            b.pack(fill="x", padx=6, pady=3)
            return b

        ctk.CTkLabel(nav, text=tr("sec_data"), text_color="gray",
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10)
        btn(ICONS.label('load', tr('load_file')), self.load_file)
        btn(ICONS.label('save', tr('export_json')), self.save_json)
        btn(ICONS.label('import', tr('import_json')), self.load_json)

        ctk.CTkLabel(nav, text=tr("sec_explore"), text_color="gray",
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10,
                                                     pady=(12, 0))
        self.btn_hist = btn(ICONS.label('hist', tr('histograms')),
                            self.show_hist)
        self.btn_heat = btn(ICONS.label('heat', tr('heatmap')),
                            self.show_heatmap)

        ctk.CTkLabel(nav, text=tr("sec_model"), text_color="gray",
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10,
                                                     pady=(12, 0))
        self.btn_models = [
            btn(ICONS.label('linear' if name == 'Linear Regression'
                            else 'tree', name),
                lambda n=name: ModelDialog(self, n))
            for name in MODEL_SPECS]
        self.btn_pred = btn(ICONS.label('predict', tr('simulate')),
                            lambda: PredictDialog(self))
        self.btn_export = btn(ICONS.label('export', tr('export_model')),
                              self.export_model)
        btn(ICONS.label('import', tr('import_model')), self.import_model)
        self.btn_report = btn(ICONS.label('report', tr('pdf_report')),
                              self.save_report)


    def _build_main(self):
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=14, pady=14)
        main.grid_rowconfigure(1, weight=3)
        main.grid_rowconfigure(3, weight=2)
        main.grid_columnconfigure(0, weight=1)

        self.status = ctk.CTkLabel(main, text="", anchor="w", justify="left",
                                   font=ctk.CTkFont(size=13, weight="bold"))
        self.status.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.preview = ctk.CTkTextbox(main, font=mono_font(self, 12))
        self.preview.grid(row=1, column=0, sticky="nsew")
        self.preview.configure(state="disabled")

        ctk.CTkLabel(main, text=tr("rename_cols"), anchor="w"
                     ).grid(row=2, column=0, sticky="ew", pady=(10, 2))
        self.rename_frame = ctk.CTkScrollableFrame(main)
        self.rename_frame.grid(row=3, column=0, sticky="nsew")

    def change_language(self, display_name: str):
        code = next((c for c, n in LANGUAGES.items() if n == display_name),
                    DEFAULT_LANG)
        if code == CURRENT_LANG:
            return
        set_language(code)
        save_config()
        self._build_ui()

    def _refresh_state(self):
        data_ok = self.dm.loaded
        for b in [self.btn_hist, self.btn_heat, self.btn_report,
                  *self.btn_models]:
            b.configure(state="normal" if data_ok else "disabled")
        model_ok = self.model is not None
        self.btn_pred.configure(state="normal" if model_ok else "disabled")
        self.btn_export.configure(state="normal" if model_ok else "disabled")
        if data_ok:
            df = self.dm.df
            text = ICONS.label("ok", tr("status_loaded",
                                        file=self.dm.source_name,
                                        rows=df.shape[0],
                                        cols=df.shape[1]))
            if self.model:
                text += "   |   " + tr("last_model", name=self.model.name)
            if self.simulations:
                text += "   |   " + tr("sim_count", n=len(self.simulations))
            if self.dm.has_dropped:
                text += "\n" + ICONS.label("warn", self.dm.cleaning_note)
            self.status.configure(text=tk_safe(text))
        else:
            self.status.configure(text=tr("status_none"))

    def _refresh_preview(self):
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        if self.dm.loaded:
            df = self.dm.df
            shown = pd.concat([df.head(6), df.tail(6)]) if len(df) > 12 else df
            self.preview.insert("1.0", tk_safe(shown.to_string()))
            if len(df) > 12:
                self.preview.insert(
                    "end", "\n\n" + tr("rows_hidden", n=len(df) - 12))
        self.preview.configure(state="disabled")

        for w in self.rename_frame.winfo_children():
            w.destroy()
        self._rename_entries.clear()
        if self.dm.loaded:
            for col in self.dm.df.columns:
                row = ctk.CTkFrame(self.rename_frame, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text=col, width=200, anchor="w"
                             ).pack(side="left")
                e = ctk.CTkEntry(row, placeholder_text=tr("new_name"),
                                 width=200)
                e.pack(side="left", padx=6)
                e.bind("<Return>", lambda _ev: self.apply_rename())
                self._rename_entries[col] = e
            ctk.CTkButton(self.rename_frame, text=tr("apply_names"),
                          fg_color=ACCENT, hover_color=ACCENT_HOVER,
                          command=self.apply_rename).pack(pady=8)

    def record_simulation(self, values, preds) -> None:
        if self.model is None:
            return
        self.simulations.append(Simulation(
            model_name=self.model.name,
            x_cols=list(self.model.x_cols),
            y_cols=list(self.model.y_cols),
            x=[float(v) for v in values],
            y=[float(v) for v in np.atleast_1d(preds)],
        ))
        del self.simulations[:-self.SIM_HISTORY_MAX]
        self._refresh_state()

    def clear_simulations(self) -> None:
        self.simulations.clear()
        self._refresh_state()

    def toggle_theme(self):
        mode = ctk.get_appearance_mode()
        ctk.set_appearance_mode("Dark" if mode == "Light" else "Light")

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[
            (tr("ft_all"), "*.xlsx *.xls *.csv *.txt *.dat"),
            ("Excel", "*.xlsx *.xls"), ("CSV / DAT", "*.csv *.dat"),
            (tr("ft_text"), "*.txt")])
        if not path:
            return
        sep = None
        if DataManager.needs_separator(path):
            sep = simpledialog.askstring(tr("sep_title"), tr("sep_msg"),
                                         parent=self)
        try:
            with progress_dialog(self, tr("load_file"), 3) as prog:
                self.dm.load(path, sep=sep, progress=prog)
                self.model = None
                self.simulations.clear()
                prog.step(tr("prog_display"))
                self._refresh_preview()
                self._refresh_state()
                prog.set(1.0, tr("prog_done"))
        except Exception as exc:
            messagebox.showerror(tr("load_err"), str(exc))

    def apply_rename(self):
        mapping = {old: e.get().strip()
                   for old, e in self._rename_entries.items()
                   if e.get().strip()}
        if not mapping:
            return
        new_names = [mapping.get(c, c) for c in self.dm.df.columns]
        if len(set(new_names)) != len(new_names):
            messagebox.showerror(tr("error"), tr("dup_names"))
            return
        self.dm.rename(mapping)
        self.model = None
        self.simulations.clear()
        self._refresh_preview()
        self._refresh_state()

    def save_json(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with progress_dialog(self, tr("export_json"), 2) as prog:
                prog.step(tr("prog_write"))
                prog.run(self.dm.to_json, path)
                prog.set(1.0, tr("prog_done"))
        except Exception as exc:
            messagebox.showerror(tr("error"), str(exc))
            return
        messagebox.showinfo(tr("success"), tr("data_saved", path=path))

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with progress_dialog(self, tr("import_json"), 3) as prog:
                prog.step(tr("prog_read"))
                prog.run(self.dm.from_json, path)
                self.model = None
                self.simulations.clear()
                prog.step(tr("prog_display"))
                self._refresh_preview()
                self._refresh_state()
                prog.set(1.0, tr("prog_done"))
        except Exception as exc:
            messagebox.showerror(tr("error"), str(exc))

    def show_hist(self):
        with progress_dialog(self, tr("histograms"), 2) as prog:
            prog.step(tr("prog_figures"))
            fig = histograms_figure(self.dm.df)
            prog.set(1.0, tr("prog_done"))
        PlotWindow(self, tr("histograms"), [fig], [tr("tab_dist")])

    def show_heatmap(self):
        with progress_dialog(self, tr("heatmap"), 2) as prog:
            prog.step(tr("prog_figures"))
            fig = heatmap_figure(self.dm.df)
            prog.set(1.0, tr("prog_done"))
        PlotWindow(self, tr("pdf_corr"), [fig], [tr("tab_heat")])

    def set_model(self, model: TrainedModel):
        self.model = model
        self.simulations.clear()
        self._refresh_state()

    def export_model(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".arborml",
            filetypes=[(tr("ft_model"), "*.arborml")])
        if not path:
            return
        try:
            with progress_dialog(self, tr("export_model"), 2) as prog:
                prog.step(tr("prog_save_model"))
                prog.run(self.model.save, path)
                prog.set(1.0, tr("prog_done"))
        except Exception as exc:
            messagebox.showerror(tr("error"), str(exc))
            return
        messagebox.showinfo(tr("success"), tr("model_saved", path=path))

    def import_model(self):
        path = filedialog.askopenfilename(
            filetypes=[(tr("ft_model"), "*.arborml"),
                       (tr("ft_allfiles"), "*.*")])
        if not path:
            return
        try:
            with progress_dialog(self, tr("import_model"), 2) as prog:
                prog.step(tr("prog_load_model"))
                self.model = prog.run(TrainedModel.load, path)
                self.simulations.clear()
                prog.set(1.0, tr("prog_done"))
        except Exception as exc:
            messagebox.showerror(tr("error"), tr("bad_model", err=exc))
            return
        self._refresh_state()
        messagebox.showinfo(
            tr("model_loaded_title"),
            f"{self.model.name}\n"
            + tr("model_loaded_msg",
                 x=", ".join(self.model.x_cols),
                 y=", ".join(self.model.y_cols)))

    def save_report(self):
        path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                            filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        sims = self.simulations if self.model else []
        if sims and not messagebox.askyesno(
                tr("pdf_sims"), tr("pdf_sim_ask", n=len(sims)), parent=self):
            sims = []
        try:
            builder = ReportBuilder(self.dm, self.model, sims)
            with progress_dialog(self, tr("pdf_report"),
                                 builder.n_steps()) as prog:
                builder.build(path, progress=prog)
                prog.set(1.0, tr("prog_done"))
        except Exception as exc:
            messagebox.showerror(tr("error"), tr("report_err", err=exc))
            return
        messagebox.showinfo(tr("success"), tr("report_saved", path=path))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S")
    ArborMLApp().mainloop()
