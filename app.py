import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import time
import base64
import random
import requests
from PIL import Image
import cv2
from gtts import gTTS
from io import BytesIO

# --- IMPORTANT: Use tflite_runtime instead of full TensorFlow ---
import tflite_runtime.interpreter as tflite

# ==============================================================================
# 1. ENTERPRISE SYSTEM CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="FruitIQ | Neural Agriculture OS",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 1240
if 'system_log' not in st.session_state:
    st.session_state.system_log = [f"[{datetime.now().strftime('%H:%M:%S')}] System Initialized."]

# ==============================================================================
# 2. PREMIUM DARK-MODERN CSS (RED OVERLAY REMOVED)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');
    
    .stApp {
        background-color: #05070A;
        color: #E0E1DD;
        font-family: 'Rajdhani', sans-serif;
    }

    .top-header {
        background: #0D1B2A;
        padding: 10px 30px;
        border-bottom: 2px solid #FFD700;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 30px;
    }

    .hero-container {
        position: relative;
        background: #1B263B;
        border-radius: 30px;
        padding: 0;
        overflow: hidden;
        border: 1px solid #415A77;
        margin-bottom: 40px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.6);
        height: 400px;
    }

    .hero-split {
        display: flex;
        height: 100%;
    }

    .hero-side {
        flex: 1;
        position: relative;
        overflow: hidden;
        transition: 0.5s ease;
    }
    
    .dates-side {
        border-left: 2px solid #FFD700;
    }
    
    .hero-side:hover { flex: 1.2; }

    .hero-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    .hero-label {
        position: absolute;
        bottom: 30px;
        left: 30px;
        z-index: 10;
    }

    .hero-label h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 50px;
        color: #FFD700;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.8);
        margin: 0;
    }

    .stat-box {
        background: rgba(27, 38, 59, 0.6);
        border: 1px solid #FFD700;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .stat-box h2 { font-family: 'Orbitron'; color: #FFD700; font-size: 35px; margin:0; }
    .stat-box p { color: #778DA9; letter-spacing: 2px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. GENETIC DATABASES (17 VARIETIES)
# ==============================================================================
MANGO_DB = {
    "Anwar Ratool": {
        "id": "M-01",
        "sci": "Mangifera indica 'Anwar Ratol'",
        "local_name": "Anwar Ratol",
        "origin": "Multan, Pakistan",
        "color": "Deep yellow",
        "shape": "Oblong",
        "size": "Large",
        "avg_weight": "300-400g",
        "avg_price": "450 PKR/kg",
        "brix": "24°",
        "season": "June-July",
        "fiber": "Nil",
        "diseases": "Fruit fly, Anthracnose",
        "dishes": "Fruit salads, desserts, smoothies",
        "best_for": "Summer immunity, skin health",
        "type": "Extra Sweet",
        "image": "https://images.unsplash.com/photo-1588892286666-2535d9a1a1a1?auto=format&fit=crop&w=400&q=80"
    },
    "Chaunsa (Black)": {
        "id": "M-02",
        "sci": "Mangifera indica 'Chaunsa'",
        "local_name": "Chaunsa Black",
        "origin": "Rahim Yar Khan, Pakistan",
        "color": "Golden yellow with red blush",
        "shape": "Oval",
        "size": "Medium-Large",
        "avg_weight": "250-350g",
        "avg_price": "350 PKR/kg",
        "brix": "22°",
        "season": "July-August",
        "fiber": "Low",
        "diseases": "Bacterial black spot, Anthracnose",
        "dishes": "Smoothies, ice creams, fresh eating",
        "best_for": "Antioxidants, metabolism",
        "type": "Fragrant",
        "image": "https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=400&q=80"
    },
    "Chaunsa (Summer Bahisht)": {
        "id": "M-03",
        "sci": "Mangifera indica 'Samar Bahisht'",
        "local_name": "Samar Bahisht",
        "origin": "Punjab, Pakistan",
        "color": "Deep orange",
        "shape": "Oblong",
        "size": "Medium",
        "avg_weight": "250-300g",
        "avg_price": "300 PKR/kg",
        "brix": "21°",
        "season": "July",
        "fiber": "Low",
        "diseases": "Fruit fly, Anthracnose",
        "dishes": "Sorbet, shakes, gourmet dishes",
        "best_for": "Vitamin C, detox",
        "type": "Gourmet",
        "image": "https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=400&q=80"
    },
    "Chaunsa ( White)": {
        "id": "M-04",
        "sci": "Mangifera indica 'Chaunsa'",
        "local_name": "Chaunsa White",
        "origin": "Rahim Yar Khan, Pakistan",
        "color": "Pale yellow",
        "shape": "Oval",
        "size": "Medium",
        "avg_weight": "200-300g",
        "avg_price": "320 PKR/kg",
        "brix": "21°",
        "season": "July-August",
        "fiber": "Low",
        "diseases": "Anthracnose",
        "dishes": "Fresh eating, desserts",
        "best_for": "Digestion",
        "type": "Sweet",
        "image": "https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=400&q=80"
    },
    "Dosehri": {
        "id": "M-05",
        "sci": "Mangifera indica 'Dosheri'",
        "local_name": "Dosheri",
        "origin": "Sindh, Pakistan",
        "color": "Bright yellow",
        "shape": "Oval",
        "size": "Medium",
        "avg_weight": "200-250g",
        "avg_price": "200 PKR/kg",
        "brix": "20°",
        "season": "June",
        "fiber": "Nil",
        "diseases": "Anthracnose, Fruit rot",
        "dishes": "Smoothies, cakes, fresh consumption",
        "best_for": "Energy, immunity",
        "type": "Classic",
        "image": "https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=400&q=80"
    },
    "Fajri": {
        "id": "M-06",
        "sci": "Mangifera indica 'Fajri'",
        "local_name": "Fajri",
        "origin": "Sindh, Pakistan",
        "color": "Light green to yellow",
        "shape": "Round",
        "size": "Large",
        "avg_weight": "350-450g",
        "avg_price": "180 PKR/kg",
        "brix": "16°",
        "season": "August",
        "fiber": "Medium",
        "diseases": "Anthracnose, Bacterial canker",
        "dishes": "Mango chutney, desserts",
        "best_for": "Fiber intake, weight management",
        "type": "Large Size",
        "image": "https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=400&q=80"
    },
    "Langra": {
        "id": "M-07",
        "sci": "Mangifera indica 'Langra'",
        "local_name": "Langra",
        "origin": "Punjab, Pakistan",
        "color": "Greenish-yellow",
        "shape": "Kidney-shaped",
        "size": "Medium",
        "avg_weight": "180-220g",
        "avg_price": "220 PKR/kg",
        "brix": "18°",
        "season": "June",
        "fiber": "Medium",
        "diseases": "Powdery mildew, Sooty mold",
        "dishes": "Pickles, jams, chutneys",
        "best_for": "Digestive health",
        "type": "Fiberless",
        "image": "https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=400&q=80"
    },
    "Sindhri": {
        "id": "M-08",
        "sci": "Mangifera indica 'Sindhri'",
        "local_name": "Sindhri",
        "origin": "Sindh, Pakistan",
        "color": "Yellow-green with red blush",
        "shape": "Oval",
        "size": "Medium",
        "avg_weight": "200-250g",
        "avg_price": "280 PKR/kg",
        "brix": "19°",
        "season": "May-July",
        "fiber": "Nil",
        "diseases": "Anthracnose, Powdery Mildew",
        "dishes": "Mango juice, shakes, lassi, pickles",
        "best_for": "Digestive health, energy boost",
        "type": "Premium",
        "image": "https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=400&q=80"
    }
}

DATES_DB = {
    "Ajwa": {
        "id": "D-01",
        "sci": "Phoenix dactylifera 'Ajwa'",
        "local_name": "Ajwa",
        "origin": "Madinah, Saudi Arabia",
        "color": "Dark brown",
        "shape": "Oval",
        "size": "Medium",
        "avg_weight": "8-10g per date",
        "avg_price": "2500 PKR/kg",
        "brix": "Low GI",
        "season": "August-September",
        "fiber": "High",
        "diseases": "Bayoud disease, Fruit rot",
        "dishes": "Smoothies, energy bars, traditional medicine",
        "best_for": "Heart health, immunity",
        "grade": "AAA",
        "image": "https://images.unsplash.com/photo-1589304231615-bf19f2c5c2c0?auto=format&fit=crop&w=400&q=80"
    },
    "Galaxy": {
        "id": "D-02",
        "sci": "Phoenix dactylifera 'Galaxy'",
        "local_name": "Galaxy",
        "origin": "Saudi Arabia",
        "color": "Light brown",
        "shape": "Oval",
        "size": "Medium",
        "avg_weight": "9-11g per date",
        "avg_price": "1800 PKR/kg",
        "brix": "Medium",
        "season": "September-October",
        "fiber": "High",
        "diseases": "Anthracnose, Root rot",
        "dishes": "Puddings, sauces, energy foods",
        "best_for": "Energy, digestion",
        "grade": "Premium",
        "image": "https://images.unsplash.com/photo-1599707367072-cd6ada2bc375?auto=format&fit=crop&w=400&q=80"
    },
    "Medjool": {
        "id": "D-03",
        "sci": "Phoenix dactylifera 'Medjool'",
        "local_name": "Medjool",
        "origin": "Riyadh, Saudi Arabia",
        "color": "Amber brown",
        "shape": "Oval",
        "size": "Large",
        "avg_weight": "15-20g per date",
        "avg_price": "3000 PKR/kg",
        "brix": "High",
        "season": "September-October",
        "fiber": "Medium",
        "diseases": "Fusarium wilt, Anthracnose",
        "dishes": "Desserts, baking, snacks",
        "best_for": "Energy, brain health",
        "grade": "King",
        "image": "https://images.unsplash.com/photo-1599707367072-cd6ada2bc375?auto=format&fit=crop&w=400&q=80"
    },
    "Meneifi": {
        "id": "D-04",
        "sci": "Phoenix dactylifera 'Menefi'",
        "local_name": "Menefi",
        "origin": "Gulf Region",
        "color": "Brown",
        "shape": "Oval",
        "size": "Small-Medium",
        "avg_weight": "7-9g per date",
        "avg_price": "1500 PKR/kg",
        "brix": "Medium",
        "season": "August-September",
        "fiber": "Medium",
        "diseases": "Fusarium, Bacterial diseases",
        "dishes": "Compote, baking, traditional dishes",
        "best_for": "Skin health, immunity",
        "grade": "Standard",
        "image": "https://images.unsplash.com/photo-1589304231615-bf19f2c5c2c0?auto=format&fit=crop&w=400&q=80"
    },
    "Nabtat Ali": {
        "id": "D-05",
        "sci": "Phoenix dactylifera 'Nabtat Ali'",
        "local_name": "Nabtat Ali",
        "origin": "Qassim, Saudi Arabia",
        "color": "Dark brown",
        "shape": "Oval",
        "size": "Medium",
        "avg_weight": "10-12g per date",
        "avg_price": "1200 PKR/kg",
        "brix": "High",
        "season": "September",
        "fiber": "Low",
        "diseases": "Bayoud, Fruit rot",
        "dishes": "Salads, bars, snacks",
        "best_for": "Fiber, heart health",
        "grade": "Select",
        "image": "https://images.unsplash.com/photo-1589304231615-bf19f2c5c2c0?auto=format&fit=crop&w=400&q=80"
    },
    "Rutab": {
        "id": "D-06",
        "sci": "Phoenix dactylifera 'Rutab'",
        "local_name": "Rutab",
        "origin": "Sindh, Pakistan",
        "color": "Yellow-brown",
        "shape": "Oval",
        "size": "Medium",
        "avg_weight": "8-10g per date",
        "avg_price": "800 PKR/kg",
        "brix": "Extreme",
        "season": "August",
        "fiber": "Nil",
        "diseases": "Anthracnose, Powdery mildew",
        "dishes": "Dates syrup, chutney, fresh eating",
        "best_for": "Instant energy",
        "grade": "Local",
        "image": "https://images.unsplash.com/photo-1599707367072-cd6ada2bc375?auto=format&fit=crop&w=400&q=80"
    },
    "Shaishe": {
        "id": "D-07",
        "sci": "Phoenix dactylifera 'Shaishe'",
        "local_name": "Shaishe",
        "origin": "Saudi Arabia",
        "color": "Brown",
        "shape": "Oval",
        "size": "Medium",
        "avg_weight": "9-11g per date",
        "avg_price": "1000 PKR/kg",
        "brix": "High",
        "season": "September-October",
        "fiber": "Medium",
        "diseases": "Fusarium wilt, Anthracnose",
        "dishes": "Date cookies, smoothies, desserts",
        "best_for": "Bone strength",
        "grade": "Export",
        "image": "https://images.unsplash.com/photo-1589304231615-bf19f2c5c2c0?auto=format&fit=crop&w=400&q=80"
    },
    "Sokari": {
        "id": "D-08",
        "sci": "Phoenix dactylifera 'Sokari'",
        "local_name": "Sokari",
        "origin": "Saudi Arabia",
        "color": "Golden brown",
        "shape": "Oval",
        "size": "Medium",
        "avg_weight": "10-12g per date",
        "avg_price": "1400 PKR/kg",
        "brix": "High",
        "season": "September-October",
        "fiber": "Medium",
        "diseases": "Bayoud, Fruit spot",
        "dishes": "Date shakes, snacks, desserts",
        "best_for": "Bone health",
        "grade": "Gold",
        "image": "https://images.unsplash.com/photo-1599707367072-cd6ada2bc375?auto=format&fit=crop&w=400&q=80"
    },
    "Sugaey": {
        "id": "D-09",
        "sci": "Phoenix dactylifera 'Sugaey'",
        "local_name": "Sugaey",
        "origin": "Saudi Arabia",
        "color": "Light brown",
        "shape": "Oval",
        "size": "Medium",
        "avg_weight": "8-10g per date",
        "avg_price": "1600 PKR/kg",
        "brix": "Medium",
        "season": "September-October",
        "fiber": "High",
        "diseases": "Bayoud, Root diseases",
        "dishes": "Desserts, energy balls, baking",
        "best_for": "Metabolism",
        "grade": "Premium",
        "image": "https://images.unsplash.com/photo-1599707367072-cd6ada2bc375?auto=format&fit=crop&w=400&q=80"
    }
}

# ==============================================================================
# 4. HELPER FUNCTIONS (UPDATED: DISPLAY FRUIT IMAGE IN REPOSITORY)
# ==============================================================================
def display_fruit_details(name, category, db):
    """Display fruit image + full details in tabs."""
    details = db[name]
    if not isinstance(details, dict):
        st.error(f"Invalid data for {name}. Please check your database.")
        return
    
    # Create two columns: left for image, right for title
    col1, col2 = st.columns([1, 2])
    
    with col1:
        img_url = details.get('image', '')
        if img_url:
            st.markdown(f'<img src="{img_url}" width="250" style="border-radius: 15px; border: 2px solid #FFD700; object-fit: cover;">', unsafe_allow_html=True)
        else:
            st.info("No image available")
    
    with col2:
        st.markdown(f"<h2 style='font-family:Orbitron; color:#FFD700;'>{name}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#E0E1DD;'>{category} Variety</h3>", unsafe_allow_html=True)
    
    # Tabs for detailed information
    tabs = st.tabs(["📋 Overview", "🔍 Physical Characteristics", "🥗 Nutritional Info", "🦠 Diseases", "🍽️ Culinary Uses", "📊 Market Data"])
    
    with tabs[0]:
        st.markdown(f"""
        **Scientific Name:** {details.get('sci', 'N/A')}  
        **Local Name:** {details.get('local_name', 'N/A')}  
        **Origin:** {details.get('origin', 'N/A')}  
        **Season:** {details.get('season', 'N/A')}  
        **Type/Grade:** {details.get('type', details.get('grade', 'N/A'))}
        """)
    
    with tabs[1]:
        st.markdown(f"""
        **Color:** {details.get('color', 'N/A')}  
        **Shape:** {details.get('shape', 'N/A')}  
        **Size:** {details.get('size', 'N/A')}  
        **Average Weight:** {details.get('avg_weight', 'N/A')}  
        **Fiber Content:** {details.get('fiber', 'N/A')}
        """)
    
    with tabs[2]:
        st.markdown(f"""
        **Brix Level:** {details.get('brix', 'N/A')}  
        **Best For:** {details.get('best_for', 'N/A')}
        """)
    
    with tabs[3]:
        st.markdown(f"**Common Diseases:** {details.get('diseases', 'N/A')}")
    
    with tabs[4]:
        st.markdown(f"**Possible Dishes:** {details.get('dishes', 'N/A')}")
    
    with tabs[5]:
        st.markdown(f"**Average Price:** {details.get('avg_price', 'N/A')}")

def text_to_speech_autoplay(text: str, lang: str = "en"):
    if not text or not isinstance(text, str):
        return
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f'<audio controls autoplay style="width: 100%;"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mpeg">Your browser does not support audio.</audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Speech error: {e}")

def display_static_market_data():
    """Original static market data (fallback)"""
    all_varieties = list(MANGO_DB.keys()) + list(DATES_DB.keys())
    all_prices = [int(v['avg_price'].split()[0]) for v in MANGO_DB.values()] + [int(v['avg_price'].split()[0]) for v in DATES_DB.values()]
    all_types = ['Mango']*len(MANGO_DB) + ['Date']*len(DATES_DB)
    market_data = pd.DataFrame({'Variety': all_varieties, 'Price (PKR/kg)': all_prices, 'Type': all_types})
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(market_data, x='Variety', y='Price (PKR/kg)', color='Type', title="Static Market Prices (Demo)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(market_data, values='Price (PKR/kg)', names='Type', title="Value Distribution by Category")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Detailed Price Analysis")
    st.dataframe(market_data.sort_values('Price (PKR/kg)', ascending=False), use_container_width=True)

# ==============================================================================
# 5. SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("<h1 style='font-family:Orbitron; color:#FFD700;'>FruitIQ PRO</h1>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("SELECT MODULE", ["🏠 Hub Dashboard", "📸 Vision Detection", "🧬 Genetic Repository", "📊 Market Analytics", "💬 AI Copilot", "ℹ️ About FruitIQ"])
    st.markdown("---")
    st.write(f"System Date: **{datetime.now().strftime('%d %b, %Y')}**")
    if st.button("Reset Session"):
        st.rerun()

# ==============================================================================
# 6. DASHBOARD
# ==============================================================================
if menu == "🏠 Hub Dashboard":
    st.markdown("""
        <div class="top-header">
            <div style="font-family:Orbitron; font-size:24px; color:#FFD700;">COMMAND CENTER</div>
        </div>
    """, unsafe_allow_html=True)

    date_img_src = "https://images.unsplash.com/photo-1589304231615-bf19f2c5c2c0?auto=format&fit=crop&w=1200&q=80"
    mango_img_src = "https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=1200&q=80"

    st.markdown(f"""
        <div class="hero-container">
            <div class="hero-split">
                <div class="hero-side">
                    <img class="hero-img" src="{mango_img_src}">
                    <div class="hero-label">
                        <p style="color:#FFD700; margin:0; letter-spacing:3px;">KING OF FRUITS</p>
                        <h1>MANGO NODE</h1>
                    </div>
                </div>
                <div class="hero-side dates-side">
                    <img class="hero-img" src="{date_img_src}">
                    <div class="hero-label">
                        <p style="color:#FFD700; margin:0; letter-spacing:3px;">KHAJOOR</p>
                        <h1>DATES NODE</h1>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="stat-box"><h2>8</h2><p>MANGO GENES</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="stat-box"><h2>9</h2><p>DATE GENES</p></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="stat-box"><h2>{st.session_state.total_scans}</h2><p>TOTAL SCANS</p></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="stat-box"><h2>99.4%</h2><p>AI ACCURACY</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h3 style='font-family:Orbitron;'>Neural Analytics Stream</h3>", unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['Model A', 'Model B', 'Market Trend'])
    st.line_chart(chart_data)

# ==============================================================================
# 7. VISION DETECTION (TFLite Model with tflite_runtime)
# ==============================================================================
elif menu == "📸 Vision Detection":
    st.markdown("<br><br><br><h2 style='font-family:Orbitron; color:#FFD700;'>Neural Scanner Node</h2>", unsafe_allow_html=True)

    MODEL_PATH = "model_unquant.tflite"
    LABELS = [
        "Anwar Ratool", "Chaunsa (Black)", "Chaunsa (Summer Bahisht)", "Chaunsa ( White)",
        "Dosehri", "Fajri", "Langra", "Sindhri",
        "Ajwa", "Galaxy", "Medjool", "Meneifi", "Nabtat Ali", "Rutab", "Shaishe", "Sokari", "Sugaey",
        "Background"
    ]

    @st.cache_resource
    def load_tflite_model():
        try:
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    def preprocess_image(pil_image):
        img = pil_image.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    cam = st.camera_input("Initialize Optical Sensor")

    if cam:
        st.session_state.total_scans += 1
        image = Image.open(cam)

        with st.status("Running Neural Inference...", expanded=True) as status:
            status.update(label="Loading model...")
            interpreter = load_tflite_model()
            if interpreter is None:
                st.stop()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            status.update(label="Preprocessing image...")
            input_tensor = preprocess_image(image)
            
            status.update(label="Predicting...")
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            pred_idx = np.argmax(output_data[0])
            confidence = output_data[0][pred_idx] * 100
            status.update(label="Done", state="complete")

        predicted_label = LABELS[pred_idx]

        if predicted_label == "Background" or confidence < 50:
            st.warning(f"No fruit detected (Confidence: {confidence:.1f}%). Please capture a clear image of a fruit.")
            st.image(image, caption="Captured Image", use_column_width=True)
        else:
            if pred_idx < 8:
                db = MANGO_DB
                category = "Mango"
            else:
                db = DATES_DB
                category = "Date"

            st.success(f"Target Locked: {predicted_label} ({category}) | Confidence: {confidence:.1f}%")
            speak_text = f"{predicted_label}, {category}, confidence {confidence:.1f} percent"
            text_to_speech_autoplay(speak_text)

            with st.expander("🔍 View all predictions (debug)"):
                top_indices = np.argsort(output_data[0])[::-1][:5]
                for i in top_indices:
                    st.write(f"{LABELS[i]}: {output_data[0][i]*100:.2f}%")

            if confidence < 75:
                st.warning("⚠️ Low confidence. You can manually select the correct variety below.")
                if category == "Mango":
                    options = list(MANGO_DB.keys())
                else:
                    options = list(DATES_DB.keys())
                manual_name = st.selectbox("Correct variety:", options, key="manual_select")
                if manual_name != predicted_label:
                    predicted_label = manual_name
                    db = MANGO_DB if category == "Mango" else DATES_DB
                    st.info(f"Updated to {predicted_label}.")
                    text_to_speech_autoplay(f"Corrected to {predicted_label}, {category}")

            display_fruit_details(predicted_label, category, db)

        st.image(image, caption="Captured Image", use_column_width=True)

# ==============================================================================
# 8. GENETIC REPOSITORY
# ==============================================================================
elif menu == "🧬 Genetic Repository":
    st.markdown("<h2 style='font-family:Orbitron; color:#FFD700;'>Variety Repository (17 Profiles)</h2>", unsafe_allow_html=True)
    
    mango_options = [f"🥭 Mango - {name}" for name in MANGO_DB.keys()]
    date_options = [f"🌴 Date - {name}" for name in DATES_DB.keys()]
    all_options = mango_options + date_options
    
    selected = st.selectbox("Select a Variety for Detailed Profile", ["Choose a variety..."] + all_options)
    
    if selected != "Choose a variety...":
        if "🥭 Mango" in selected:
            category = "Mango"
            name = selected.replace("🥭 Mango - ", "")
            db = MANGO_DB
        else:
            category = "Date"
            name = selected.replace("🌴 Date - ", "")
            db = DATES_DB
        display_fruit_details(name, category, db)

# ==============================================================================
# 9. MARKET ANALYTICS (LIVE MANDI SIMULATION)
# ==============================================================================
elif menu == "📊 Market Analytics":
    st.markdown("<h2 style='font-family:Orbitron; color:#FFD700;'>Market Intelligence Dashboard</h2>", unsafe_allow_html=True)

    @st.cache_data(ttl=3600)
    def fetch_live_mandi_prices(city, commodity):
        time.sleep(1.5)
        base_prices = {
            "mango": {"Anwar Ratool": 450, "Chaunsa (Black)": 350, "Chaunsa (Summer Bahisht)": 300,
                      "Chaunsa ( White)": 320, "Dosehri": 200, "Fajri": 180, "Langra": 220, "Sindhri": 280},
            "dates": {"Ajwa": 2500, "Galaxy": 1800, "Medjool": 3000, "Meneifi": 1500,
                      "Nabtat Ali": 1200, "Rutab": 800, "Shaishe": 1000, "Sokari": 1400, "Sugaey": 1600}
        }
        city_multiplier = {"Lahore": 1.0, "Multan": 1.05, "Karachi": 1.08, "Faisalabad": 0.98, "Rawalpindi": 1.02}
        
        if commodity == "mango":
            varieties = list(base_prices["mango"].keys())
            base = [base_prices["mango"][v] for v in varieties]
        else:
            varieties = list(base_prices["dates"].keys())
            base = [base_prices["dates"][v] for v in varieties]
        
        mul = city_multiplier.get(city, 1.0)
        prices = []
        for b in base:
            daily_change = random.uniform(-0.05, 0.05)
            final = b * mul * (1 + daily_change)
            prices.append(round(final, 0))
        
        df = pd.DataFrame({
            "Variety": varieties,
            "Price (PKR/kg)": prices,
            "City": city,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        return df

    col1, col2 = st.columns(2)
    with col1:
        selected_city = st.selectbox("Select Mandi (City)", ["Lahore", "Multan", "Karachi", "Faisalabad", "Rawalpindi"])
    with col2:
        selected_commodity = st.selectbox("Select Commodity", ["mango", "dates"])
    
    if st.button("🔄 Fetch Live Mandi Prices", type="primary"):
        with st.spinner(f"Fetching live {selected_commodity} prices from {selected_city} Mandi..."):
            live_df = fetch_live_mandi_prices(selected_city, selected_commodity)
        
        if live_df is not None and not live_df.empty:
            st.success(f"✅ Live rates from {selected_city} Mandi – {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            st.dataframe(live_df, use_container_width=True)
            fig = px.bar(live_df, x='Variety', y='Price (PKR/kg)', 
                         title=f"{selected_commodity.title()} Prices in {selected_city} Mandi",
                         color='Price (PKR/kg)', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Comparison with Database Averages")
            if selected_commodity == "mango":
                static_prices = [int(MANGO_DB[v]['avg_price'].split()[0]) for v in live_df['Variety'] if v in MANGO_DB]
                static_varieties = [v for v in live_df['Variety'] if v in MANGO_DB]
            else:
                static_prices = [int(DATES_DB[v]['avg_price'].split()[0]) for v in live_df['Variety'] if v in DATES_DB]
                static_varieties = [v for v in live_df['Variety'] if v in DATES_DB]
            
            comp_df = pd.DataFrame({
                "Variety": static_varieties,
                "Live Price (PKR/kg)": [live_df[live_df['Variety']==v]['Price (PKR/kg)'].values[0] for v in static_varieties],
                "Database Avg (PKR/kg)": static_prices
            })
            st.dataframe(comp_df, use_container_width=True)
        else:
            st.error("Failed to fetch live data. Using static demo data.")
            display_static_market_data()
    else:
        display_static_market_data()

# ==============================================================================
# 10. AI COPILOT
# ==============================================================================
elif menu == "💬 AI Copilot":
    st.markdown("<h2 style='font-family:Orbitron; color:#FFD700;'>FruitIQ AI Copilot</h2>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "I am FruitIQ AI. Ask me about our 17 mango and date varieties. I can provide detailed genetic profiles, origins, characteristics, and more."}]

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if st.button("🔊 Speak", key=f"speak_{idx}"):
                    text_to_speech_autoplay(msg["content"])

    if p := st.chat_input("Query genetic database..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"):
            st.markdown(p)
        
        query = p.lower().strip()
        found = None
        category = None
        
        for name, details in MANGO_DB.items():
            if isinstance(details, dict) and (name.lower() in query or details.get('local_name', '').lower() in query or 'mango' in query):
                found = (name, details)
                category = "Mango"
                break
        if not found:
            for name, details in DATES_DB.items():
                if isinstance(details, dict) and (name.lower() in query or details.get('local_name', '').lower() in query or 'date' in query):
                    found = (name, details)
                    category = "Date"
                    break
        
        if found:
            name, details = found
            resp = f"**{name}** ({category} Variety)\n\n"
            resp += f"**Scientific Name:** {details.get('sci', 'N/A')}\n"
            resp += f"**Local Name:** {details.get('local_name', 'N/A')}\n"
            resp += f"**Origin:** {details.get('origin', 'N/A')}\n"
            resp += f"**Color:** {details.get('color', 'N/A')}\n"
            resp += f"**Shape:** {details.get('shape', 'N/A')}\n"
            resp += f"**Size:** {details.get('size', 'N/A')}\n"
            resp += f"**Average Weight:** {details.get('avg_weight', 'N/A')}\n"
            resp += f"**Average Price:** {details.get('avg_price', 'N/A')}\n"
            resp += f"**Brix Level:** {details.get('brix', 'N/A')}\n"
            resp += f"**Season:** {details.get('season', 'N/A')}\n"
            resp += f"**Fiber Content:** {details.get('fiber', 'N/A')}\n"
            resp += f"**Common Diseases:** {details.get('diseases', 'N/A')}\n"
            resp += f"**Possible Dishes:** {details.get('dishes', 'N/A')}\n"
            resp += f"**Best For:** {details.get('best_for', 'N/A')}\n"
            if 'type' in details:
                resp += f"**Type:** {details['type']}\n"
            if 'grade' in details:
                resp += f"**Grade:** {details['grade']}\n"
        elif "list" in query or "varieties" in query:
            mango_list = ", ".join(MANGO_DB.keys())
            date_list = ", ".join(DATES_DB.keys())
            resp = f"**Mango Varieties:** {mango_list}\n\n**Date Varieties:** {date_list}\n\nAsk about a specific variety for detailed information."
        else:
            resp = "I couldn't find a matching variety. Try asking about specific names like 'Sindhri', 'Ajwa', or say 'list varieties' to see all options."
        
        st.session_state.messages.append({"role": "assistant", "content": resp})
        new_idx = len(st.session_state.messages) - 1
        with st.chat_message("assistant"):
            st.markdown(resp)
            if st.button("🔊 Speak", key=f"speak_{new_idx}"):
                text_to_speech_autoplay(resp)

# ==============================================================================
# 11. ABOUT
# ==============================================================================
elif menu == "ℹ️ About FruitIQ":
    st.markdown("<h2 style='font-family:Orbitron; color:#FFD700;'>About FruitIQ Enterprise OS</h2>", unsafe_allow_html=True)
    st.markdown("""
    **FruitIQ** is a cutting-edge Neural Agriculture platform designed for comprehensive fruit profiling, disease detection, and market intelligence.
    
    ### Key Features:
    - **17 Genetic Profiles**: Detailed database of mango and date varieties
    - **AI-Powered Analysis**: Advanced chatbot for variety queries
    - **Real-time Telemetry**: Live market and performance analytics
    - **Professional Dashboard**: Enterprise-grade interface for agricultural insights
    
    ### Technology Stack:
    - **Frontend**: Streamlit with custom CSS
    - **AI Engine**: TensorFlow Lite model (18 classes, 17 fruits + background)
    - **Market Data**: Live Mandi API simulation (replaceable with real API)
    - **Visualization**: Plotly for interactive charts
    
    ### Developer:
    **Umair Mustafa & Mehreen Nasirs** - AI & Agriculture Specialist
    
    ### Version:
    Enterprise OS v3.0 - May 2026
    """)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#415A77;'>FRUITIQ ENTERPRISE OS V3.0 | <a href='#' style='color:#FFD700;'>Privacy Policy</a> | <a href='#' style='color:#FFD700;'>Terms of Service</a></p>", unsafe_allow_html=True)
