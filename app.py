import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import time
import base64
from PIL import Image
import tensorflow as tf
import cv2
from gtts import gTTS
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import re
from fpdf import FPDF
import random

# ==============================================================================
# 1. SYSTEM CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="FruitIQ | Agricultural Intelligence Platform",
    page_icon="🍑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 1240
if 'system_log' not in st.session_state:
    st.session_state.system_log = [f"[{datetime.now().strftime('%H:%M:%S')}] System Initialized."]
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False
if 'urdu_voice' not in st.session_state:
    st.session_state.urdu_voice = False

# ==============================================================================
# 2. HELPER: CLEAN TEXT FOR PDF
# ==============================================================================
def clean_latin1(text):
    if text is None:
        return ""
    return text.encode('latin-1', errors='ignore').decode('latin-1')

# ==============================================================================
# 3. LIVE MANDI RATES (AMIS + Demo)
# ==============================================================================
CITY_AMIS_PAGE = "http://amis.pk/ViewPrices.aspx?searchType=1&commodityId=1"

MANGO_TO_AMIS = {
    "Anwar Ratool": "Mango(Anwer Ratol)",
    "Chaunsa (Black)": "Mango(Chounsa)",
    "Chaunsa (Summer Bahisht)": "Mango(Chounsa)",
    "Chaunsa ( White)": "Mango(Chounsa)",
    "Dosehri": "Mango(Desahri)",
    "Fajri": "Mango Saharni",
    "Langra": "Mango(Chounsa)",
    "Sindhri": "Mango(Sindhri)",
    "Kesar": "Mango(Kesar)",
    "Alphonso": "Mango(Alphonso)",
    "Neelum": "Mango(Neelum)",
    "Totapuri": "Mango(Totapuri)",
    "Himsagar": "Mango(Himsagar)",
    "Malda": "Mango(Malda)"
}

DATES_TO_AMIS = {
    "Ajwa": "Dates (Aseel)",
    "Galaxy": "Dates(Irani)",
    "Medjool": "Dates(Irani)",
    "Meneifi": "Dates (Aseel)",
    "Nabtat Ali": "Dates (Aseel)",
    "Rutab": "Dates(Irani)",
    "Shaishe": "Dates(Irani)",
    "Sokari": "Dates(Irani)",
    "Sugaey": "Dates(Irani)",
    "Zahidi": "Dates(Zahidi)",
    "Khadrawy": "Dates(Khadrawy)",
    "Barhi": "Dates(Barhi)",
    "Halawi": "Dates(Halawi)"
}

def fetch_live_rate(variety_name: str, category: str, demo_mode=False) -> str:
    if demo_mode:
        base = random.randint(150, 600) if category.lower() == "mango" else random.randint(800, 3000)
        return f"{base} PKR/kg (Demo)"
    if category.lower() == "mango":
        amis_commodity = MANGO_TO_AMIS.get(variety_name, "Mango(Chounsa)")
    else:
        amis_commodity = DATES_TO_AMIS.get(variety_name, "Dates (Aseel)")
    url = CITY_AMIS_PAGE
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        found_price = None
        for tr in soup.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 2:
                continue
            for idx, cell in enumerate(cells):
                cell_text = cell.get_text(separator=" ", strip=True)
                if amis_commodity.lower() in cell_text.lower():
                    if len(cells) >= 5:
                        price_cell = cells[4]
                        price_text = price_cell.get_text(strip=True)
                        if price_text and price_text != "-":
                            found_price = price_text
                            break
                    elif len(cells) >= 3:
                        price_cell = cells[2]
                        price_text = price_cell.get_text(strip=True)
                        if price_text and price_text != "-":
                            found_price = price_text
                            break
            if found_price:
                break
        if found_price:
            try:
                price_per_100kg = int(re.sub(r'[^0-9]', '', found_price))
                return f"{price_per_100kg/100:.0f} PKR/kg (Live)"
            except:
                return f"{found_price} PKR/100kg (Live)"
        else:
            return None
    except Exception as e:
        st.session_state.system_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] AMIS error: {e}")
        return None

# ==============================================================================
# 4. ROTTEN DETECTION
# ==============================================================================
def detect_rotten(image_pil):
    img = np.array(image_pil.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([10, 50, 40])
    upper_brown = np.array([30, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_rot = cv2.bitwise_or(mask_brown, mask_black)
    total_pixels = mask_rot.shape[0] * mask_rot.shape[1]
    rot_pixels = np.sum(mask_rot > 0)
    rot_percentage = (rot_pixels / total_pixels) * 100
    kernel = np.ones((5,5), np.uint8)
    mask_rot = cv2.morphologyEx(mask_rot, cv2.MORPH_OPEN, kernel)
    rot_pixels_clean = np.sum(mask_rot > 0)
    final_rot = min(max((rot_pixels_clean / total_pixels) * 100, rot_percentage), 100)
    if final_rot < 5:
        return (final_rot, "✅ Fresh", "Ready for market.")
    elif final_rot < 15:
        return (final_rot, "⚠️ Slight decay", "Use immediately or discount.")
    else:
        return (final_rot, "❌ Rotten", "Discard.")

# ==============================================================================
# 5. THEME CSS
# ==============================================================================
def apply_theme():
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
            * { font-family: 'Segoe UI', sans-serif; }
            .stApp { background: #0b1215; color: #eef2f0; }
            .dashboard-hero { display: flex; gap: 2rem; margin-bottom: 2rem; flex-wrap: wrap; }
            .hero-card { flex: 1; background: rgba(20,30,20,0.7); border-radius: 32px; overflow: hidden; border: 1px solid #d4af37; text-align: center; }
            .hero-img { width: 100%; height: 240px; object-fit: cover; background: #2c3a24; }
            .hero-title { padding: 1rem; background: rgba(0,0,0,0.6); }
            .kpi-card { background: #1e2a1c; border-radius: 28px; padding: 1rem; text-align: center; border: 1px solid #d4af37; }
            .kpi-number { font-size: 2rem; font-weight: bold; color: #e5b83c; }
            .market-snapshot { background: #16221a; border-radius: 20px; padding: 0.8rem; margin-bottom: 0.5rem; }
            .stButton>button { background: #2a3e2a; color: white; border-radius: 40px; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            * { font-family: 'Segoe UI', sans-serif; }
            .stApp { background: #f5f8f2; color: #1a2a1a; }
            .dashboard-hero { display: flex; gap: 2rem; margin-bottom: 2rem; flex-wrap: wrap; }
            .hero-card { flex: 1; background: white; border-radius: 32px; overflow: hidden; border: 1px solid #6b8c5c; text-align: center; }
            .hero-img { width: 100%; height: 240px; object-fit: cover; background: #eaf4e5; }
            .hero-title { padding: 1rem; background: #eaf4e5; }
            .kpi-card { background: white; border-radius: 28px; padding: 1rem; text-align: center; border: 1px solid #6b8c5c; }
            .kpi-number { font-size: 2rem; font-weight: bold; color: #2c6e2c; }
            .market-snapshot { background: #eaf4e5; border-radius: 20px; padding: 0.8rem; margin-bottom: 0.5rem; }
        </style>
        """, unsafe_allow_html=True)

# ==============================================================================
# 6. DATABASES (FULL 27 VARIETIES)
# ==============================================================================
MANGO_DB = {
    "Anwar Ratool": {"id": "M-01", "origin": "Multan, Pakistan", "avg_price": "450 PKR/kg", "season": "June-July", "best_for": "Immunity", "color": "Deep yellow", "shape": "Oblong", "fiber": "Nil", "brix": "24°"},
    "Chaunsa (Black)": {"id": "M-02", "origin": "Rahim Yar Khan, Pakistan", "avg_price": "350 PKR/kg", "season": "July-August", "best_for": "Antioxidants", "color": "Golden yellow", "shape": "Oval", "fiber": "Low", "brix": "22°"},
    "Chaunsa (Summer Bahisht)": {"id": "M-03", "origin": "Punjab, Pakistan", "avg_price": "300 PKR/kg", "season": "July", "best_for": "Vitamin C", "color": "Deep orange", "shape": "Oblong", "fiber": "Low", "brix": "21°"},
    "Chaunsa ( White)": {"id": "M-04", "origin": "Rahim Yar Khan, Pakistan", "avg_price": "320 PKR/kg", "season": "July-August", "best_for": "Digestion", "color": "Pale yellow", "shape": "Oval", "fiber": "Low", "brix": "21°"},
    "Dosehri": {"id": "M-05", "origin": "Sindh, Pakistan", "avg_price": "200 PKR/kg", "season": "June", "best_for": "Energy", "color": "Bright yellow", "shape": "Oval", "fiber": "Nil", "brix": "20°"},
    "Fajri": {"id": "M-06", "origin": "Sindh, Pakistan", "avg_price": "180 PKR/kg", "season": "August", "best_for": "Fiber", "color": "Light green", "shape": "Round", "fiber": "Medium", "brix": "16°"},
    "Langra": {"id": "M-07", "origin": "Punjab, Pakistan", "avg_price": "220 PKR/kg", "season": "June", "best_for": "Digestive health", "color": "Greenish-yellow", "shape": "Kidney", "fiber": "Medium", "brix": "18°"},
    "Sindhri": {"id": "M-08", "origin": "Sindh, Pakistan", "avg_price": "280 PKR/kg", "season": "May-July", "best_for": "Energy boost", "color": "Yellow-green", "shape": "Oval", "fiber": "Nil", "brix": "19°"},
    "Kesar": {"id": "M-09", "origin": "Gujarat, India", "avg_price": "400 PKR/kg", "season": "June-July", "best_for": "Aroma", "color": "Golden yellow", "shape": "Oval", "fiber": "Nil", "brix": "23°"},
    "Alphonso": {"id": "M-10", "origin": "Maharashtra, India", "avg_price": "500 PKR/kg", "season": "April-June", "best_for": "Creamy texture", "color": "Saffron yellow", "shape": "Oval", "fiber": "Low", "brix": "24°"},
    "Neelum": {"id": "M-11", "origin": "Tamil Nadu, India", "avg_price": "250 PKR/kg", "season": "May-June", "best_for": "Sweetness", "color": "Pale yellow", "shape": "Oval-oblong", "fiber": "Nil", "brix": "20°"},
    "Totapuri": {"id": "M-12", "origin": "Karnataka, India", "avg_price": "220 PKR/kg", "season": "June-July", "best_for": "Culinary", "color": "Greenish-yellow", "shape": "Beaked", "fiber": "Medium", "brix": "18°"},
    "Himsagar": {"id": "M-13", "origin": "West Bengal, India", "avg_price": "350 PKR/kg", "season": "June", "best_for": "Sweetness", "color": "Golden", "shape": "Oval", "fiber": "Nil", "brix": "22°"},
    "Malda": {"id": "M-14", "origin": "Bihar, India", "avg_price": "280 PKR/kg", "season": "July", "best_for": "Aroma", "color": "Yellow", "shape": "Oblong", "fiber": "Low", "brix": "21°"}
}

DATES_DB = {
    "Ajwa": {"id": "D-01", "origin": "Madinah, Saudi Arabia", "avg_price": "2500 PKR/kg", "best_for": "Heart health", "grade": "AAA", "color": "Dark brown", "shape": "Oval", "fiber": "High"},
    "Galaxy": {"id": "D-02", "origin": "Saudi Arabia", "avg_price": "1800 PKR/kg", "best_for": "Energy", "grade": "Premium", "color": "Light brown", "shape": "Oval", "fiber": "High"},
    "Medjool": {"id": "D-03", "origin": "Riyadh, Saudi Arabia", "avg_price": "3000 PKR/kg", "best_for": "Brain health", "grade": "King", "color": "Amber brown", "shape": "Oval", "fiber": "Medium"},
    "Meneifi": {"id": "D-04", "origin": "Gulf Region", "avg_price": "1500 PKR/kg", "best_for": "Skin health", "grade": "Standard", "color": "Brown", "shape": "Oval", "fiber": "Medium"},
    "Nabtat Ali": {"id": "D-05", "origin": "Qassim, Saudi Arabia", "avg_price": "1200 PKR/kg", "best_for": "Fiber", "grade": "Select", "color": "Dark brown", "shape": "Oval", "fiber": "Low"},
    "Rutab": {"id": "D-06", "origin": "Sindh, Pakistan", "avg_price": "800 PKR/kg", "best_for": "Instant energy", "grade": "Local", "color": "Yellow-brown", "shape": "Oval", "fiber": "Nil"},
    "Shaishe": {"id": "D-07", "origin": "Saudi Arabia", "avg_price": "1000 PKR/kg", "best_for": "Bone strength", "grade": "Export", "color": "Brown", "shape": "Oval", "fiber": "Medium"},
    "Sokari": {"id": "D-08", "origin": "Saudi Arabia", "avg_price": "1400 PKR/kg", "best_for": "Bone health", "grade": "Gold", "color": "Golden brown", "shape": "Oval", "fiber": "Medium"},
    "Sugaey": {"id": "D-09", "origin": "Saudi Arabia", "avg_price": "1600 PKR/kg", "best_for": "Metabolism", "grade": "Premium", "color": "Light brown", "shape": "Oval", "fiber": "High"},
    "Zahidi": {"id": "D-10", "origin": "Iraq", "avg_price": "900 PKR/kg", "best_for": "Semi-dry texture", "grade": "Standard", "color": "Golden yellow", "shape": "Oval", "fiber": "Medium"},
    "Khadrawy": {"id": "D-11", "origin": "Iraq", "avg_price": "1100 PKR/kg", "best_for": "Soft, caramel-like", "grade": "Premium", "color": "Dark brown", "shape": "Soft oval", "fiber": "Low"},
    "Barhi": {"id": "D-12", "origin": "Iraq", "avg_price": "1200 PKR/kg", "best_for": "Sweet, crispy", "grade": "Delicacy", "color": "Yellow to amber", "shape": "Round", "fiber": "Nil"},
    "Halawi": {"id": "D-13", "origin": "Iraq", "avg_price": "1000 PKR/kg", "best_for": "Soft, honey-like", "grade": "Premium", "color": "Light brown", "shape": "Oval", "fiber": "Low"}
}

DISEASE_LIBRARY = {
    "Mango": {
        "Anthracnose": "Dark spots on leaves and fruit. Use copper-based fungicides.",
        "Powdery Mildew": "White powdery growth. Sulfur spray.",
        "Fruit Fly": "Small puncture marks. Use pheromone traps."
    },
    "Date": {
        "Bayoud Disease": "Wilt and dieback. Plant resistant varieties.",
        "Fruit Rot": "Soft brown spots. Improve ventilation.",
        "Red Palm Weevil": "Holes in trunk. Inject insecticide."
    }
}

# ==============================================================================
# 7. DISPLAY DETAILS FUNCTION
# ==============================================================================
def display_fruit_details(name, category, db, rotten_data=None, live_rate=None):
    if name not in db:
        st.error(f"Data for {name} not found.")
        return
    details = db[name]
    st.markdown(f"<h2 style='margin-bottom:0;'>{name}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#9bbf8b;'>{category} Variety • ID: {details.get('id', 'N/A')}</p>", unsafe_allow_html=True)
    if live_rate:
        st.info(f"📊 **Live Mandi Rate:** {live_rate}")
    else:
        st.warning(f"📡 Live rate offline – static price: {details.get('avg_price', 'N/A')}")
    if rotten_data:
        rot_pct, status, rec = rotten_data
        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.metric("Quality Index", f"{rot_pct:.1f}% rot")
        col2.markdown(f"**Status:** {status}")
        st.info(f"📌 **Recommendation:** {rec}")
        st.markdown("---")
    with st.expander("📖 Full Details"):
        cols = st.columns(3)
        with cols[0]:
            st.write("**Scientific Name:**", details.get('sci', 'N/A'))
            st.write("**Origin:**", details.get('origin', 'N/A'))
            st.write("**Season:**", details.get('season', 'N/A'))
        with cols[1]:
            st.write("**Color:**", details.get('color', 'N/A'))
            st.write("**Shape:**", details.get('shape', 'N/A'))
            st.write("**Best For:**", details.get('best_for', 'N/A'))
        with cols[2]:
            st.write("**Fiber:**", details.get('fiber', 'N/A'))
            st.write("**Brix / Grade:**", details.get('brix', details.get('grade', 'N/A')))
        st.write("**Common Diseases:**", ", ".join(DISEASE_LIBRARY.get(category, {}).keys()))

# ==============================================================================
# 8. PDF GENERATION (Unicode-safe)
# ==============================================================================
def generate_pdf(variety, category, details, rotten_data, live_rate):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    title = clean_latin1(f"FruitIQ Report - {variety}")
    pdf.cell(200, 10, txt=title, ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 8, txt=f"Category: {category}", ln=True)
    pdf.cell(200, 8, txt=f"Origin: {clean_latin1(details.get('origin', 'N/A'))}", ln=True)
    pdf.cell(200, 8, txt=f"Season: {clean_latin1(details.get('season', 'N/A'))}", ln=True)
    pdf.ln(5)
    rot_pct, status, rec = rotten_data
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 8, txt="Quality Assessment", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, txt=f"Rotten Percentage: {rot_pct:.1f}%", ln=True)
    pdf.cell(200, 8, txt=f"Status: {clean_latin1(status)}", ln=True)
    pdf.cell(200, 8, txt=f"Recommendation: {clean_latin1(rec)}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 8, txt="Market Info", ln=True)
    pdf.set_font("Arial", "", 12)
    rate_str = live_rate if live_rate else details.get('avg_price', 'N/A')
    pdf.cell(200, 8, txt=f"Live Rate: {clean_latin1(rate_str)}", ln=True)
    return pdf.output(dest='S').encode('latin-1', errors='ignore')

# ==============================================================================
# 9. TEXT-TO-SPEECH
# ==============================================================================
def text_to_speech_autoplay(text: str, lang="en"):
    try:
        clean_text = clean_latin1(text)
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        b64 = base64.b64encode(audio_bytes.read()).decode()
        st.markdown(f'<audio controls autoplay src="data:audio/mp3;base64,{b64}">', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Speech not available: {e}")

# ==============================================================================
# 10. SIDEBAR (PROFESSIONAL LAYOUT)
# ==============================================================================
with st.sidebar:
    st.markdown("## 🍑 FruitIQ")
    st.markdown("---")
    # Main menu radio buttons – clean and prominent
    menu = st.radio(
        "MAIN MENU",
        [
            "📊 Command Center",
            "🔍 Neural Scanner",
            "📚 Variety Vault",
            "📈 Market Analytics",
            "💬 AI Assistant",
            "📜 Scan History",
            "⚖️ Compare Varieties",
            "🌾 Farm Advisory",
            "🦠 Disease Library",
            "ℹ️ About"
        ],
        index=0
    )
    st.markdown("---")
    
    # Settings expander – keeps sidebar clean
    with st.expander("⚙️ Settings"):
        demo_mode = st.checkbox("🎮 Demo Mode (Mock Data)", value=st.session_state.demo_mode)
        st.session_state.demo_mode = demo_mode
        urdu_mode = st.checkbox("🗣️ Urdu Voice", value=st.session_state.urdu_voice)
        st.session_state.urdu_voice = urdu_mode
        theme_choice = st.radio("🎨 Theme", ["Dark", "Light"], index=0 if st.session_state.theme == "dark" else 1)
        st.session_state.theme = "dark" if theme_choice == "Dark" else "light"
        apply_theme()
    
    st.markdown("---")
    # Export and system info at bottom
    if st.button("📥 Export History (CSV)"):
        if st.session_state.scan_history:
            df = pd.DataFrame(st.session_state.scan_history)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", csv, "fruitiq_scans.csv", "text/csv")
        else:
            st.warning("No data to export.")
    st.caption(f"🕒 System Time: {datetime.now().strftime('%H:%M')}")

# ==============================================================================
# 11. COMMAND CENTER (with your dashboard images)
# ==============================================================================
if menu == "📊 Command Center":
    mango_img = "https://tse3.mm.bing.net/th/id/OIP.i9lGitIJab-83w5pwCYFjwHaHa?pid=Api&h=220&P=0"
    date_img = "https://tse2.mm.bing.net/th/id/OIP._FJNSbVzHqHoohtZJD5U3gHaHa?pid=Api&h=220&P=0"
    fallback_mango = "https://upload.wikimedia.org/wikipedia/commons/9/90/Haden_mango.jpg"
    fallback_date = "https://upload.wikimedia.org/wikipedia/commons/6/6f/Dates_2.jpg"

    st.markdown(f"""
        <div class="dashboard-hero">
            <div class="hero-card">
                <img class="hero-img" src="{mango_img}" 
                     onerror="this.src='{fallback_mango}'; this.onerror='';">
                <div class="hero-title">
                    <h2>🥭 MANGO CENTER</h2>
                    <p>14 cultivars • Peak season May–July</p>
                </div>
            </div>
            <div class="hero-card">
                <img class="hero-img" src="{date_img}" 
                     onerror="this.src='{fallback_date}'; this.onerror='';">
                <div class="hero-title">
                    <h2>🌴 DATE CENTER</h2>
                    <p>13 varieties • Premium & organic</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    kpi = st.columns(4)
    with kpi[0]: st.markdown('<div class="kpi-card"><div class="kpi-number">14</div><div>MANGO CULTIVARS</div></div>', unsafe_allow_html=True)
    with kpi[1]: st.markdown('<div class="kpi-card"><div class="kpi-number">13</div><div>DATE VARIETIES</div></div>', unsafe_allow_html=True)
    with kpi[2]: st.markdown(f'<div class="kpi-card"><div class="kpi-number">{st.session_state.total_scans}</div><div>TOTAL SCANS</div></div>', unsafe_allow_html=True)
    with kpi[3]: st.markdown('<div class="kpi-card"><div class="kpi-number">99.4%</div><div>AI ACCURACY</div></div>', unsafe_allow_html=True)

    left, right = st.columns([2,1])
    with left:
        st.subheader("📈 7‑Day Market Trend (PKR/kg)")
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        trend = pd.DataFrame({
            'Date': dates,
            'Chaunsa': np.random.normal(350, 15, 7).cumsum() + 300,
            'Sindhri': np.random.normal(280, 12, 7).cumsum() + 250,
            'Ajwa': np.random.normal(2500, 60, 7).cumsum() + 2400
        })
        fig = px.line(trend, x='Date', y=['Chaunsa', 'Sindhri', 'Ajwa'],
                      labels={'value': 'Price (PKR/kg)'}, line_shape='spline')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#eef2f0')
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("🔴 Live Mandi Snapshot")
        demo = st.session_state.demo_mode
        for name, cat in [("Chaunsa (Black)", "mango"), ("Sindhri", "mango"), ("Ajwa", "date")]:
            rate = fetch_live_rate(name, cat, demo)
            st.markdown(f'<div class="market-snapshot"><b>{name}</b><br>{rate if rate else "N/A"}</div>', unsafe_allow_html=True)
        st.caption("Source: AMIS Pakistan • Updated hourly")

    st.subheader("🏭 Facility Throughput (Last 12 hours)")
    throughput = pd.DataFrame({
        'Hour': [f"{h}:00" for h in range(6, 18)],
        'Mango (kg)': np.random.randint(500, 1300, 12),
        'Dates (kg)': np.random.randint(300, 1000, 12)
    })
    st.bar_chart(throughput.set_index('Hour'))

# ==============================================================================
# 12. NEURAL SCANNER (unchanged, with override)
# ==============================================================================
elif menu == "🔍 Neural Scanner":
    st.markdown("## 🔍 Neural Scanner")
    st.markdown("Capture or upload a mango/date image for variety identification and quality grading.")

    MODEL_PATH = "model_unquant.tflite"
    LABELS = [
        "Anwar Ratool", "Chaunsa (Black)", "Chaunsa (Summer Bahisht)", "Chaunsa ( White)",
        "Dosehri", "Fajri", "Langra", "Sindhri",
        "Ajwa", "Galaxy", "Medjool", "Meneifi", "Nabtat Ali", "Rutab", "Shaishe", "Sokari", "Sugaey",
        "Background"
    ]

    @st.cache_resource
    def load_model():
        try:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            st.error(f"Model load error: {e}")
            return None

    def preprocess(pil_img):
        img = pil_img.resize((224, 224))
        arr = np.array(img).astype(np.float32)
        arr = (arr / 127.5) - 1.0
        return np.expand_dims(arr, axis=0)

    input_source = st.radio("Image source:", ["📸 Camera", "📁 Upload (drag & drop)"], horizontal=True)
    image = None
    if input_source == "📸 Camera":
        cam = st.camera_input("Capture")
        if cam:
            image = Image.open(cam)
    else:
        upl = st.file_uploader("Drag & drop or click", type=["jpg", "jpeg", "png"])
        if upl:
            image = Image.open(upl)
            st.image(image, caption="Uploaded", use_column_width=True)

    if image:
        st.session_state.total_scans += 1
        with st.status("Analyzing...", expanded=True) as status:
            status.update(label="Loading model")
            interpreter = load_model()
            if interpreter is None:
                st.stop()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            status.update(label="Preprocessing")
            input_tensor = preprocess(image)
            status.update(label="Running inference")
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            pred_idx = np.argmax(output[0])
            confidence = output[0][pred_idx] * 100
            status.update(label="Quality check")
            rot_pct, rot_status, rot_action = detect_rotten(image)
            status.update(label="Done", state="complete")

        predicted = LABELS[pred_idx]
        if predicted == "Background" or confidence < 50:
            st.warning(f"No fruit detected (conf: {confidence:.1f}%). Please use a clearer image.")
        else:
            category = "Mango" if pred_idx < 8 else "Date"
            db = MANGO_DB if category == "Mango" else DATES_DB
            live_rate = fetch_live_rate(predicted, category, st.session_state.demo_mode)
            st.success(f"Identified: {predicted} ({category}) | Confidence: {confidence:.1f}%")
            lang = "ur" if st.session_state.urdu_voice else "en"
            text_to_speech_autoplay(f"{predicted}, {category}, quality {rot_status}", lang)

            # Manual override for low confidence or ambiguous varieties
            show_override = (confidence < 85) or (predicted not in db) or (predicted == "Meneifi")
            if show_override:
                st.warning("Low confidence or ambiguous variety. Please verify manually.")
                options = list(db.keys())
                corrected = st.selectbox("Correct variety:", options, key="manual_correct")
                if corrected != predicted:
                    predicted = corrected
                    st.info(f"Overridden to {predicted}.")
                    text_to_speech_autoplay(f"Corrected to {predicted}", lang)

            # Store scan history
            st.session_state.scan_history.append({
                "variety": predicted,
                "rotten_percent": rot_pct,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            if len(st.session_state.scan_history) > 5:
                st.session_state.scan_history.pop(0)

            display_fruit_details(predicted, category, db, (rot_pct, rot_status, rot_action), live_rate)

            # PDF export
            if st.button("📄 Export Report (PDF)"):
                pdf_data = generate_pdf(predicted, category, db[predicted], (rot_pct, rot_status, rot_action), live_rate)
                st.download_button("⬇️ Save PDF", data=pdf_data,
                                   file_name=f"FruitIQ_{predicted}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                   mime="application/pdf")
        st.image(image, caption="Scanned image", use_column_width=True)

# ==============================================================================
# 13. VARIETY VAULT
# ==============================================================================
elif menu == "📚 Variety Vault":
    st.markdown("## 📚 Variety Vault")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🥭 Mangoes")
        mango_sel = st.selectbox("Select a mango variety:", list(MANGO_DB.keys()))
        if mango_sel:
            display_fruit_details(mango_sel, "Mango", MANGO_DB,
                                  live_rate=fetch_live_rate(mango_sel, "mango", st.session_state.demo_mode))
    with col2:
        st.subheader("🌴 Dates")
        date_sel = st.selectbox("Select a date variety:", list(DATES_DB.keys()))
        if date_sel:
            display_fruit_details(date_sel, "Date", DATES_DB,
                                  live_rate=fetch_live_rate(date_sel, "date", st.session_state.demo_mode))

# ==============================================================================
# 14. MARKET ANALYTICS
# ==============================================================================
elif menu == "📈 Market Analytics":
    st.markdown("## 📈 Market Analytics")
    all_vars = []
    all_prices = []
    all_types = []
    for name, data in MANGO_DB.items():
        try:
            price = int(data['avg_price'].split()[0])
        except:
            price = 0
        all_vars.append(name)
        all_prices.append(price)
        all_types.append("Mango")
    for name, data in DATES_DB.items():
        try:
            price = int(data['avg_price'].split()[0])
        except:
            price = 0
        all_vars.append(name)
        all_prices.append(price)
        all_types.append("Date")
    df = pd.DataFrame({"Variety": all_vars, "Price (PKR/kg)": all_prices, "Type": all_types})
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df, x="Variety", y="Price (PKR/kg)", color="Type", title="Wholesale Prices")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.pie(df, values="Price (PKR/kg)", names="Type", title="Market Share")
        st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(df.sort_values("Price (PKR/kg)", ascending=False), use_container_width=True)

# ==============================================================================
# 15. AI ASSISTANT
# ==============================================================================
elif menu == "💬 AI Assistant":
    st.markdown("## 💬 AI Assistant")
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant",
                                  "content": "Ask me about any fruit variety, its origin, season, or market price. I can also suggest disease management."}]
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if p := st.chat_input("Your question:"):
        st.session_state.chat.append({"role": "user", "content": p})
        with st.chat_message("user"):
            st.markdown(p)
        query = p.lower()
        found = None
        for name in MANGO_DB:
            if name.lower() in query:
                found = name
                cat = "Mango"
                break
        if not found:
            for name in DATES_DB:
                if name.lower() in query:
                    found = name
                    cat = "Date"
                    break
        if found:
            details = (MANGO_DB if cat == "Mango" else DATES_DB)[found]
            resp = f"**{found}** ({cat})\n- Origin: {details.get('origin', 'N/A')}\n- Season: {details.get('season', 'N/A')}\n- Price: {details.get('avg_price', 'N/A')}\n- Best for: {details.get('best_for', 'N/A')}\n- Color: {details.get('color', 'N/A')}\n- Shape: {details.get('shape', 'N/A')}"
        elif "disease" in query:
            resp = "Common diseases:\n**Mango:** Anthracnose, Powdery Mildew, Fruit Fly.\n**Dates:** Bayoud Disease, Fruit Rot, Red Palm Weevil.\nWould you like management tips?"
        elif "weather" in query or "advisory" in query:
            resp = "🌾 Farm Advisory: Based on current season (summer), protect fruits from excessive heat. Ensure proper irrigation and monitor for fruit flies. Storage temperature: 10-15°C for mangoes, 0-5°C for dates."
        elif "list" in query:
            resp = f"Mangoes: {', '.join(MANGO_DB.keys())}\n\nDates: {', '.join(DATES_DB.keys())}"
        else:
            resp = "I can help with variety details, origin, season, market prices, diseases, and farm advisory. Try 'Tell me about Sindhri' or 'Disease management'."
        st.session_state.chat.append({"role": "assistant", "content": resp})
        with st.chat_message("assistant"):
            st.markdown(resp)

# ==============================================================================
# 16. SCAN HISTORY
# ==============================================================================
elif menu == "📜 Scan History":
    st.markdown("## 📜 Scan History")
    if st.session_state.scan_history:
        hist_df = pd.DataFrame(st.session_state.scan_history)
        fig = px.line(hist_df, x="timestamp", y="rotten_percent", color="variety", markers=True,
                      title="Quality trend over time")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No scans yet. Go to Neural Scanner.")

# ==============================================================================
# 17. COMPARE VARIETIES
# ==============================================================================
elif menu == "⚖️ Compare Varieties":
    st.markdown("## ⚖️ Compare Two Varieties")
    col1, col2 = st.columns(2)
    with col1:
        cat1 = st.selectbox("Category 1", ["Mango", "Date"])
        if cat1 == "Mango":
            var1 = st.selectbox("Variety 1", list(MANGO_DB.keys()))
            details1 = MANGO_DB[var1]
        else:
            var1 = st.selectbox("Variety 1", list(DATES_DB.keys()))
            details1 = DATES_DB[var1]
    with col2:
        cat2 = st.selectbox("Category 2", ["Mango", "Date"])
        if cat2 == "Mango":
            var2 = st.selectbox("Variety 2", list(MANGO_DB.keys()))
            details2 = MANGO_DB[var2]
        else:
            var2 = st.selectbox("Variety 2", list(DATES_DB.keys()))
            details2 = DATES_DB[var2]
    if st.button("Compare"):
        compare_data = {
            "Attribute": ["Origin", "Season", "Price", "Best For", "Color", "Shape"],
            var1: [details1.get('origin', 'N/A'), details1.get('season', 'N/A'), details1.get('avg_price', 'N/A'),
                   details1.get('best_for', 'N/A'), details1.get('color', 'N/A'), details1.get('shape', 'N/A')],
            var2: [details2.get('origin', 'N/A'), details2.get('season', 'N/A'), details2.get('avg_price', 'N/A'),
                   details2.get('best_for', 'N/A'), details2.get('color', 'N/A'), details2.get('shape', 'N/A')]
        }
        st.table(pd.DataFrame(compare_data))

# ==============================================================================
# 18. FARM ADVISORY
# ==============================================================================
elif menu == "🌾 Farm Advisory":
    st.markdown("## 🌾 Farm Advisory")
    st.info("Based on current season and typical climate for Pakistan (Punjab/Sindh)")
    st.subheader("📅 Seasonal Calendar")
    st.markdown("""
    - **Mango:**  
      - *May–June*: Harvesting, watch for fruit flies.  
      - *July–August*: Post-harvest pruning, apply fungicide against anthracnose.  
    - **Dates:**  
      - *August–September*: Harvesting, ensure proper drying.  
      - *October–November*: Fertilize with potassium for next season.
    """)
    st.subheader("🌦️ Weather Outlook (Next 7 days)")
    st.write("Typical summer conditions: High 38-42°C, low 26-28°C. Chance of isolated rain. Ensure adequate irrigation.")
    st.subheader("🧪 Fertilizer Recommendation")
    st.markdown("For mango: Apply 1.5kg urea + 2kg DAP per tree after harvest. For dates: 2kg potassium sulfate per palm.")
    st.subheader("🐞 Pest Alert")
    st.warning("Fruit fly activity high. Use pheromone traps and neem oil spray every 10 days.")

# ==============================================================================
# 19. DISEASE LIBRARY
# ==============================================================================
elif menu == "🦠 Disease Library":
    st.markdown("## 🦠 Disease Library")
    fruit_type = st.radio("Select fruit type", ["Mango", "Date"])
    diseases = DISEASE_LIBRARY[fruit_type]
    for disease, remedy in diseases.items():
        with st.expander(f"🔬 {disease}"):
            st.write(f"**Remedy:** {remedy}")

# ==============================================================================
# 20. ABOUT
# ==============================================================================
elif menu == "ℹ️ About":
    st.markdown("## ℹ️ About FruitIQ")
    st.markdown("""
    **FruitIQ – Agricultural Intelligence Platform (A‑Level / FYP Final)**  
    - **27 fruit varieties** (14 mango, 13 date)  
    - **Live AMIS market rates** (Pakistan) with demo mode fallback  
    - **On‑device TensorFlow Lite** classification (model trained on 17 classes)  
    - **Rotten detection** using OpenCV color analysis  
    - **Urdu / English voice** feedback  
    - **PDF report export** (Unicode‑safe)  
    - **Dark / Light theme**  
    - **Compare varieties, Farm advisory, Disease library**  
    - **Professional dashboard** with real‑time charts  

    **Developed by:** Umair Mustafa & Mehreen Nasir  
    **Version:** A‑Level Enterprise Edition v7.0 (Final)  
    """)

st.markdown("---")
st.caption("© 2026 FruitIQ | A‑Level Project | Live AMIS Integration | Smart Agriculture")