import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import qrcode
from io import BytesIO
import json
import time

# Configuration de la page
st.set_page_config(
    page_title="FindSpot - Détection de Places de Stationnement",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import des modèles
try:
    from mobilenet import MobileNetV3Classifier
except ImportError:
    st.error("Erreur: Impossible d'importer MobileNetV3Classifier.")

try:
    from efficientnet import EfficientNetClassifier
except ImportError:
    pass  # EfficientNet optionnel

try:
    from resnet18 import ResNet18Classifier
except ImportError:
    pass  # ResNet18 optionnel

# Chemins des modèles
MODEL_PATHS = {
    'mobilenet': "best_mobilenet.pt",
    'efficientnet': "best_efficientnet.pt",
    'resnet': "best_resnet.pt"
}

# Métriques des modèles
MODELS_METRICS = {
    "MobileNetV3": {
        "accuracy": 97.79,
        "val_accuracy": 97.85,
        "inference_time": 17.94,
        "fps": 56,
        "model_size": 2.54,
        "params": 1.52
    },
    "EfficientNet": {
        "accuracy": 96.98,
        "val_accuracy": 98.06,
        "inference_time": 27.37,
        "fps": 36.53,
        "model_size": 15.59,
        "params": 4.01
    },
    "ResNet18": {
        "accuracy": 94.97,
        "val_accuracy": 95.85,
        "inference_time": 4.81,
        "fps": 208.07,
        "model_size": 42.71,
        "params": 11.18
    }
}

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Classes
CLASSES = ["Libre", "Occupé"]

@st.cache_resource
def load_model(model_type='mobilenet'):
    """Charge un modèle spécifique"""
    try:
        if model_type == 'mobilenet':
            model = MobileNetV3Classifier(num_classes=2, pretrained=False, version='small')
            checkpoint_path = MODEL_PATHS['mobilenet']
        elif model_type == 'efficientnet':
            model = EfficientNetClassifier(num_classes=2)
            checkpoint_path = MODEL_PATHS['efficientnet']
        elif model_type == 'resnet':
            model = ResNet18Classifier(num_classes=2)
            checkpoint_path = MODEL_PATHS['resnet']
        else:
            return None
        
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Erreur chargement {model_type}: {e}")
        return None

def predict_image(model, image):
    """Fait une prédiction sur l'image"""
    try:
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        return predicted_class, confidence, probabilities[0].numpy()
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None, None, None

def generate_qr_code(url):
    """Génère un QR code pour l'URL donnée"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    return img

def find_rois_for_image(image_name, annotations):
    """Trouve les ROIs pour une image donnée"""
    try:
        if "train" in annotations:
            file_names = annotations["train"]["file_names"]
            rois_list = annotations["train"]["rois_list"]
        elif "test" in annotations:
            file_names = annotations["test"]["file_names"]
            rois_list = annotations["test"]["rois_list"]
        elif "file_names" in annotations:
            file_names = annotations["file_names"]
            rois_list = annotations["rois_list"]
        else:
            return None
        
        if image_name in file_names:
            index = file_names.index(image_name)
            return rois_list[index]
        
        return None
    
    except (KeyError, IndexError):
        return None

def analyze_parking_with_rois(image, rois, model):
    """Analyse chaque place de parking définie par les ROIs"""
    results = []
    img_width, img_height = image.size
    
    for roi in rois:
        points = []
        for point in roi:
            x = int(point[0] * img_width)
            y = int(point[1] * img_height)
            points.append((x, y))
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        try:
            cropped = image.crop((x_min, y_min, x_max, y_max))
            predicted_class, confidence, _ = predict_image(model, cropped)
            
            results.append({
                'prediction': predicted_class,
                'confidence': confidence,
                'points': points
            })
        except Exception as e:
            results.append({
                'prediction': None,
                'confidence': 0,
                'points': points,
                'error': str(e)
            })
    
    return results

def display_annotated_results(image, rois, results, show_labels, show_confidence, 
                              line_width, font_size):
    """Affiche l'image avec les prédictions sur chaque place"""
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image, 'RGBA')
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    num_libre = sum(1 for r in results if r['prediction'] == 0)
    num_occupe = sum(1 for r in results if r['prediction'] == 1)
    num_error = sum(1 for r in results if r['prediction'] is None)
    
    for i, result in enumerate(results):
        points = result['points']
        prediction = result['prediction']
        confidence = result['confidence']
        
        if prediction == 0:
            color = (0, 255, 0, 100)
            outline_color = (0, 200, 0, 255)
            label = "Libre"
        elif prediction == 1:
            color = (255, 0, 0, 100)
            outline_color = (200, 0, 0, 255)
            label = "Occupé"
        else:
            color = (128, 128, 128, 100)
            outline_color = (100, 100, 100, 255)
            label = "Erreur"
        
        draw.polygon(points, fill=color, outline=outline_color, width=line_width)
        
        if show_labels or show_confidence:
            center_x = sum(p[0] for p in points) // 4
            center_y = sum(p[1] for p in points) // 4
            
            text_parts = []
            if show_labels:
                text_parts.append(f"#{i+1}")
            if show_confidence and prediction is not None:
                text_parts.append(f"{confidence:.0f}%")
            
            text = " ".join(text_parts)
            
            try:
                bbox = draw.textbbox((center_x, center_y), text, font=font)
                draw.rectangle(bbox, fill=(0, 0, 0, 180))
                draw.text((center_x, center_y), text, fill='white', font=font, anchor='mm')
            except:
                draw.text((center_x, center_y), text, fill='white')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(annotated_image, caption=f"Parking analysé - {len(results)} places", 
                use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Résumé")
        
        st.metric("Places Totales", len(results))
        st.metric("🟢 Libres", f"{num_libre} ({num_libre/len(results)*100:.1f}%)")
        st.metric("🔴 Occupées", f"{num_occupe} ({num_occupe/len(results)*100:.1f}%)")
        
        if num_error > 0:
            st.metric("⚠️ Erreurs", num_error)
        
        st.markdown("### Distribution")
        if len(results) > 0:
            progress_html = f"""
            <div style="background-color: #f0f0f0; border-radius: 10px; overflow: hidden; height: 30px; display: flex; margin: 10px 0;">
                <div style="background-color: #2ecc71; width: {num_libre/len(results)*100}%; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: bold;">
                    {num_libre}
                </div>
                <div style="background-color: #e74c3c; width: {num_occupe/len(results)*100}%; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: bold;">
                    {num_occupe}
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        valid_confidences = [r['confidence'] for r in results if r['prediction'] is not None]
        if valid_confidences:
            avg_confidence = sum(valid_confidences) / len(valid_confidences)
            st.metric("Confiance Moyenne", f"{avg_confidence:.1f}%")
    
    with st.expander("📋 Détails par Place"):
        for i, result in enumerate(results):
            if result['prediction'] is not None:
                label = "🟢 Libre" if result['prediction'] == 0 else "🔴 Occupé"
                st.write(f"**Place {i+1}:** {label} - Confiance: {result['confidence']:.1f}%")
            else:
                st.write(f"**Place {i+1}:** ⚠️ Erreur - {result.get('error', 'Inconnu')}")
    
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        buf = BytesIO()
        annotated_image.save(buf, format="PNG")
        
        st.download_button(
            label="💾 Télécharger l'Image Annotée",
            data=buf.getvalue(),
            file_name=f"parking_annotated_{len(results)}_places.png",
            mime="image/png"
        )
    
    with col_dl2:
        report = f"""
RAPPORT D'ANALYSE DE PARKING
═══════════════════════════════════════

Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Places analysées: {len(results)}

RÉSULTATS:
─────────────────────────────────────
Places Libres:    {num_libre} ({num_libre/len(results)*100:.1f}%)
Places Occupées:  {num_occupe} ({num_occupe/len(results)*100:.1f}%)

DÉTAILS PAR PLACE:
─────────────────────────────────────
"""
        for i, result in enumerate(results):
            if result['prediction'] is not None:
                label = "Libre" if result['prediction'] == 0 else "Occupé"
                report += f"Place {i+1}: {label} ({result['confidence']:.1f}%)\n"
        
        report += f"\nGénéré par FindSpot - GIF-4101\nUniversité Laval - Automne 2025"
        
        st.download_button(
            label="📄 Télécharger le Rapport",
            data=report,
            file_name=f"rapport_parking_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ============================================
# COMPARAISON DES MODÈLES - NOUVELLE FEATURE
# ============================================

def show_model_comparison():
    """Page de comparaison des 3 modèles sur une même image"""
    st.title("⚔️ Comparaison des Modèles")
    
    st.markdown("""
    ## Démarche Scientifique Rigoureuse
    
    Pour choisir le meilleur modèle pour FindSpot, nous avons **comparé 3 architectures CNN** 
    sur le même dataset. Cette page vous permet de voir comment chaque modèle performe 
    sur **la même image**.
    """)
    
    # Upload image
    uploaded_file = st.file_uploader(
        "📸 Uploadez une image de place de stationnement",
        type=["jpg", "jpeg", "png"],
        key="comparison_upload"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Afficher l'image
        st.image(image, caption="Image à analyser", width=400)
        
        st.markdown("---")
        
        # Bouton de comparaison
        if st.button("🔍 Comparer les 3 Modèles", type="primary", use_container_width=True):
            
            st.markdown("## 📊 Résultats de la Comparaison")
            
            # Créer 3 colonnes
            col1, col2, col3 = st.columns(3)
            
            # Configuration des modèles
            models_config = [
                {
                    'name': 'MobileNetV3-Small',
                    'type': 'mobilenet',
                    'col': col1,
                    'color': '#2ecc71',
                    'icon': '📱',
                    'tagline': 'Le plus léger',
                    'metrics': MODELS_METRICS['MobileNetV3']
                },
                {
                    'name': 'EfficientNet-B0',
                    'type': 'efficientnet',
                    'col': col2,
                    'color': '#3498db',
                    'icon': '⚖️',
                    'tagline': 'Meilleure validation',
                    'metrics': MODELS_METRICS['EfficientNet']
                },
                {
                    'name': 'ResNet18',
                    'type': 'resnet',
                    'col': col3,
                    'color': '#e74c3c',
                    'icon': '⚡',
                    'tagline': 'Le plus rapide',
                    'metrics': MODELS_METRICS['ResNet18']
                }
            ]
            
            results = []
            
            # Pour chaque modèle
            for config in models_config:
                with config['col']:
                    st.markdown(f"### {config['icon']} {config['name']}")
                    st.caption(config['tagline'])
                    
                    # Essayer de charger le modèle
                    with st.spinner(f"Chargement..."):
                        model = load_model(config['type'])
                    
                    if model is not None:
                        # Prédiction réelle
                        with st.spinner("Analyse..."):
                            start = time.time()
                            predicted_class, confidence, _ = predict_image(model, image)
                            inference_time = (time.time() - start) * 1000
                            
                            if predicted_class is not None:
                                prediction_text = CLASSES[predicted_class]
                                color = "#2ecc71" if predicted_class == 0 else "#e74c3c"
                                
                                st.markdown(
                                    f"<h2 style='text-align: center; color: {color};'>{prediction_text}</h2>",
                                    unsafe_allow_html=True
                                )
                                
                                st.metric("Confiance", f"{confidence:.2f}%")
                                st.metric("Temps", f"{inference_time:.2f} ms")
                                
                                results.append({
                                    'model': config['name'],
                                    'prediction': prediction_text,
                                    'predicted_class': predicted_class,
                                    'confidence': confidence,
                                    'time': inference_time,
                                    'color': config['color']
                                })
                            else:
                                st.error("Erreur de prédiction")
                    else:
                        # Afficher métriques moyennes
                        st.info(f"Modèle non chargé")
                        st.caption("Métriques moyennes du test:")
                        st.metric("Test Accuracy", f"{config['metrics']['accuracy']:.2f}%")
                        st.metric("Temps Moyen", f"{config['metrics']['inference_time']:.2f} ms")
            
            # Analyse comparative
            if len(results) > 0:
                st.markdown("---")
                st.markdown("## 🎯 Analyse Comparative")
                
                # Consensus ou divergence?
                predictions = [r['prediction'] for r in results]
                if len(set(predictions)) == 1:
                    st.success(f"✅ **Consensus parfait:** Tous les modèles prédisent **{predictions[0]}**")
                else:
                    st.warning("⚠️ **Divergence détectée:** Les modèles ne sont pas tous d'accord")
                    for r in results:
                        emoji = "🟢" if r['predicted_class'] == 0 else "🔴"
                        st.write(f"{emoji} **{r['model']}:** {r['prediction']} ({r['confidence']:.1f}% confiance)")
                
                # Graphiques comparatifs
                st.markdown("### 📊 Comparaison Visuelle")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Graphique 1: Confiances
                model_names = [r['model'].replace('-Small', '').replace('-B0', '') for r in results]
                confidences = [r['confidence'] for r in results]
                colors_conf = [r['color'] for r in results]
                
                bars1 = ax1.bar(model_names, confidences, color=colors_conf, alpha=0.7, edgecolor='black', linewidth=2)
                ax1.set_ylabel('Confiance (%)', fontsize=11, fontweight='bold')
                ax1.set_title('Niveau de Confiance par Modèle', fontsize=13, fontweight='bold')
                ax1.set_ylim(0, 100)
                ax1.grid(True, alpha=0.3, axis='y')
                
                for i, v in enumerate(confidences):
                    ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)
                
                # Graphique 2: Temps d'inférence
                times = [r['time'] for r in results]
                
                bars2 = ax2.bar(model_names, times, color=colors_conf, alpha=0.7, edgecolor='black', linewidth=2)
                ax2.set_ylabel('Temps (ms)', fontsize=11, fontweight='bold')
                ax2.set_title('Temps d\'Inférence', fontsize=13, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                
                for i, v in enumerate(times):
                    ax2.text(i, v + max(times)*0.02, f'{v:.1f}ms', ha='center', fontweight='bold', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Insights
                st.markdown("### 💡 Observations")
                
                fastest = min(results, key=lambda x: x['time'])
                most_confident = max(results, key=lambda x: x['confidence'])
                
                col_ins1, col_ins2 = st.columns(2)
                
                with col_ins1:
                    st.info(f"⚡ **Plus rapide:** {fastest['model']} ({fastest['time']:.2f} ms)")
                    
                with col_ins2:
                    st.info(f"🎯 **Plus confiant:** {most_confident['model']} ({most_confident['confidence']:.1f}%)")
                
                # Pourquoi MobileNetV3?
                st.markdown("---")
                st.markdown("### 🏆 Pourquoi MobileNetV3 a été choisi pour FindSpot?")
                
                st.success("""
                **Décision basée sur les données:**
                
                - ✅ **Meilleure test accuracy (97.79%)** - Performance réelle optimale
                - ✅ **Le plus léger (2.54 MB)** - Déploiement facile sur cloud
                - ✅ **Vitesse suffisante (56 FPS)** - Largement assez pour notre usage
                - ✅ **Trade-off optimal** - Équilibre parfait pour une application web
                
                **ResNet18** serait meilleur pour un système avec GPU dédié (208 FPS!).
                
                **EfficientNet** serait meilleur si validation accuracy était critique (98.06%).
                """)
    
    else:
        st.info("👆 Uploadez une image pour commencer la comparaison")
        
        st.markdown("---")
        st.markdown("## 🔬 Méthodologie de Comparaison")
        
        st.markdown("""
        ### Pourquoi comparer 3 architectures?
        
        Dans un projet de machine learning rigoureux, il est essentiel de:
        
        1. **Tester plusieurs architectures** - Ne pas se limiter à une seule approche
        2. **Comparer objectivement** - Mêmes données, mêmes conditions
        3. **Analyser les trade-offs** - Vitesse vs précision vs taille
        4. **Justifier le choix final** - Décision basée sur données, pas intuition
        
        ### Les 3 architectures testées:
        """)
        
        col_arch1, col_arch2, col_arch3 = st.columns(3)
        
        with col_arch1:
            st.markdown("""
            **📱 MobileNetV3-Small**
            - Test Acc: **97.79%** 🏆
            - Val Acc: 97.85%
            - Taille: **2.54 MB** 🏆
            - FPS: 56
            - **✅ Choisi pour FindSpot**
            
            *Optimisé pour mobile et edge devices*
            """)
        
        with col_arch2:
            st.markdown("""
            **⚖️ EfficientNet-B0**
            - Test Acc: 96.98%
            - Val Acc: **98.06%** 🏆
            - Taille: 15.59 MB
            - FPS: 36.53
            
            *Architecture efficace, meilleure validation*
            """)
        
        with col_arch3:
            st.markdown("""
            **⚡ ResNet18**
            - Test Acc: 94.97%
            - Val Acc: 95.85%
            - Taille: 42.71 MB
            - FPS: **208** 🏆
            
            *Architecture classique, ultra-rapide!*
            """)

def main():
    # Sidebar
    st.sidebar.title("🅿️ FindSpot")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Accueil", "🔍 Prédiction", "⚔️ Comparaison", "🅿️ Avec Annotations", "📊 Performance", "📈 Statistiques", "👥 À propos"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### À propos")
    st.sidebar.info(
        "**Projet GIF-4101**\n\n"
        "Détection automatique d'occupation de places de stationnement "
        "utilisant l'apprentissage profond.\n\n"
        "**Université Laval - Automne 2025**"
    )
    
    # QR Code dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📱 Accès Rapide")
    
    app_url = "https://findspot.streamlit.app"
    qr_img = generate_qr_code(app_url)
    
    buf = BytesIO()
    qr_img.save(buf, format="PNG")
    st.sidebar.image(buf.getvalue(), caption="Scannez pour accéder à l'app", use_container_width=True)
    st.sidebar.caption(f"URL: {app_url}")
    
    # Pages
    if page == "🏠 Accueil":
        show_home()
    elif page == "🔍 Prédiction":
        show_prediction()
    elif page == "⚔️ Comparaison":
        show_model_comparison()
    elif page == "🅿️ Avec Annotations":
        show_annotated_prediction()
    elif page == "📊 Performance":
        show_performance()
    elif page == "📈 Statistiques":
        show_statistics()
    elif page == "👥 À propos":
        show_about_team()

def show_home():
    """Page d'accueil"""
    st.title("🅿️ FindSpot - Détection de Places de Stationnement")
    
    st.markdown("""
    ## Bienvenue sur FindSpot!
    
    FindSpot est un système de détection automatique d'occupation de places de stationnement 
    utilisant l'apprentissage profond et les réseaux de neurones convolutifs (CNN).
    
    ### 🎯 Objectif du Projet
    
    Développer un système efficace de détection d'occupation de places de stationnement capable de
    fonctionner en temps réel sur des appareils à ressources limitées.
    
    ### 🏗️ Architecture Utilisée
    
    **MobileNetV3-Small** - Architecture légère optimisée pour mobile avec:
    - Convolutions séparables en profondeur
    - Résiduels inversés et goulots d'étranglement linéaires
    - Modules Squeeze-and-Excite
    - Activation H-Swish
    
    ### 📊 Dataset
    
    **Action-Camera Parking Dataset (GoPro Hero 6)** - Images de stationnement capturées en hauteur:
    - Images capturées à ~10 mètres de hauteur
    - Caméra GoPro Hero 6
    - 2 classes: Libre et Occupé
    - 293 images avec annotations de régions d'intérêt (ROIs)
    
    ### 🚀 Comment Utiliser
    
    1. **Prédiction** - Uploadez une image pour détecter si une place est libre ou occupée
    2. **Comparaison** - Comparez les 3 modèles sur une même image
    3. **Avec Annotations** - Analysez un parking complet avec visualisation de chaque place
    4. **Performance** - Consultez les métriques détaillées du modèle
    5. **Statistiques** - Explorez les données du dataset
    """)
    
    # Métriques en colonnes
    st.markdown("### 🏆 Performance de MobileNetV3-Small")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Précision Test", "97.79%")
    with col2:
        st.metric("Score F1", "97.71%")
    with col3:
        st.metric("Vitesse", "56 FPS")
    with col4:
        st.metric("Taille", "2.54 MB")
    
    st.markdown("---")
    
    # Exemples
    st.markdown("### 📸 Fonctionnalités")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔍 Analyse Place par Place")
        st.success("Upload une image → Prédiction immédiate")
        st.info("Idéal pour: Vérification rapide d'une place")
    
    with col2:
        st.markdown("#### ⚔️ Comparaison Modèles")
        st.success("3 modèles → 1 image → Comparaison")
        st.info("Idéal pour: Comprendre les trade-offs")
    
    with col3:
        st.markdown("#### 🅿️ Analyse Parking Complet")
        st.success("Upload image + annotations → Visualisation complète")
        st.info("Idéal pour: Gestion d'un parking entier")
    
    st.markdown("---")
    
    # Section déploiement
    with st.expander("🌐 À propos du Déploiement"):
        st.markdown("""
        Cette application est déployée sur **Streamlit Cloud** et accessible partout dans le monde.
        
        **Technologies utilisées:**
        - Framework: Streamlit
        - ML: PyTorch + MobileNetV3 + EfficientNet + ResNet
        - Visualisation: Matplotlib, Seaborn, PIL
        
        **Équipe - GIF-4101:**
        - **Salem N. Nyisingize** (Créateur - MobileNetV3-Small)
        - Félix Légaré (ResNet18)
        - Rayan Nadeau (EfficientNet-B0)
        
        **Université Laval - Automne 2025**
        
        Pour plus d'informations, consultez la page **"👥 À propos"**
        """)

def show_prediction():
    """Page de prédiction"""
    st.title("🔍 Prédiction de Place de Stationnement")
    
    st.markdown("""
    Uploadez une image de place de stationnement et le modèle MobileNetV3 déterminera 
    automatiquement si la place est **Libre** ou **Occupée**.
    """)
    
    # Upload de fichier
    uploaded_file = st.file_uploader(
        "Choisissez une image de place de stationnement",
        type=["jpg", "jpeg", "png"],
        help="Formats supportés: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Afficher l'image
        image = Image.open(uploaded_file).convert('RGB')
        
        col_img, col_result = st.columns([1, 1])
        
        with col_img:
            st.image(image, caption="Image uploadée", use_container_width=True)
        
        # Charger le modèle et faire la prédiction
        with st.spinner("Chargement du modèle MobileNetV3..."):
            model = load_model('mobilenet')
        
        if model is not None:
            with st.spinner("Analyse en cours..."):
                predicted_class, confidence, probabilities = predict_image(model, image)
            
            if predicted_class is not None:
                with col_result:
                    st.markdown("### 🎯 Résultat de la Prédiction")
                    
                    # Afficher la prédiction avec un style coloré
                    prediction_text = CLASSES[predicted_class]
                    color = "green" if predicted_class == 0 else "red"
                    
                    st.markdown(
                        f"<h1 style='text-align: center; color: {color};'>{prediction_text}</h1>",
                        unsafe_allow_html=True
                    )
                    
                    st.metric("Confiance", f"{confidence:.2f}%")
                    
                    # Graphique des probabilités
                    st.markdown("#### Probabilités par Classe")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    colors_bar = ['green', 'red']
                    ax.barh(CLASSES, probabilities * 100, color=colors_bar, alpha=0.7)
                    ax.set_xlabel("Probabilité (%)")
                    ax.set_xlim(0, 100)
                    for i, v in enumerate(probabilities * 100):
                        ax.text(v + 1, i, f'{v:.1f}%', va='center')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Informations du modèle
                    st.markdown("---")
                    st.markdown("#### ℹ️ Informations du Modèle")
                    
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.write(f"**Modèle:** MobileNetV3-Small")
                        st.write(f"**Précision:** {MODELS_METRICS['MobileNetV3']['accuracy']:.2f}%")
                    with info_col2:
                        st.write(f"**Temps moyen:** {MODELS_METRICS['MobileNetV3']['inference_time']:.2f} ms")
                        st.write(f"**FPS:** {MODELS_METRICS['MobileNetV3']['fps']}")
    else:
        st.info("👆 Uploadez une image pour commencer l'analyse")
        
        # Instructions
        with st.expander("💡 Conseils pour de meilleurs résultats"):
            st.markdown("""
            - Utilisez des images claires de places de stationnement
            - Privilégiez les vues aériennes ou en angle
            - Assurez-vous que la place est bien visible
            - Le modèle fonctionne dans différentes conditions météo
            """)

def show_annotated_prediction():
    """Page de prédiction avec annotations"""
    st.title("🅿️ Analyse avec Annotations")
    
    st.markdown("""
    Uploadez une image de parking et son fichier d'annotations pour visualiser 
    la prédiction sur **chaque place individuelle**.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_image = st.file_uploader(
            "📸 Image de parking",
            type=["jpg", "jpeg", "png"],
            key="annotated_image"
        )
    
    with col2:
        uploaded_json = st.file_uploader(
            "📄 Fichier annotations.json",
            type=["json"],
            key="annotations_file"
        )
    
    if uploaded_image is not None and uploaded_json is not None:
        image = Image.open(uploaded_image).convert('RGB')
        image_name = uploaded_image.name
        
        try:
            annotations = json.load(uploaded_json)
            rois = find_rois_for_image(image_name, annotations)
            
            if rois is None:
                st.error(f"❌ Image '{image_name}' non trouvée dans le fichier d'annotations!")
                st.info("💡 Vérifiez que le nom de l'image correspond exactement à un nom dans 'file_names'")
                return
            
            st.success(f"✅ {len(rois)} places détectées dans l'image!")
            
            st.markdown("---")
            col_opts1, col_opts2 = st.columns(2)
            
            with col_opts1:
                show_labels = st.checkbox("Afficher les numéros de places", value=True)
                show_confidence = st.checkbox("Afficher la confiance", value=True)
            
            with col_opts2:
                line_width = st.slider("Épaisseur des contours", 1, 10, 3)
                font_size = st.slider("Taille du texte", 10, 40, 20)
            
            if st.button("🔍 Analyser toutes les places", type="primary"):
                with st.spinner(f"Analyse de {len(rois)} places en cours..."):
                    model = load_model('mobilenet')
                    
                    if model is not None:
                        results = analyze_parking_with_rois(image, rois, model)
                        display_annotated_results(image, rois, results, show_labels, 
                                                 show_confidence, line_width, font_size)
        
        except json.JSONDecodeError:
            st.error("❌ Erreur: Le fichier JSON est invalide!")
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
    
    else:
        st.info("👆 Uploadez une image ET son fichier d'annotations pour commencer")

def show_performance():
    """Page de performance du modèle"""
    st.title("📊 Performance du Modèle")
    
    st.markdown("""
    Métriques détaillées de performance du modèle MobileNetV3-Small sur le dataset.
    """)
    
    # Métriques principales
    st.markdown("### 🎯 Métriques de Classification")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = MODELS_METRICS['MobileNetV3']
    
    with col1:
        st.metric("Précision (Accuracy)", f"{metrics['accuracy']:.2f}%")
    with col2:
        st.metric("Précision (Precision)", "97.63%")
    with col3:
        st.metric("Rappel (Recall)", "97.80%")
    with col4:
        st.metric("Score F1", "97.71%")
    
    st.markdown("---")
    st.markdown("### ⚡ Performance d'Inférence")
    
    speed_col1, speed_col2, speed_col3 = st.columns(3)
    
    with speed_col1:
        st.metric("Temps Moyen", f"{metrics['inference_time']:.2f} ms")
    with speed_col2:
        st.metric("FPS", metrics['fps'])
    with speed_col3:
        st.metric("Taille du Modèle", f"{metrics['model_size']:.2f} MB")

def show_statistics():
    """Page de statistiques du dataset"""
    st.title("📈 Statistiques du Dataset")
    
    st.markdown("""
    ### Dataset Action-Camera Parking (GoPro Hero 6)
    
    Statistiques détaillées sur l'ensemble de données utilisé pour l'entraînement 
    et l'évaluation du modèle. Images capturées à ~10 mètres de hauteur avec GoPro Hero 6.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Images", "293")
    with col2:
        st.metric("Classes", "2")
    with col3:
        st.metric("Caméra", "GoPro Hero 6")
    with col4:
        st.metric("Hauteur", "~10m")

def show_about_team():
    """Page À propos de l'équipe - version simplifiée pour ce fichier"""
    st.title("👥 À propos de FindSpot")
    
    st.markdown("""
    ## 🎯 Le Projet
    
    **FindSpot** est un système intelligent de détection d'occupation de places de stationnement 
    développé dans le cadre du cours **GIF-4101** à l'Université Laval (Automne 2025).
    """)
    
    st.markdown("## 👨‍💻 Créateur Principal")
    st.markdown("### Salem N. Nyisingize")
    st.markdown("**MobileNetV3-Small** - 97.79% test accuracy")
    
    st.markdown("## 🤝 Membres de l'Équipe")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Félix Légaré")
        st.markdown("**ResNet18** - 208 FPS! 🚀")
    
    with col2:
        st.markdown("### Rayan Nadeau")
        st.markdown("**EfficientNet-B0** - 98.06% val acc! 🏆")

if __name__ == "__main__":
    main()