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

# Import du modèle
try:
    from mobilenet import MobileNetV3Classifier
except ImportError:
    st.error("Erreur: Impossible d'importer MobileNetV3Classifier. Vérifiez que le fichier mobilenet.py existe.")

# Chemin du modèle (CORRIGÉ avec underscore)
MODEL_PATH = "best_mobilenet.pt"

# Métriques du modèle (MobileNetV3-Small)
MODEL_METRICS = {
    "accuracy": 97.79,
    "precision": 97.63,
    "recall": 97.80,
    "f1_score": 97.71,
    "inference_time": 17.94,
    "fps": 56,
    "model_size": 2.54
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
def load_model():
    """Charge le modèle MobileNetV3"""
    try:
        model = MobileNetV3Classifier(num_classes=2, pretrained=False, version='small')
        
        # Charger les poids
        if Path(MODEL_PATH).exists():
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            
            # Gérer différents formats de checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            return model
        else:
            st.error(f"Checkpoint non trouvé: {MODEL_PATH}")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

def predict_image(model, image):
    """Fait une prédiction sur l'image"""
    try:
        # Prétraiter l'image
        img_tensor = transform(image).unsqueeze(0)
        
        # Prédiction
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
    """
    Trouve les ROIs pour une image donnée dans le fichier d'annotations
    """
    try:
        # Supporter plusieurs formats de JSON
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
        
        # Trouver l'index de l'image
        if image_name in file_names:
            index = file_names.index(image_name)
            return rois_list[index]
        
        return None
    
    except (KeyError, IndexError):
        return None

def analyze_parking_with_rois(image, rois, model):
    """
    Analyse chaque place de parking définie par les ROIs
    """
    results = []
    img_width, img_height = image.size
    
    for roi in rois:
        # Convertir les coordonnées normalisées en pixels
        points = []
        for point in roi:
            x = int(point[0] * img_width)
            y = int(point[1] * img_height)
            points.append((x, y))
        
        # Extraire la région (bounding box du ROI)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Crop la région
        try:
            cropped = image.crop((x_min, y_min, x_max, y_max))
            
            # Faire la prédiction
            predicted_class, confidence, _ = predict_image(model, cropped)
            
            results.append({
                'prediction': predicted_class,
                'confidence': confidence,
                'points': points
            })
        except Exception as e:
            # Si erreur, marquer comme inconnu
            results.append({
                'prediction': None,
                'confidence': 0,
                'points': points,
                'error': str(e)
            })
    
    return results

def display_annotated_results(image, rois, results, show_labels, show_confidence, 
                              line_width, font_size):
    """
    Affiche l'image avec les prédictions sur chaque place
    """
    # Créer une copie de l'image
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image, 'RGBA')
    
    # Charger une police (ou utiliser par défaut)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Statistiques
    num_libre = sum(1 for r in results if r['prediction'] == 0)
    num_occupe = sum(1 for r in results if r['prediction'] == 1)
    num_error = sum(1 for r in results if r['prediction'] is None)
    
    # Dessiner chaque ROI avec sa prédiction
    for i, result in enumerate(results):
        points = result['points']
        prediction = result['prediction']
        confidence = result['confidence']
        
        # Choisir la couleur selon la prédiction
        if prediction == 0:  # Libre
            color = (0, 255, 0, 100)  # Vert semi-transparent
            outline_color = (0, 200, 0, 255)  # Vert foncé
            label = "Libre"
        elif prediction == 1:  # Occupé
            color = (255, 0, 0, 100)  # Rouge semi-transparent
            outline_color = (200, 0, 0, 255)  # Rouge foncé
            label = "Occupé"
        else:  # Erreur
            color = (128, 128, 128, 100)  # Gris
            outline_color = (100, 100, 100, 255)
            label = "Erreur"
        
        # Dessiner le polygone rempli
        draw.polygon(points, fill=color, outline=outline_color, width=line_width)
        
        # Ajouter le numéro de place
        if show_labels or show_confidence:
            # Calculer le centre du ROI
            center_x = sum(p[0] for p in points) // 4
            center_y = sum(p[1] for p in points) // 4
            
            text_parts = []
            if show_labels:
                text_parts.append(f"#{i+1}")
            if show_confidence and prediction is not None:
                text_parts.append(f"{confidence:.0f}%")
            
            text = " ".join(text_parts)
            
            # Dessiner le texte avec fond
            try:
                bbox = draw.textbbox((center_x, center_y), text, font=font)
                draw.rectangle(bbox, fill=(0, 0, 0, 180))
                draw.text((center_x, center_y), text, fill='white', font=font, anchor='mm')
            except:
                draw.text((center_x, center_y), text, fill='white')
    
    # Afficher l'image annotée
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(annotated_image, caption=f"Parking analysé - {len(results)} places", 
                use_container_width=True)
    
    with col2:
        # Statistiques
        st.markdown("### 📊 Résumé")
        
        st.metric("Places Totales", len(results))
        st.metric("🟢 Libres", f"{num_libre} ({num_libre/len(results)*100:.1f}%)")
        st.metric("🔴 Occupées", f"{num_occupe} ({num_occupe/len(results)*100:.1f}%)")
        
        if num_error > 0:
            st.metric("⚠️ Erreurs", num_error)
        
        # Barre de progression
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
        
        # Confidence moyenne
        valid_confidences = [r['confidence'] for r in results if r['prediction'] is not None]
        if valid_confidences:
            avg_confidence = sum(valid_confidences) / len(valid_confidences)
            st.metric("Confiance Moyenne", f"{avg_confidence:.1f}%")
    
    # Détails par place (optionnel)
    with st.expander("📋 Détails par Place"):
        for i, result in enumerate(results):
            if result['prediction'] is not None:
                label = "🟢 Libre" if result['prediction'] == 0 else "🔴 Occupé"
                st.write(f"**Place {i+1}:** {label} - Confiance: {result['confidence']:.1f}%")
            else:
                st.write(f"**Place {i+1}:** ⚠️ Erreur - {result.get('error', 'Inconnu')}")
    
    # Téléchargement
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        # Télécharger l'image annotée
        buf = BytesIO()
        annotated_image.save(buf, format="PNG")
        
        st.download_button(
            label="💾 Télécharger l'Image Annotée",
            data=buf.getvalue(),
            file_name=f"parking_annotated_{len(results)}_places.png",
            mime="image/png"
        )
    
    with col_dl2:
        # Télécharger le rapport texte
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

def main():
    # Sidebar
    st.sidebar.title("🅿️ FindSpot")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Accueil", "🔍 Prédiction", "🅿️ Avec Annotations", "📊 Performance", "📈 Statistiques", "👥 À propos"]
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
    
    # URL de ton app déployée
    app_url = "https://findspot.streamlit.app"
    qr_img = generate_qr_code(app_url)
    
    # Convertir en format affichable
    buf = BytesIO()
    qr_img.save(buf, format="PNG")
    st.sidebar.image(buf.getvalue(), caption="Scannez pour accéder à l'app", use_container_width=True)
    st.sidebar.caption(f"URL: {app_url}")
    
    # Pages
    if page == "🏠 Accueil":
        show_home()
    elif page == "🔍 Prédiction":
        show_prediction()
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
    2. **Avec Annotations** - Analysez un parking complet avec visualisation de chaque place
    3. **Performance** - Consultez les métriques détaillées du modèle
    4. **Statistiques** - Explorez les données du dataset
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Analyse Place par Place")
        st.success("Upload une image → Prédiction immédiate")
        st.info("Idéal pour: Vérification rapide d'une place")
    
    with col2:
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
        - ML: PyTorch + MobileNetV3 + EfficientNet
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
            model = load_model()
        
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
                        st.write(f"**Précision:** {MODEL_METRICS['accuracy']:.2f}%")
                        st.write(f"**F1-Score:** {MODEL_METRICS['f1_score']:.2f}%")
                    with info_col2:
                        st.write(f"**Temps d'inférence:** {MODEL_METRICS['inference_time']:.2f} ms")
                        st.write(f"**FPS:** {MODEL_METRICS['fps']}")
                        st.write(f"**Taille:** {MODEL_METRICS['model_size']:.2f} MB")
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
    """Page de prédiction avec annotations (visualisation complète du parking)"""
    st.title("🅿️ Analyse avec Annotations")
    
    st.markdown("""
    Uploadez une image de parking et son fichier d'annotations pour visualiser 
    la prédiction sur **chaque place individuelle**.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Upload image
        uploaded_image = st.file_uploader(
            "📸 Image de parking",
            type=["jpg", "jpeg", "png"],
            key="annotated_image"
        )
    
    with col2:
        # Upload annotations JSON
        uploaded_json = st.file_uploader(
            "📄 Fichier annotations.json",
            type=["json"],
            key="annotations_file"
        )
    
    if uploaded_image is not None and uploaded_json is not None:
        # Charger l'image
        image = Image.open(uploaded_image).convert('RGB')
        image_name = uploaded_image.name
        
        # Charger les annotations
        try:
            annotations = json.load(uploaded_json)
            
            # Trouver les ROIs pour cette image
            rois = find_rois_for_image(image_name, annotations)
            
            if rois is None:
                st.error(f"❌ Image '{image_name}' non trouvée dans le fichier d'annotations!")
                st.info("💡 Vérifiez que le nom de l'image correspond exactement à un nom dans 'file_names'")
                return
            
            st.success(f"✅ {len(rois)} places détectées dans l'image!")
            
            # Options de visualisation
            st.markdown("---")
            col_opts1, col_opts2 = st.columns(2)
            
            with col_opts1:
                show_labels = st.checkbox("Afficher les numéros de places", value=True)
                show_confidence = st.checkbox("Afficher la confiance", value=True)
            
            with col_opts2:
                line_width = st.slider("Épaisseur des contours", 1, 10, 3)
                font_size = st.slider("Taille du texte", 10, 40, 20)
            
            # Bouton d'analyse
            if st.button("🔍 Analyser toutes les places", type="primary"):
                with st.spinner(f"Analyse de {len(rois)} places en cours..."):
                    # Charger le modèle
                    model = load_model()
                    
                    if model is not None:
                        # Analyser toutes les places
                        results = analyze_parking_with_rois(image, rois, model)
                        
                        # Afficher les résultats
                        display_annotated_results(image, rois, results, show_labels, 
                                                 show_confidence, line_width, font_size)
        
        except json.JSONDecodeError:
            st.error("❌ Erreur: Le fichier JSON est invalide!")
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
    
    else:
        # Instructions
        st.info("👆 Uploadez une image ET son fichier d'annotations pour commencer")
        
        with st.expander("💡 Format du fichier JSON"):
            st.code("""
{
  "train": {
    "file_names": ["GOPR0025.JPG", "GOPR0027.JPG", ...],
    "rois_list": [
      [  // ROIs pour GOPR0025.JPG
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  // Place 1
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  // Place 2
        ...
      ],
      ...
    ]
  }
}
            """, language="json")
            
            st.markdown("""
            **Note:** Les coordonnées sont normalisées (0-1)
            - x = position horizontale / largeur de l'image
            - y = position verticale / hauteur de l'image
            """)

def show_performance():
    """Page de performance du modèle"""
    st.title("📊 Performance du Modèle")
    
    st.markdown("""
    Métriques détaillées de performance du modèle MobileNetV3-Small sur le dataset PKLot.
    """)
    
    # Métriques principales
    st.markdown("### 🎯 Métriques de Classification")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Précision (Accuracy)", f"{MODEL_METRICS['accuracy']:.2f}%")
    with col2:
        st.metric("Précision (Precision)", f"{MODEL_METRICS['precision']:.2f}%")
    with col3:
        st.metric("Rappel (Recall)", f"{MODEL_METRICS['recall']:.2f}%")
    with col4:
        st.metric("Score F1", f"{MODEL_METRICS['f1_score']:.2f}%")
    
    # Métriques par classe
    st.markdown("### 📋 Performance par Classe")
    
    class_metrics = {
        "Classe": ["Libre", "Occupé"],
        "Précision (%)": [98.52, 96.73],
        "Rappel (%)": [97.74, 97.85],
        "F1-Score (%)": [98.13, 97.29],
        "Support": [885, 605]
    }
    
    st.dataframe(class_metrics, use_container_width=True)
    
    # Visualisation
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique des métriques par classe
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(class_metrics["Classe"]))
        width = 0.25
        
        ax.bar(x - width, class_metrics["Précision (%)"], width, label='Précision', alpha=0.8)
        ax.bar(x, class_metrics["Rappel (%)"], width, label='Rappel', alpha=0.8)
        ax.bar(x + width, class_metrics["F1-Score (%)"], width, label='F1-Score', alpha=0.8)
        
        ax.set_ylabel('Pourcentage (%)')
        ax.set_title('Métriques par Classe')
        ax.set_xticks(x)
        ax.set_xticklabels(class_metrics["Classe"])
        ax.legend()
        ax.set_ylim(95, 100)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Matrice de confusion
        st.markdown("#### Matrice de Confusion")
        confusion_matrix = np.array([[865, 20], [13, 592]])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Vraie Classe')
        ax.set_title('Matrice de Confusion')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Métriques de vitesse
    st.markdown("### ⚡ Performance d'Inférence")
    
    speed_col1, speed_col2, speed_col3 = st.columns(3)
    
    with speed_col1:
        st.metric("Temps Moyen", f"{MODEL_METRICS['inference_time']:.2f} ms")
    with speed_col2:
        st.metric("FPS", MODEL_METRICS['fps'])
    with speed_col3:
        st.metric("Taille du Modèle", f"{MODEL_METRICS['model_size']:.2f} MB")
    
    # Graphique de distribution du temps
    st.markdown("#### Distribution du Temps d'Inférence")
    
    # Données d'exemple (basées sur tes benchmarks)
    inference_times = [17.94, 4.92, 13.26, 72.15, 16.75]  # Mean, Std, Min, Max, Median
    labels = ['Moyenne', 'Écart-type', 'Minimum', 'Maximum', 'Médiane']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors_perf = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        ax.bar(labels, inference_times, color=colors_perf, alpha=0.7)
        ax.set_ylabel('Temps (ms)')
        ax.set_title('Statistiques du Temps d\'Inférence')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("##### Statistiques")
        st.write(f"**Moyenne:** {inference_times[0]:.2f} ms")
        st.write(f"**Écart-type:** {inference_times[1]:.2f} ms")
        st.write(f"**Minimum:** {inference_times[2]:.2f} ms")
        st.write(f"**Maximum:** {inference_times[3]:.2f} ms")
        st.write(f"**Médiane:** {inference_times[4]:.2f} ms")

def show_statistics():
    """Page de statistiques du dataset"""
    st.title("📈 Statistiques du Dataset")
    
    st.markdown("""
    ### Dataset Action-Camera Parking (GoPro Hero 6)
    
    Statistiques détaillées sur l'ensemble de données utilisé pour l'entraînement 
    et l'évaluation du modèle. Images capturées à ~10 mètres de hauteur avec GoPro Hero 6.
    """)
    
    # Statistiques générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Images Test", "1,490")
    with col2:
        st.metric("Classes", "2")
    with col3:
        st.metric("Précision", "97.79%")
    with col4:
        st.metric("Correct", "1,457")
    
    # Distribution des classes
    st.markdown("### 📊 Distribution des Classes (Test Set)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig, ax = plt.subplots(figsize=(7, 7))
        sizes = [885, 605]  # Libre, Occupé
        labels_pie = ['Libre', 'Occupé']
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)
        
        ax.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie,
               autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 14})
        ax.set_title('Répartition des Classes', fontsize=16, fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        # Bar chart
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.bar(labels_pie, sizes, color=colors_pie, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Nombre d\'Images', fontsize=12)
        ax.set_title('Distribution des Classes', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(sizes):
            ax.text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Résultats de prédiction
    st.markdown("### ✅ Résultats des Prédictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**Prédictions Correctes:** 1,457 / 1,490")
        st.info(f"**Taux de Réussite:** {(1457/1490)*100:.2f}%")
    
    with col2:
        st.error("**Prédictions Incorrectes:** 33 / 1,490")
        st.warning(f"**Taux d'Erreur:** {(33/1490)*100:.2f}%")
    
    # Analyse des erreurs
    st.markdown("### 🔍 Analyse des Erreurs")
    
    error_data = {
        "Type d'Erreur": ["Faux Positifs (Libre)", "Faux Négatifs (Occupé)"],
        "Nombre": [20, 13],
        "Description": [
            "Places libres classées comme occupées",
            "Places occupées classées comme libres"
        ]
    }
    
    st.dataframe(error_data, use_container_width=True)
    
    # Informations additionnelles
    st.markdown("---")
    st.markdown("### 📝 Informations Additionnelles sur le Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Caractéristiques des Images:**
        - Résolution: 224x224 pixels (redimensionnées)
        - Format: RGB
        - Normalisation: ImageNet standards
        - Augmentation: rotation, flip, color jitter
        """)
    
    with col2:
        st.markdown("""
        **Conditions de Capture:**
        - Caméra: GoPro Hero 6
        - Hauteur: ~10 mètres
        - Vue aérienne de parkings
        - Annotations: Régions d'intérêt (ROIs)
        - Format: Coordonnées normalisées (0-1)
        """)

def show_about_team():
    """Page À propos de l'équipe"""
    st.title("👥 À propos de FindSpot")
    
    st.markdown("""
    ## 🎯 Le Projet
    
    **FindSpot** est un système intelligent de détection d'occupation de places de stationnement 
    développé dans le cadre du cours **GIF-4101 - Introduction à l'Apprentissage Automatique** 
    à l'Université Laval (Automne 2025).
    
    Le projet utilise des techniques d'apprentissage profond pour analyser des images de parkings 
    et détecter automatiquement si les places sont libres ou occupées, avec une précision de plus de 97%.
    """)
    
    st.markdown("---")
    
    # Créateur Principal
    st.markdown("## 👨‍💻 Créateur Principal")
    
    col_creator = st.columns([1, 3])
    
    with col_creator[0]:
        st.markdown("### Salem N. Nyisingize")
        st.markdown("**@42edelweiss**")
    
    with col_creator[1]:
        st.markdown("""
        **Rôle:** Architecte principal & Développeur
        
        **Contributions:**
        - 🏗️ Architecture du modèle MobileNetV3-Small
        - 💻 Développement de l'application Streamlit
        - 📊 Pipeline d'entraînement et d'évaluation
        - 🎨 Interface utilisateur et visualisations
        - 🚀 Déploiement et optimisation
        
        **Modèle:** MobileNetV3-Small (97.79% accuracy)
        """)
        
        st.metric("Test Accuracy", "97.79%", "+0.5%")
        st.metric("Model Size", "2.54 MB", "Léger")
        st.metric("Inference Speed", "56 FPS", "Rapide")
    
    st.markdown("---")
    
    # Membres de l'équipe
    st.markdown("## 🤝 Membres de l'Équipe")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Félix Légaré")
        st.markdown("**@flegare07**")
        st.markdown("""
        **Contribution:** Modèle ResNet18
        
        **Résultats:**
        - Test Accuracy: **94.97%**
        - Validation Accuracy: **95.85%**
        - Best Epoch: 9
        - **Le plus rapide:** 208 FPS! 🚀
        """)
        
        with st.expander("📊 Métriques ResNet18"):
            st.markdown("### Performance")
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Test Acc", "94.97%")
            with col_res2:
                st.metric("Val Acc", "95.85%")
            with col_res3:
                st.metric("Best Epoch", "9")
            
            st.markdown("### Vitesse d'Inférence ⚡")
            col_speed1, col_speed2, col_speed3 = st.columns(3)
            with col_speed1:
                st.metric("Temps Moyen", "4.81 ms", "🔥 Le plus rapide!")
            with col_speed2:
                st.metric("FPS", "208.07", "🚀 Record!")
            with col_speed3:
                st.metric("Médiane", "4.71 ms")
            
            st.success("⚡ ResNet18 est le modèle le PLUS RAPIDE avec 208 FPS!")
            
            st.markdown("### Modèle")
            col_model1, col_model2, col_model3 = st.columns(3)
            with col_model1:
                st.metric("Taille", "42.71 MB")
            with col_model2:
                st.metric("Paramètres", "11.18M")
            with col_model3:
                st.metric("Device", "CUDA")
            
            st.markdown("### Détails Vitesse")
            st.write(f"**Écart-type:** 0.37 ms")
            st.write(f"**Min:** 4.33 ms")
            st.write(f"**Max:** 7.68 ms")
            
            st.markdown("### Matrice de Confusion")
            confusion_res = np.array([[855, 30], [45, 560]])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion_res, annot=True, fmt='d', cmap='Greens',
                       xticklabels=["Libre", "Occupé"], 
                       yticklabels=["Libre", "Occupé"], ax=ax)
            ax.set_xlabel('Prédiction')
            ax.set_ylabel('Vraie Classe')
            ax.set_title('ResNet18 - Matrice de Confusion')
            st.pyplot(fig)
            
            st.markdown("""
            **Statistiques:**
            - Total d'échantillons: 1,490
            - Prédictions correctes: 1,415
            - Prédictions incorrectes: 75
            - Faux positifs: 30
            - Faux négatifs: 45
            """)
    
    with col2:
        st.markdown("### Rayan Nadeau")
        st.markdown("**GameScopeX5**")
        st.markdown("""
        **Contribution:** Modèle EfficientNet-B0
        
        **Résultats:**
        - Test Accuracy: **96.98%**
        - Validation Accuracy: **98.06%** (La meilleure!)
        - Best Epoch: 5
        """)
        
        with st.expander("📊 Métriques EfficientNet-B0"):
            st.markdown("### Performance")
            col_eff1, col_eff2, col_eff3 = st.columns(3)
            with col_eff1:
                st.metric("Test Acc", "96.98%")
            with col_eff2:
                st.metric("Val Acc", "98.06%", "🏆 La meilleure!")
            with col_eff3:
                st.metric("Best Epoch", "5")
            
            st.success("🏆 EfficientNet a la MEILLEURE validation accuracy!")
            
            st.markdown("### Vitesse d'Inférence")
            col_speed1, col_speed2, col_speed3 = st.columns(3)
            with col_speed1:
                st.metric("Temps Moyen", "27.37 ms")
            with col_speed2:
                st.metric("FPS", "36.53")
            with col_speed3:
                st.metric("Médiane", "25.03 ms")
            
            st.markdown("### Modèle")
            col_model1, col_model2 = st.columns(2)
            with col_model1:
                st.metric("Taille", "15.59 MB")
            with col_model2:
                st.metric("Paramètres", "4.01M")
            
            st.markdown("### Matrice de Confusion")
            confusion_eff = np.array([[867, 18], [27, 578]])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion_eff, annot=True, fmt='d', cmap='Blues',
                       xticklabels=["Libre", "Occupé"], 
                       yticklabels=["Libre", "Occupé"], ax=ax)
            ax.set_xlabel('Prédiction')
            ax.set_ylabel('Vraie Classe')
            ax.set_title('EfficientNet-B0 - Matrice de Confusion')
            st.pyplot(fig)
            
            st.markdown("""
            **Statistiques:**
            - Total d'échantillons: 1,490
            - Prédictions correctes: 1,445
            - Prédictions incorrectes: 45
            - Faux positifs: 18
            - Faux négatifs: 27
            """)
    
    st.markdown("---")
    
    # Comparaison des modèles
    st.markdown("## 📊 Comparaison des Modèles")
    
    comparison_data = {
        "Modèle": ["MobileNetV3-Small", "EfficientNet-B0", "ResNet18"],
        "Test Accuracy (%)": [97.79, 96.98, 94.97],
        "Val Accuracy (%)": [97.85, 98.06, 95.85],
        "Taille (MB)": [2.54, 15.59, 42.71],
        "FPS": [56, 36.53, 208.07],
        "Temps (ms)": [17.94, 27.37, 4.81],
        "Paramètres (M)": [1.52, 4.01, 11.18]
    }
    
    st.dataframe(comparison_data, use_container_width=True)
    
    # Points forts de chaque modèle
    st.markdown("### 🏆 Points Forts")
    
    col_strong1, col_strong2, col_strong3 = st.columns(3)
    
    with col_strong1:
        st.success("**MobileNetV3-Small**")
        st.write("🏆 Meilleur test accuracy (97.79%)")
        st.write("⚡ Le plus léger (2.54 MB)")
        st.write("📱 Idéal pour mobile")
    
    with col_strong2:
        st.info("**EfficientNet-B0**")
        st.write("🏆 Meilleure val accuracy (98.06%)")
        st.write("⚖️ Bon équilibre taille/perf")
        st.write("🎯 Moins d'erreurs (45)")
    
    with col_strong3:
        st.warning("**ResNet18**")
        st.write("🏆 LE PLUS RAPIDE (208 FPS!)")
        st.write("⚡ Seulement 4.81 ms/image")
        st.write("🚀 Idéal pour temps réel")
    
    # Graphique de comparaison
    st.markdown("### 📈 Comparaison Visuelle")
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        # Accuracy comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        models = ["MobileNetV3", "EfficientNet-B0", "ResNet18"]
        test_acc = [97.79, 96.98, 94.97]
        val_acc = [97.85, 98.06, 95.85]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, test_acc, width, label='Test Accuracy', alpha=0.8, color=['#2ecc71', '#3498db', '#e74c3c'])
        ax.bar(x + width/2, val_acc, width, label='Validation Accuracy', alpha=0.8, color=['#27ae60', '#2980b9', '#c0392b'])
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Comparaison des Précisions')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.legend()
        ax.set_ylim(93, 99)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_comp2:
        # Speed vs Size comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sizes = [2.54, 15.59, 42.71]
        fps = [56, 36.53, 208.07]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        labels = ["MobileNetV3", "EfficientNet", "ResNet18"]
        
        scatter = ax.scatter(sizes, fps, s=[800, 800, 800], c=colors, alpha=0.6, edgecolors='black', linewidth=2)
        
        for i, model in enumerate(labels):
            offset_x = 3 if i == 2 else 1
            offset_y = 15 if i == 2 else 5
            ax.annotate(model, (sizes[i], fps[i]), 
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
        
        ax.set_xlabel('Taille du Modèle (MB)')
        ax.set_ylabel('Vitesse (FPS)')
        ax.set_title('Trade-off Taille vs Vitesse')
        ax.grid(True, alpha=0.3)
        
        # Annotations spéciales
        ax.annotate('Le plus rapide!\n208 FPS 🚀', xy=(42.71, 208.07), 
                   xytext=(30, 180), fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.annotate('Le plus léger!\n2.54 MB 📱', xy=(2.54, 56), 
                   xytext=(8, 80), fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Analyse comparative finale
    st.markdown("## 🎯 Analyse Comparative")
    
    st.markdown("""
    ### Quel modèle choisir selon l'application?
    
    Notre comparaison de 3 architectures révèle des trade-offs intéressants:
    """)
    
    col_use1, col_use2, col_use3 = st.columns(3)
    
    with col_use1:
        st.markdown("#### 📱 Application Mobile")
        st.success("**Gagnant: MobileNetV3**")
        st.markdown("""
        **Pourquoi?**
        - Seulement 2.54 MB
        - 97.79% accuracy
        - 56 FPS suffisant
        - Conçu pour mobile
        
        **Idéal pour:**
        - Apps iOS/Android
        - Appareils contraints
        - Déploiement edge
        """)
    
    with col_use2:
        st.markdown("#### ⚡ Temps Réel Critique")
        st.warning("**Gagnant: ResNet18**")
        st.markdown("""
        **Pourquoi?**
        - 208 FPS incroyable!
        - 4.81 ms par image
        - Performance GPU
        
        **Idéal pour:**
        - Systèmes embarqués
        - Traitement vidéo
        - Surveillance temps réel
        - Avec GPU disponible
        """)
    
    with col_use3:
        st.markdown("#### 🎯 Précision Maximale")
        st.info("**Gagnant: EfficientNet**")
        st.markdown("""
        **Pourquoi?**
        - 98.06% val accuracy
        - Seulement 45 erreurs
        - Bon équilibre
        
        **Idéal pour:**
        - Applications critiques
        - Validation nécessaire
        - Cloud deployment
        - Moins d'erreurs critiques
        """)
    
    st.markdown("---")
    
    st.markdown("### 💡 Recommandations Finales")
    
    rec_col1, rec_col2 = st.columns([2, 1])
    
    with rec_col1:
        st.markdown("""
        **Pour FindSpot (cette application):**
        
        Nous avons choisi **MobileNetV3-Small** comme modèle principal car:
        
        1. ✅ **Meilleur test accuracy (97.79%)** - Performance réelle optimale
        2. ✅ **Le plus léger (2.54 MB)** - Déploiement facile sur Streamlit Cloud
        3. ✅ **Vitesse suffisante (56 FPS)** - Largement assez pour notre usage
        4. ✅ **Accessible partout** - Fonctionne même sur appareils limités
        5. ✅ **Trade-off optimal** - Meilleur équilibre pour une web app
        
        **ResNet18** serait meilleur pour un système avec GPU dédié.
        
        **EfficientNet** serait meilleur si la précision maximale était critique.
        """)
    
    with rec_col2:
        st.markdown("#### 📊 Résumé")
        st.metric("Modèles testés", "3")
        st.metric("Gagnant test acc", "MobileNetV3")
        st.metric("Gagnant val acc", "EfficientNet")  
        st.metric("Gagnant vitesse", "ResNet18")
        st.metric("Choix déployé", "MobileNetV3")
    
    st.markdown("---")
    
    # Technologies utilisées
    st.markdown("## 🛠️ Technologies Utilisées")
    
    col_tech1, col_tech2, col_tech3 = st.columns(3)
    
    with col_tech1:
        st.markdown("""
        **Machine Learning:**
        - PyTorch
        - torchvision
        - MobileNetV3
        - EfficientNet
        - ResNet
        """)
    
    with col_tech2:
        st.markdown("""
        **Visualisation:**
        - Streamlit
        - Matplotlib
        - Seaborn
        - PIL/Pillow
        """)
    
    with col_tech3:
        st.markdown("""
        **Dataset:**
        - Action-Camera Parking Dataset
        - GoPro Hero 6
        - 293 images annotées
        - Vue aérienne (~10m)
        - Annotations ROI (JSON)
        """)
    
    st.markdown("---")
    
    # Contact et liens
    st.markdown("## 📞 Contact & Liens")
    
    col_contact1, col_contact2 = st.columns(2)
    
    with col_contact1:
        st.markdown("""
        **GitHub du Projet:**
        - Repository: [flegare07/GIF-4101](https://github.com/flegare07/GIF-4101)
        - Créateur: [@42edelweiss](https://github.com/42edelweiss)
        
        **Application:**
        - URL: https://findspot.streamlit.app
        - Déployé sur: Streamlit Cloud
        """)
    
    with col_contact2:
        st.markdown("""
        **Cours:**
        - GIF-4101 - Introduction à l'Apprentissage Automatique
        - Université Laval
        - Automne 2025
        
        **Dataset:**
        - Action-Camera Parking Dataset
        - Source: [Martin Marek (2021)](https://github.com/martin-marek/parking-space-occupancy)
        - arXiv:2107.12207
        
        **Remerciements:**
        - Professeur et assistants du cours
        - Martin Marek (dataset creator)
        - Communauté Streamlit
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f0f0f0; border-radius: 10px;'>
        <h3>🅿️ FindSpot</h3>
        <p><strong>Développé avec ❤️ par Salem N. Nyisingize et l'équipe</strong></p>
        <p>GIF-4101 | Université Laval | Automne 2025</p>
        <p style='font-size: 0.9em; color: gray;'>
            Powered by PyTorch • Streamlit • MobileNetV3 • EfficientNet
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()