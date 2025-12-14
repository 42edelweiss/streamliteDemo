import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import qrcode
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="OpenSpot - Détection de Places de Stationnement",
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

def main():
    # Sidebar
    st.sidebar.title("🅿️ OpenSpot")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Accueil", "🔍 Prédiction", "📊 Performance", "📈 Statistiques"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### À propos")
    st.sidebar.info(
        "**Projet GIF-4101**\n\n"
        "Détection automatique d'occupation de places de stationnement "
        "utilisant l'apprentissage profond.\n\n"
        "**Université Laval - Automne 2024**"
    )
    
    # QR Code dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📱 Accès Rapide")
    
    # URL de ton app déployée (modifie selon ton URL réelle)
    app_url = "https://openspotsfr.streamlit.app"
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
    elif page == "📊 Performance":
        show_performance()
    elif page == "📈 Statistiques":
        show_statistics()

def show_home():
    """Page d'accueil"""
    st.title("🅿️ OpenSpot - Détection de Places de Stationnement")
    
    st.markdown("""
    ## Bienvenue sur OpenSpot!
    
    OpenSpot est un système de détection automatique d'occupation de places de stationnement 
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
    
    **PKLot Dataset** - Ensemble complet d'images de stationnement avec:
    - Images de multiples stationnements
    - Différentes conditions météorologiques
    - 2 classes: Libre et Occupé
    - Plus de 12,000 images annotées
    
    ### 🚀 Comment Utiliser
    
    1. **Prédiction** - Uploadez une image pour détecter si une place est libre ou occupée
    2. **Performance** - Consultez les métriques détaillées du modèle
    3. **Statistiques** - Explorez les données du dataset
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
    
    # Image d'exemple
    st.markdown("### 📸 Exemple de Détection")
    st.info("Uploadez une image de place de stationnement dans l'onglet **Prédiction** pour tester le modèle!")
    
    st.markdown("---")
    
    # Section déploiement
    with st.expander("🌐 Déployer cette app publiquement"):
        st.markdown("""
        Pour rendre cette app accessible à tous via QR code:
        
        **1. Créer un compte sur Streamlit Cloud**
        - Aller sur [share.streamlit.io](https://share.streamlit.io)
        - Se connecter avec GitHub
        
        **2. Pousser le code sur GitHub**
        ```bash
        git add .
        git commit -m "Add Streamlit app"
        git push
        ```
        
        **3. Déployer**
        - Sélectionner le repo GitHub
        - Spécifier le fichier: `app.py`
        - Cliquer "Deploy"
        
        **4. Obtenir l'URL publique**
        - Exemple: `https://openspotsfr.streamlit.app`
        - Générer un QR code avec cette URL
        
        L'app sera accessible partout dans le monde! 🌍
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
    ### Dataset PKLot
    
    Statistiques détaillées sur l'ensemble de données utilisé pour l'entraînement 
    et l'évaluation du modèle.
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
        - Ensoleillé, nuageux, pluvieux
        - Multiples stationnements (UFPR, PUC)
        - Vue aérienne
        - Différents moments de la journée
        """)

if __name__ == "__main__":
    main()