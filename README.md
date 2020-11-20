# Emotions_Detection_on_videos
This python code is able to detect in real time on a video the emotion of the deteced people.  This is my project during the coding weeks at Centralesupelec

## Prédiction du "genre" ou ton émotionel du film par analyse des sentiments, et calcul de statistiques (temps à l'écran, diversité, parité)

Groupe 15 constitué de : 
- Thomas Zonabend
- Vincent Michelangeli 
- Armand Morin
- Baptiste Carbillet
- Théodore de Pomereu

# Descriptif global du produit et objectifs : 
- Développer une application qui permet de déterminer la durée totale de présence d'un acteur/personnage dans un film avec des outils de traçage vidéo et de détection de visage. 
- Calculer un certain nombre de statistiques : évaluer la diversité sociale/parité dans un film, etc.
- Déterminer le genre et la tendance du film en évaluant les émotions des acteurs. 

# Sprint 0: Mise en place des outils pour le projet de semaine 2
- Fonctionnalité 1 : création d’un dépôt Gitlab et clonage en local par chaque membre du Groupe
- Fonctionnalité 2a : Identification du MVP et découpage du projet en différents sprints
- Fonctionnalité 2b : Identification des principaux et différents utilisateurs de mon produit ainsi que des principaux besoins de ces utilisateurs.
- Fonctionnalité 3 : Le découpage du projet est rédigé et partagé sur le dépôt. Les différents rôles et tâches ont été distiribués au sein du groupe. Mise en place de branches pour chaque nouvelle fonctionnalité
# Sprint 1: Fonctions de base de modifications d’images, transformation d’une vidéo en liste de frames
- Fonctionnalité 1 : fonctions de bases pour images 
- Fonctionnalité 2 : chargement des données relatives au film (liste des personnages)
- Fonctionnalité 3 : transformation de vidéo en liste de frames
# Sprint 2: Reconnaissance des visages dans une vidéo, calcul du nombre de frames dans lequel un personnage est présent
- Fonctionnalité 1a : Détection de l'émotion d'une personne sur une image
- Fonctionnalité 1b : Détection du genre, l'origine (utiliser un modèle distinct pour entrainer et détecter chaque caractéristique)
- Fonctionnalité 2 : calcul de statistiques
# Sprint 3: Implémentation du module détection d’émotion pour calculer le nombre de frames par type d’émotion
- Fonctionnalité 1 : Détection d'émotion sur une image
- Fonctionnalité 2 : Détection du ton du film à travers les émotions
# Sprint 4: Mise en forme du programme final
- Fonctionnalité 1 : Mise en commun des différents travaux
- Fonctionnalité 2 : Presentation des resultats sous forme de pdf regroupant toutes les infos du film
- Fonctionnalité 3 : Préparation d'une demo et commentaires sur le code


Pour le MVP on commence par analyser des bandes annonces au lieu de film car ils sont trop longs à analyser. 

# Modules à installer :
- tensorflow, keras et cv2
- request
- reportlab (pour générer des pdf)
- selenium
- PIL
- hashlib
- io

# Utilisation:
Pour lancer le programme on exécute interface.py . Puis une fenêtre s'ouvre et on entre le nom du film que l'on veut analyser. Un pdf est ensuite chargé affichant les données après un temps d'analyse.

# Results preview:

![BAscreen](https://user-images.githubusercontent.com/72650161/99811035-e2562900-2b44-11eb-8dd3-7bc81c75940f.png)

![La La Land screen](https://user-images.githubusercontent.com/72650161/99811069-eeda8180-2b44-11eb-8b77-1794127af83e.png)

[Résultat_LA_La_Land.pdf](https://github.com/Armand-Morin/Emotions_Detection_on_videos/files/5574091/Resultat_LA_La_Land.pdf)
