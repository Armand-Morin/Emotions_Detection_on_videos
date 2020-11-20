## Prédiction du "genre" ou plutôt de la tendance du film par analyse des sentiments et du temps de parole des comediens dans le film.

Groupe 15 constitué de: 
 - Thomas Zonabend
 - Vincent Michelangeli 
 - Armand Morin
  - Baptiste Carbillet
 - Théodore de Pomereu

Objectifs: 
- Développer une application qui permet de déterminer la durée totale de présence d'un acteur/personnage dans un film avec outils de traçage vidéo et détection de visage. 
- Calculer un certain nombre de statistiques: calculer salaire par minute d'un acteur, évaluer diversité sociale/parité dans un film, etc.
- Déterminer le genre du film en évaluant les émotions des personnages.

# Sprint 1: Fonctions de base de modifications d’images, transformation d’une vidéo en liste de frames
- Fonctionnalité 1: fonctions de bases pour images 
- Fonctionnalité 2: chargement des données relatif au film (liste des personnages)
- Fonctionnalité 3: transformation vidéo en liste de frames
# Sprint 2: Reconnaissance des visages dans une vidéo, calcul du nombre de frames dans lequel un personnage est présent
- Fonctionnalité 1: reconnaissance des visages dans les frames
- Fonctionnalité 2: calcul de statistiques
# Sprint 3: Implémentation du module détection d’émotion pour calculer le nombre de frames par type d’émotion
- Fonctionnalité 1a: Détection de l'émotion d'une personne sur une image
- Fonctionnalité 1b: Detection de la nationnalité, l'origine, l'age des comediens (utiliser un modèle distinct pour entrainer et détecter chaques caractéristiques)
- Fonctionnalité 2: Prédiction du genre du film
# Sprint 4: Mise en forme du programme final
On doit encore reflechir pour savoir sous quelle forme est ce que l'on present le resultat: tableau avec répartition des temps etc... 


Pour le MVP on commence par analyser des bandes annonces.