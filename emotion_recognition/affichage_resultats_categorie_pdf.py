import os
import cv2
from reportlab.pdfgen import canvas

#### C'est un code que l'on peut mettre a la fin de emotion_recognition.py mais c'est pas
# conseillé car c'est deja trop lourd a lui tout seul


Emotions = {'happy': 27.1,
            'surprised': 1.7,
            'sad': 24.2,
            'neutral': 42.6,
            'angry': 4.1
            }


# fonction qui renvoie le max d'un dictionnaire
def max(dic):
    maxi = 0
    for x in dic:
        if dic[x] > maxi:
            maxi = x
    return x


theme = max(Emotions)


# Pour changer la taille d'une image (dim est de la forme dim=(largeur,hauteur))
def resize_image(photo, dim):
    image = cv2.imread(photo, 1)
    img = cv2.resize(image, dim)
    cv2.imwrite(str('resized_') + str(photo), img)


def cartouche(Y, nom, photo, statistiques, pdf):
    pdf.line(0, Y, 600, Y)
    pdf.line(0, Y + 100, 600, Y + 100)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawInlineImage(photo, 20, Y + 10)
    pdf.drawString(110, Y + 10, "Le film est :")
    pdf.drawString(200, Y + 10, nom)
    pdf.drawString(300, Y + 10, "avec probabilité de :")
    pdf.drawString(517, Y + 10, str(statistiques))
    pdf.drawString(560, Y + 10, "%")


if os.path.exists("Résultat.pdf"):
    os.remove("Résultat.pdf")
pdf = canvas.Canvas("Résultat.pdf")
pdf.setFont("Helvetica-Bold", 18)
pdf.drawCentredString(300, 800, "Statistiques sur le type de film")
pdf.drawCentredString(300, 720, "Le film annalysé est: Confinés")
cartouche(600, 'joyeux', 'resized_happy.jpg', Emotions['happy'], pdf)
cartouche(500, 'triste', 'resized_sad.jpg', Emotions['sad'], pdf)
cartouche(400, 'surprenant', 'resized_surprised.jpg', Emotions['surprised'], pdf)
cartouche(300, 'neutre', 'resized_neutral.jpg', Emotions['neutral'], pdf)
cartouche(200, 'colérique', 'resized_angry.jpg', Emotions['angry'], pdf)
pdf.drawString(20, 100, "L'analyse des émotions des acteurs nous indique que le genre est : " + str(theme))
pdf.save()
