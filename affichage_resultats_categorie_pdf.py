import os
import cv2
from reportlab.pdfgen import canvas

dictEmo = {'happy': 44.31,
           'surprised': 0,
           'sad': 42.7,
           'neutral': 11.84,
           'angry': 1.15
           }
theme = 'joyeux,comique'


# Pour changer la taille d'une image (dim est de la forme dim=(largeur,hauteur))
def resize_image(photo, dim):
    image = cv2.imread(photo, 1)
    img = cv2.resize(image, dim)
    cv2.imwrite(str('resized_') + str(photo), img)


def convert(dic):
    L = []
    for label in dic:
        L.append([label, dic[label]])
    return L


L = convert(dictEmo)


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


# Fonction transformant liste de [nom de célébrité, temps d'écran] en  pdf, en ayant dans le même dossier des photos de taille 80x80 leur correspondant, femaleest le pourcentage de femmes à l'écran et ethnic le pourcentage de minorités ethniques à l'écran

def resultat(liste, female, ethnic, theme):
    if os.path.exists("Résultat.pdf"):
        os.remove("Résultat.pdf")
    pdf1 = canvas.Canvas("Résultat.pdf")
    pdf1.setFont("Helvetica-Bold", 30)
    pdf1.drawCentredString(300, 800, "Respect de la diversité")
    n = 1
    for i in liste:
        s = i[0].split()
        photo = name = str('resized_') + str(s[0]) + str('_') + str(s[1]) + str('.jpg')
        cartouche(780 - n * 100, i[0], photo, i[1], pdf1)
        n += 1
    pdf1.setFont("Courier-Bold", 13)
    pdf1.drawString(20, 730 - (n - 1) * 100, "Pourcentage de femmes à l'écran :")
    pdf1.drawString(150, 710 - (n - 1) * 100, female)
    pdf1.drawString(305, 730 - (n - 1) * 100, "Pourcentage de minorités à l'écran :")
    pdf1.drawString(450, 710 - (n - 1) * 100, ethnic)
    pdf1.line(300, 780 - n * 100, 300, 780 - (n - 1) * 100)
    pdf1.line(0, 780 - n * 100, 600, 780 - n * 100)
    pdf1.setFont("Courier-Bold", 13)
    pdf1.drawString(0, 730 - n * 100,
                    "L'analyse des émotions des acteurs nous indique que le genre est : " + str(theme))
    pdf1.save()


if os.path.exists("Résultat_LA_La_Land.pdf"):
    os.remove("Résultat_LA_La_Land.pdf")
pdf = canvas.Canvas("Résultat_LA_La_Land.pdf")
pdf.setFont("Helvetica-Bold", 18)
pdf.drawCentredString(300, 800, "Statistiques sur le type de film")
pdf.drawCentredString(300, 720, "Le film annalysé est: La La Land")
cartouche(600, 'joyeux', 'resized_happy.jpg', dictEmo['happy'], pdf)
cartouche(500, 'triste', 'resized_sad.jpg', dictEmo['sad'], pdf)
cartouche(400, 'surprenant', 'resized_surprised.jpg', dictEmo['surprised'], pdf)
cartouche(300, 'neutre', 'resized_neutral.jpg', dictEmo['neutral'], pdf)
cartouche(200, 'colérique','resized_angry.jpg', dictEmo['angry'], pdf)
pdf.drawString(20, 100,"L'analyse des émotions des acteurs nous indique que le genre est : " + str(theme))
pdf.save()
