import os
import cv2
from reportlab.pdfgen import canvas

moyage = 30.69
Genre = {'Homme': 53.7, 'Femme': 46.2}


def resize_image(photo, dim):
    image = cv2.imread(photo, 1)
    img = cv2.resize(image, dim)
    cv2.imwrite(str('resized_') + str(photo), img)


def cartouchef(Y, photo, statistiques, pdf):
    pdf.line(0, Y, 600, Y)
    pdf.line(0, Y + 180, 600, Y + 180)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawInlineImage(photo, 20, Y + 10)
    pdf.drawString(110, Y + 10, "Le pourcentage de femmes à l'écran est de:")
    pdf.drawString(517, Y + 10, str(statistiques))
    pdf.drawString(560, Y + 10, "%")


def cartoucheh(Y, photo, statistiques, pdf):
    pdf.line(0, Y, 600, Y)
    pdf.line(0, Y + 180, 600, Y + 180)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawInlineImage(photo, 20, Y + 10)
    pdf.drawString(110, Y + 10, "Le pourcentage d'hommes à l'écran est de:")
    pdf.drawString(517, Y + 10, str(statistiques))
    pdf.drawString(560, Y + 10, "%")

if os.path.exists("Résultat_diversité_.pdf"):
    os.remove("Résultat_diversité_.pdf")
pdf = canvas.Canvas("Résultat_diversité_.pdf")
pdf.setFont("Helvetica-Bold", 15)
pdf.drawCentredString(300, 780,
                      "Statistiques sur le genre des acteurs dan le but d'analyser")
pdf.drawCentredString(300, 750,"le respect de la diversité des acteurs")
pdf.drawCentredString(300, 720, "Le film annalysé est: La La land")
cartouchef(500, 'femme.jpg', Genre['Femme'], pdf)
cartoucheh(300, 'homme.jpg', Genre['Homme'], pdf)
pdf.drawString(20, 100,"L'analyse de l'age des acteurs nous indique que les visages que nous voyons")
pdf.drawCentredString(300, 80,"sont en moyenne agés de: " + str(moyage) + "ans")
pdf.save()

