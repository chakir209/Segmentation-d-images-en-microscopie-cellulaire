import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from darkflow.net.build import TFNet
import time
from utils import iou
from scipy import spatial
# Définir les options et le modèle Darkflow
options = {'model': 'cfg/tiny-yolo-voc-3c.cfg',
           'load': 3750,
           'threshold': 0.1,
           'gpu': 0.7}
tfnet = TFNet(options)

# Créer une fenêtre tkinter
root = tk.Tk()
root.title("Blood Cell Count")
bg_image = cv2.imread("data/background.jpg")
bg_image = Image.fromarray(cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB))
bg_image = bg_image.resize((800, 700))
bg_image = ImageTk.PhotoImage(bg_image)
# Créer un Label et l'affecter à la racine
bg_label = tk.Label(root, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Définir la taille de la fenêtre
root.geometry("800x700")
root.resizable(False, False)
scrollbar = tk.Scrollbar(root)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
# Définir une fonction pour charger une image
label2 = tk.Label(root, text=" Segmentation d’images en microscopie cellulaire:",font=("Arial", 21),highlightbackground="white")
label2.pack(pady=9)
inputtext="Ce projet utilise des algorithmes pour segmenter  \n et analyser des images de cellules en microscopie. \n En particulier, il permet de compter les globules de \n sang dans ces images,  ce qui peut être utile pour \n le diagnostic  de   certaines maladies.."
expli = tk.Label(root, text=inputtext,bg="white" , font=("Arial", 14))
expli.pack()



result_label = tk.Label(root,bg="white" )
label = tk.Label(root, font=("Arial", 13),bg="white")
def load_image():
    # Ouvrir une boîte de dialogue pour sélectionner un fichier
    file_path = filedialog.askopenfilename()
    print('image chemin ', file_path)
    # Vérifier si un fichier a été sélectionné
    if file_path:
        # Appeler la fonction pour compter les cellules sanguines
        image = blood_cell_count(file_path)
        result_label.config(image=image)
        result_label.image = image
        result_label.pack(pady=5)
        label.configure(text=texte)
    # Créer une étiquette pour afficher l'image
    label.pack()
pred_bb = []  
pred_cls = []  
pred_conf = []          
# Définir une fonction pour compter les cellules sanguines
def blood_cell_count(file_name):
    rbc = 0
    wbc = 0
    platelets = 0

    cell = []
    cls = []
    conf = []

    record = []
    tl_ = []
    br_ = []
    iou_ = []
    iou_value = 0

    tic = time.time()
    image = cv2.imread(file_name)
    output = tfnet.return_predict(image)

    for prediction in output:
        label = prediction['label']
        confidence = prediction['confidence']
        tl = (prediction['topleft']['x'], prediction['topleft']['y'])
        br = (prediction['bottomright']['x'], prediction['bottomright']['y'])

        if label == 'RBC' and confidence < .5:
            continue
        if label == 'WBC' and confidence < .25:
            continue
        if label == 'Platelets' and confidence < .25:
            continue

        # clearing up overlapped same platelets
        if label == 'Platelets':
            if record:
                tree = spatial.cKDTree(record)
                index = tree.query(tl)[1]
                iou_value = iou(tl + br, tl_[index] + br_[index])
                iou_.append(iou_value)

            if iou_value > 0.1:
                continue

            record.append(tl)
            tl_.append(tl)
            br_.append(br)

        center_x = int((tl[0] + br[0]) / 2)
        center_y = int((tl[1] + br[1]) / 2)
        center = (center_x, center_y)

        if label == 'RBC':
            color = (0, 0, 255)
            rbc = rbc + 1
        if label == 'WBC':
            color = (255, 255, 255)
            wbc = wbc + 1
        if label == 'Platelets':
            color = (0, 255, 0)
            platelets = platelets + 1

        radius = int((br[0] - tl[0]) / 2)
        image = cv2.circle(image, center, radius, color, 2)
        font = cv2.FONT_HERSHEY_COMPLEX
        image = cv2.putText(image, label, (center_x - 15, center_y + 5), font, .5, color, 1)
        cell.append([tl[0], tl[1], br[0], br[1]])

        if label == 'RBC':
            cls.append(0)
        if label == 'WBC':
            cls.append(1)
        if label == 'Platelets':
            cls.append(2)

        conf.append(confidence)

    toc = time.time()
    pred_bb.append(cell)
    pred_cls.append(cls)
    pred_conf.append(conf)
    avg_time = (toc - tic) * 1000
    print('{0:.5}'.format(avg_time), 'ms')

    cv2.imwrite('output/' + file_name, image)
    global texte
    texte = 'Cette image contient {} RBC, {} WBC et {} plaquettes'.format(rbc, wbc, platelets)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (5, 470)
    fontScale = 0.4
    color = (0, 0, 0)
    thickness = 2
    #image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    #cv2.imshow('cette image contient ==>'  + str(rbc) + ' RBC ,' + str(wbc) + ' WBC ,'+ str(platelets)+ '  plaquettes ' , image)
    print('Press "ESC" to close . . .')
    #cv2.imwrite('output/' + file_name, image)
    result_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result_image = result_image.resize((400, 350))
    result_image = ImageTk.PhotoImage(result_image)
    return result_image 
# Créer un bouton pour charger une image
load_button = tk.Button(root,  bg='blue',height="3",text="télécharger une image de sang", command=load_image,font=("Arial", 14))

load_button.pack(pady=5)

# Positionnement de l'étiquette dans la fenêtre
# Lancer la boucle principale de la fenêtre tkinter
root.mainloop()