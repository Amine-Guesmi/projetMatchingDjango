from matchingImagesProject.models import Image
from django.shortcuts import render
from django.http import Http404
import os
import cv2 as cv
import numpy as np
from pathlib import Path
import fnmatch
import math
from itertools import islice

base_dir = "images/searchImages/"

def existe_img_base(img):
    my_file = Path(img)
    if my_file.is_file():
        return True
    else:
        return False

def construction_histogram(img, echelle_reduction):
    k = [echelle_reduction] * 3
    image = cv.imread(img)
    # Convertir de BGR en RGB
    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    histo = cv.calcHist([img_rgb], [0, 1, 2], None,k , [0, 256, 0, 256, 0, 256])
    #histo_normalise = cv.normalize(histo, histo).flatten()
    return histo

def calcul_histogramme_base(base,image_requete,echelle_reduction):
    dic_histogramme = {}
    extensions = ['jpg','png','jpeg']
    test = False
    for path, _, files in os.walk(base):
        for file in files:
        #La deuxième condition permet d'exclure l'image de la requete dans le cacul de l'histogramme
          if ((fnmatch.fnmatch(file, f'*.{extensions[0]}')) and (file != image_requete)):
            fullname = os.path.join(path, file)
            histo = construction_histogram(fullname, echelle_reduction)
            dic_histogramme[file]=histo
            test = True 
          elif not test:
            if ((fnmatch.fnmatch(file, f'*.{extensions[1]}')) and (file != image_requete)):
              fullname = os.path.join(path, file)
              histo = construction_histogram(fullname, echelle_reduction)
              dic_histogramme[file]=histo
          else:
            if ((fnmatch.fnmatch(file, f'*.{extensions[2]}')) and (file != image_requete)):
              fullname = os.path.join(path, file)
              histo = construction_histogram(fullname, echelle_reduction)
              dic_histogramme[file]=histo

            
    return dic_histogramme

def calcul_histogramme_img_requete(img_requete, echelle_reduction):
    if existe_img_base(img_requete):
        histo = construction_histogram(img_requete, echelle_reduction)
        return histo
    else:
        print("Image inexistante")

def calcul_distance_histogramme(histo_img_req, histo_bases):
    dic_distance = {}
    for i, j in histo_bases.items():
        distance = cv.compareHist(j,histo_img_req, cv.HISTCMP_CHISQR)
        dic_distance[i]=distance
    return dic_distance

def calcul_moment_hu(img):
    im = cv.imread(img)
    # Convertir l'image en Gray
    img_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    # Filtrer l'image pour reduire les bruits:  
    img_gray = cv.blur(img_gray, (3,3))
    ''' Binarisation de l'image à l'aide du seuillage
    _,img_bin = cv.threshold(img_gray, 128, 255, cv.THRESH_BINARY) '''  
    moments = cv.moments(img_gray)
    # Calculate Hu Moments
    huMoments = cv.HuMoments(moments)
    # Log scale hu moments
    for i in range(0,7): 
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
    return huMoments

def calcul_moment_hu_base(base,image_requete):
    dic_moment_hu = {}
    extensions = ['jpg','png','jpeg']
    test = False
    for path, _, files in os.walk(base):
        for file in files:
          if ((fnmatch.fnmatch(file, f'*.{extensions[0]}')) and (file != image_requete)):
            fullname = os.path.join(path, file)
            moment_hu = calcul_moment_hu(fullname)
            dic_moment_hu[file]=moment_hu
            test = True
          elif not test:
            if ((fnmatch.fnmatch(file, f'*.{extensions[1]}')) and (file != image_requete)):
              fullname = os.path.join(path, file)
              moment_hu = calcul_moment_hu(fullname)
              dic_moment_hu[file]=moment_hu
          else:
            if ((fnmatch.fnmatch(file, f'*.{extensions[2]}')) and (file != image_requete)):
              fullname = os.path.join(path, file)
              moment_hu = calcul_moment_hu(fullname)
              dic_moment_hu[file]=moment_hu
   
    return dic_moment_hu

def calcul_distance_euclidienne(a,b):
    dic_distance = {}
    print(len(a))
    for i, j in a.items():
        dist = np.linalg.norm(j - b)
        dic_distance[i]=dist
    return(dic_distance)

def calcul_similarite(dis_histo, dis_eclu, w1, w2):
    dic_similarite = {}
    for i, j in dis_histo.items():
        for k, l in dis_eclu.items():
            if (i == k):
                tot = (j*w1) + (l*w2)
                dic_similarite[i]=tot
                dic_similarite_trie = {keys: val for keys, val in sorted(dic_similarite.items(), key=lambda item: item[1])}
    return(dic_similarite_trie)

def get_k_plus_proche(k, dic_similarite):
    return list(islice(dic_similarite, k))


def calcul_moment_hu_img_requete(img_requete):
    if existe_img_base(img_requete):
        moment_hu = calcul_moment_hu(img_requete)
        return moment_hu
    else:
        print("Image inexistante")
        
def image(request, file_id):
    image = Image.objects.get(pk=file_id)
    if(image is not None):
        return render(request, 'files/file.html', {'file' : image})
    else:
        raise Http404('File deos not exist')

def application(request):
    imagesDataSet = os.listdir("images/searchImages") 
    histogramme_bases = calcul_histogramme_base(base_dir, '5.png', 32)
    histogramme_img_req = calcul_histogramme_img_requete('images/searchImages/5.png',32)
    distantce_histo = calcul_distance_histogramme(histogramme_img_req,histogramme_bases)
    hu_moment_base = calcul_moment_hu_base(base_dir, '5.png')
    hu_moment_img_req = calcul_moment_hu_img_requete(base_dir+'5.png')
    distance_ecludienne = calcul_distance_euclidienne(hu_moment_base, hu_moment_img_req)
    imgResult = get_k_plus_proche(100, calcul_similarite(distantce_histo, distance_ecludienne,0.5,0.5).items())
    resultList = []
    for element in imgResult:
        resultList.append(list(element)[0])
    return render(request, 'index.html', {"images" : resultList})
