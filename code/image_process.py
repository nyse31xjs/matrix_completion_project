from PIL import Image
import numpy as np

# Charger l'image
image_path = 'assets/images/IMG_BW.png'
image = Image.open(image_path)

# On reconverti en noir et blanc pour s'assurer de n'avoir qu'une valeur par pixel
image_gray = image.convert('L')

# Convertir l'image en une matrice numpy
matrix = np.array(image_gray,dtype=float)
print(matrix.shape)

image_deconstructed = Image.fromarray(matrix)

# Afficher l'image déconstruite
image_deconstructed.show()

######

######

import utils
import loss_functions as lf
import robust_matrix_completion as rmc

n1, n2, r = 320, 240, 120     # Matrix dimensions and rank
# c = 0.9                      # Outlier percentage
# SNR = 90
p = 0.10                     # Sampling percentage

# # M, U_true, V_true = utils.createMatrix(n1, n2, r)
# # contaminate M
# M_cont, outlier_locations = utils.contaminate_SNR(matrix, c, SNR, sigma_ratio = np.sqrt(1000))
# #print RMSE after contamination
# print('RMSE after contamination :', np.sqrt(np.sum((matrix-M_cont)**2)*(1/(n1*n2))))

# image_contaminated = Image.fromarray(M_cont)

# # Afficher l'image reconstruite
# image_contaminated.show()

# Obtain a sample from M_cont
Omega, data = utils.randomSample(matrix, p)

image_contaminated = Image.fromarray(data)

# Afficher l'image reconstruite
image_contaminated.show()

# #loss_functions = [lf.PseudoHuber(), lf.Huber(), lf.LeastSquares()]
loss_functions = [lf.PseudoHuber()]

for loss_fun in loss_functions:
    # estimate M
    M_est, U, V = rmc.complete_matrix(data, Omega, r, loss_fun=loss_fun)
    
    # uncomment next line to use gradient descent rather than joint regression and scale estimation
    #M_est, U, V = rmc.complete_matrix_GD(data, Omega, r, loss_fun=loss_fun)
    print(M_est)
    RMSE = np.sqrt(np.sum((matrix-M_est)**2)*(1/(n1*n2)))
            
    print('RMSE,', loss_fun.name(),': ', RMSE)

    image_reconstructed = Image.fromarray(M_est)

    # Afficher l'image reconstruite
    image_reconstructed.show()