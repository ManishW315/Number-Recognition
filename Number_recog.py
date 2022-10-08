import numpy as np
from tensorflow import keras
import cv2
import pygame
import matplotlib.pyplot as plt
import tensorflow as tf
# Initializing pygame
pygame.init()

# Colors
black = (0, 0, 0)
white = (255, 255, 255)

# Creating game window
display_wind = pygame.display.set_mode((550, 550))
pygame.display.set_caption("Drawing Pad")
display_wind.fill(black)
font = pygame.font.SysFont("ebrima", 55)
draw = False

def text_screen(text, color, x, y):
    screen_text = font.render(text, True, color)
    display_wind.blit(screen_text, (x, y))
    

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_x:
            display_wind.fill(black)
        if event.type == pygame.MOUSEBUTTONDOWN:
            draw = True

        if event.type == pygame.MOUSEBUTTONUP:
            draw = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            string_image = pygame.image.tostring(display_wind, 'RGB')
            temp_surf = pygame.image.fromstring(string_image, (550, 550), 'RGB')
            tmp_arr = pygame.surfarray.array2d(temp_surf)
            pygame.image.save(display_wind, 'sample.jpg')
            # load json and create model
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = tf.keras.models.model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("Number_Recognition_Model.h5")
 
            # evaluate loaded model on test data
            loaded_model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            img = cv2.imread('sample.jpg')[:, :, 1]
            img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
            plt.imshow(img)
            img = np.invert(np.array([img]))
            img = np.invert(np.array([img]))
            img = img/255.
            print(img.shape)
            pred = loaded_model.predict(img[0])
            output = str(np.argmax(pred[0]))
            text_screen(output, white, 500, 20)
            
    if draw:
        pointer = pygame.mouse.get_pos()
        pygame.draw.rect(display_wind, white, pygame.Rect(pointer[0], pointer[1], 32, 32))

    pygame.display.update()
