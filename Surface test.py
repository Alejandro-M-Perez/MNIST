#Use a MNIST model to take 28x28 pixel greyscale images and classify the symbol in the image and output it

import pygame
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.models as models
from torchvision import datasets


# initialize a model with the same architecture as the model which parameters you saved into the .pt/h file
model = models.MNIST() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
model.eval()


# setup pygame
pygame.init()
clock = pygame.time.Clock()

ScreenSize = (330, 280)
screen = pygame.display.set_mode(ScreenSize)
screen.fill((0, 0, 0))

#create toolbar with buttons for clearing and submitting the canvas
toolbar = pygame.Surface((50, 280))
toolbar.fill((200, 200, 200))

#clear button
clear_button = pygame.Rect(0, 0, 50, 25)
pygame.draw.rect(toolbar, (200, 0, 0), clear_button)
font = pygame.font.Font(None, 16)
text_clear = font.render("Clear", True, (255, 255, 255))

#submit button
submit_button = pygame.Rect(0, 25, 50, 25)
pygame.draw.rect(toolbar, (0, 200, 0), submit_button)
text_sub = font.render("Submit", True, (255, 255, 255))


#create canvas
canvas = pygame.Surface((280, 280))
canvas.fill((255, 255, 255))

#set up the brush
brush_radius = 10
brush = pygame.Surface((brush_radius, brush_radius), pygame.SRCALPHA)

#create ghost brush courser that follows the mouse
# new type, "color" cursor
cursor = pygame.cursors.diamond
pygame.mouse.set_cursor(cursor)

#export surface
export_surface = pygame.Surface((28, 28))



#draw loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
            canvas.blit(brush, event.pos)
            pygame.draw.circle(canvas, (0, 0, 0, 100), (event.pos), brush_radius)
        if event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
            canvas.blit(brush, event.pos)
            pygame.draw.circle(canvas, (0, 0, 0, 100), (event.pos), brush_radius)


        if event.type == pygame.MOUSEBUTTONDOWN and clear_button.collidepoint(event.pos):
            screen.blit(toolbar, (0, 0))
            screen.blit(canvas, (50, 0))
            canvas.fill((255, 255, 255))
            pygame.draw.rect(toolbar, (0, 200, 0), submit_button)
            pygame.draw.rect(toolbar, (200, 0, 0), clear_button)

        if event.type == pygame.MOUSEBUTTONDOWN and submit_button.collidepoint(event.pos):
            screen.blit(toolbar, (0, 0))
            screen.blit(canvas, (50, 0))
            pygame.draw.rect(toolbar, (0, 200, 0), submit_button)
            pygame.draw.rect(toolbar, (200, 0, 0), clear_button)
            pygame.transform.scale(canvas, (28, 28), export_surface)
            pygame.image.save(export_surface, "Sample.png")
            print("Image saved")
            canvas.fill((255, 255, 255))
            #add the image to the dataset
                

        screen.blit(toolbar, (0, 0))
        screen.blit(canvas, (50, 0))


        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                drawing = False
    
    pygame.display.flip()
    clock.tick(60)