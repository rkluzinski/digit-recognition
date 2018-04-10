import pygame
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle

digit_classifier = pickle.load(open("models/hn_80_9751.pickle", "rb"))

pygame.init()

# display window of the program
screen = pygame.display.set_mode((400,400))
done = False
draw = False
clear = False
process = False
image = False
drawing = pygame.Surface((0,0))
clock = pygame.time.Clock()

# colours
BLACK = (0,0,0)
WHITE = (255,255,255)
GREY = (180,180,180)
LIGHT_GREY = (210,210,210)
GREY_BORDER = (150,150,150)
color = ()

# mouse button keys when it is pressed down
LEFT = 1
MIDDLE = 2
RIGHT = 3
SCROLL_UP = 4
SCROLL_DOWN = 5

# font and text setup
font = pygame.font.Font(None, 45)
reading_text = font.render("I read:", True, BLACK)
font = pygame.font.Font(None, 60)

# left as an empty string so that it can be
# updated when user draws something
respond_text = font.render("", True, BLACK)
text_read = ""

#  text instruction
font = pygame.font.Font(None, 24)
erase_text = font.render("RClick = erase", True, BLACK)
draw_text = font.render("LClick = draw", True, BLACK)
font = pygame.font.Font(None,40)
clear_text = font.render("Clear", True, BLACK)
send_text = font.render("Read", True, BLACK)

# fill the screen white much like ms paint and add other
#  ui stuff like instruction of using the program
def setup_screen():
    screen.fill(WHITE)
    pygame.draw.rect(screen,GREY,pygame.Rect(0,0,120,400))
    pygame.draw.rect(screen,GREY,pygame.Rect(0,280,400,120))
    pygame.draw.rect(screen,WHITE,pygame.Rect(240,293,148,97))
    # buttons for text
    pygame.draw.rect(screen,GREY_BORDER,pygame.Rect(10,80,90,40))
    pygame.draw.rect(screen,LIGHT_GREY,pygame.Rect(14,84,82,32))
    pygame.draw.rect(screen,GREY_BORDER,pygame.Rect(10,140,90,40))
    pygame.draw.rect(screen,LIGHT_GREY,pygame.Rect(14,144,82,32))
    pygame.draw.rect(screen,GREY_BORDER,pygame.Rect(6,296,98,98))
    # border for the image processed
    pygame.draw.rect(screen,GREY_BORDER,pygame.Rect(120,280,4,120))
    # printing text
    screen.blit(reading_text,(133,310))
    screen.blit(draw_text,(5,20))
    screen.blit(erase_text,(5,50))
    screen.blit(clear_text,(18,86))
    screen.blit(send_text,(20,146))
    # is there anything that was read when user hits enter?
    if text_read is not None:
        font = pygame.font.Font(None, 60)
        respond_text = font.render(str(text_read), True, BLACK)
        screen.blit(respond_text,(300,326))

def process_image():
    save_window = pygame.Surface((280,280))
    save_window.blit(screen,(0,0),(120,0,400,280))
    save_window = pygame.transform.scale(save_window,(28,28))
    save_window = pygame.transform.rotate(save_window,-90)
    save_window = pygame.transform.flip(save_window,1,0)
    vector = pygame.surfarray.array2d(save_window)
    #print("read successfully")

    # black pixels are 1 and white are 0
    vector = vector / 16777215
    vector = np.ones([28, 28]) - vector

    # NOTE white pixel = 16777215 black pixel = 0
    # if user draw on the side of the screen center the drawing
    # by calculating center of mass
    x_moment = 0
    y_moment = 0
    mass = 0
    for i in range(28):
        for j in range(28):
            x_moment += i * vector[i][j]
            y_moment += j * vector[i][j]
            mass += vector[i][j]
    if mass == 0:
        return "?",False,None
    else:
        delx = math.floor(y_moment/mass) # rounding
        dely = math.floor(x_moment/mass) # rounding

    # axis 0 = translation for y and axis 1 = translation in x
    # translating the image to center it
        vector = np.roll(vector,int(14 - dely), axis=0)
        vector = np.roll(vector,int(14 - delx), axis=1)

    # reprints the image in a downscaled form to show user what the program reads
    drawing = vector
    drawing = (drawing * -16777215) + 16777215
    drawing = pygame.surfarray.make_surface(drawing)
    drawing = pygame.transform.rotate(drawing,-90)
    drawing = pygame.transform.flip(drawing,1,0)
    drawing = pygame.transform.scale(drawing,(90,90))

    # checking if the image was process correctly
    #plt.imshow(vector,interpolation = 'none')
    #plt.set_cmap('binary')
    #plt.show()

    onehot = digit_classifier.feedforward(vector.reshape(784, 1))
    # print(np.argmax(onehot))

    # once program reads what user draws
    # send the value back to the
    clear = True
    return np.argmax(onehot),True, drawing

# --------------- MAIN STARTS HERE -----------------
setup_screen()
while not done:
    for event in pygame.event.get():
        mouse_pos = pygame.mouse.get_pos()

        # program quit when user hits X on the window
        if event.type == pygame.QUIT:
            done = True

        # checking whether the mouse button is pressed
        #  left click is to draw and right click is to erase
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == LEFT:
            color = BLACK
            brush_size = 12
            draw = True
            if mouse_pos[1] < 120 and mouse_pos[1] > 80:
                if mouse_pos[0] >10 and mouse_pos[0] < 100:
                    clear = True
            if mouse_pos[1] > 140 and mouse_pos[1] < 180:
                if mouse_pos[0] >10 and mouse_pos[0] < 100:
                    process = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == LEFT:
            draw = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == RIGHT:
            color = WHITE
            brush_size = 12
            draw = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == RIGHT:
            draw = False
        if draw == True:
            # is the mouse in the area where it can draw?
            # if so then draw or erase
            if mouse_pos[0] > 131 and mouse_pos[0] < 389 and mouse_pos[1] < 269 and mouse_pos[1]:
                pygame.draw.circle(screen,color,mouse_pos,brush_size)
        # refresh the screen
        if clear == True:
            text_read = ""
            setup_screen()
            clear = False
            image = False
        # make the image for the program to read
        if process == True:
            text_read,image,drawing = process_image()
            setup_screen()
            process = False
        if image == True:
            screen.blit(drawing,(10,300))
        else:
            pygame.draw.rect(screen,WHITE,pygame.Rect(10,300,90,90))

    pygame.display.flip()
    clock.tick(300)
