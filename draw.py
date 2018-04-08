import pygame
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle

digit_classifier = pickle.load(open("models/demo_network.pickle", "rb"))

pygame.init()

# display window of the program
screen = pygame.display.set_mode((400,400))
done = False
draw = False

clock = pygame.time.Clock()

# colours
BLACK = (0,0,0)
WHITE = (255,255,255)
GREY = (180,180,180)
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
clear_text = font.render("C = clear", True, BLACK)
send_text = font.render("Enter = read", True, BLACK)


clear = False

# fill the screen white much like ms paint and add other
#  ui stuff like instruction of using the program
def setup_screen():
    screen.fill(WHITE)
    pygame.draw.rect(screen,GREY,pygame.Rect(0,0,120,400))
    pygame.draw.rect(screen,GREY,pygame.Rect(0,280,400,120))
    pygame.draw.rect(screen,WHITE,pygame.Rect(240,293,148,97))
    pygame.draw.rect(screen,GREY_BORDER,pygame.Rect(120,280,4,120))
    screen.blit(reading_text,(133,310))
    screen.blit(draw_text,(5,20))
    screen.blit(erase_text,(5,50))
    screen.blit(clear_text,(5,80))
    screen.blit(send_text,(5,110))
    # is there anything that was read when user hits enter?
    if text_read is not None:
        font = pygame.font.Font(None, 60)
        respond_text = font.render(str(text_read), True, BLACK)
        screen.blit(respond_text,(300,326))


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
        elif event.type == pygame.MOUSEBUTTONUP and event.button == LEFT:
            draw = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == RIGHT:
            color = WHITE
            brush_size = 12
            draw = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == RIGHT:
            draw = False

        # did user hit a key?
        elif event.type == pygame.KEYDOWN:
            # if enter was pressed
            if event.key == pygame.K_RETURN:
                # once user decide the disired number to enter
                # this will take and scale the image
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
                
                delx = math.floor(y_moment/mass) # rounding
                dely = math.floor(x_moment/mass) # rounding

                # axis 0 = translation for y and axis 1 = translation in x
                # translating the image to center it
                vector = np.roll(vector,int(14 - dely), axis=0)
                vector = np.roll(vector,int(14 - delx), axis=1)

                # ------------------------placeholder-------------------
                # this reprints the scaled image if needed

                # drawing = pygame.surfarray.make_surface(vector)
                # drawing = pygame.transform.rotate(drawing,-90)
                # drawing = pygame.transform.flip(drawing,1,0)
                # drawing = pygame.transform.scale(drawing,(280,280))
                # screen.blit(drawing,(120,0))
                # pygame.display.flip()
                # pygame.time.wait(2000)
                # -------------------------

                # checking if the image was process correctly

                #plt.imshow(vector,interpolation = 'none')
                #plt.set_cmap('binary')
                #plt.show()

                onehot = digit_classifier.feedforward(vector.reshape(784, 1))
                print(np.argmax(onehot))
                               

                # once program reads what user draws
                # send the value back to them
                # text_read =

                clear = True
            # if c was pressed
            elif event.key == pygame.K_c:
                text_read = ""
                clear = True

        if draw == True:
            # is the mouse in the area where it can draw?
            # if so then draw or erase
            if mouse_pos[0] > 131 and mouse_pos[1] < 269:
                pygame.draw.circle(screen,color,mouse_pos,brush_size)
        # refresh the screen
        if clear == True:
            setup_screen()
            clear = False
    pygame.display.flip()
    clock.tick(300)
