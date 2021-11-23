import pygame
import Functions as F

if __name__ == '__main__':
    r = 4
    size = 100

    points, colors = F.random_points(size)
    model = F.svm_fit(points, size)

    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    pygame.display.update()
    play = True
    isdrawline = False
    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    points.append([event.pos[0], event.pos[1]])
                    colors.append((0, 0, 0))
                    F.predict_array(model, points, colors)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    F.predict_array(model, points, colors)
                    isdrawline = True
            screen.fill('WHITE')
            for i in range(len(points)):
                pygame.draw.circle(screen, colors[i], points[i], r)
            pygame.display.update()
