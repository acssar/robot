import cv2
import numpy as np

SIZE = (533, 300)  # 16:9 соотношение сторон
KP = 0.32  # 0.22 0.32
KD = 0.18

last = 0
RECT = np.float32([[0, SIZE[1]],
                   [SIZE[0], SIZE[1]],
                   [SIZE[0], 0],
                   [0, 0]])

TRAP = np.float32([[10, 299],
                   [523, 299],
                   [440, 200],
                   [93, 200]])

src_draw = np.array(TRAP, dtype=np.int32)  # нужен int, чтобы отрисовалось

cap = cv2.VideoCapture(0)
key = 1
ESCAPE = 27


while key != ESCAPE:
    status, frame = cap.read()
    if status:
        cv2.imshow("Frame", frame)

        img = cv2.resize(frame, SIZE)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.inRange(gray, 120, 255)  # 130
        cv2.imshow("binary", binary)

        matrix_trans = cv2.getPerspectiveTransform(TRAP, RECT)
        perspective = cv2.warpPerspective(binary, matrix_trans, SIZE,
                                          flags=cv2.INTER_LINEAR)  # изобр, с помощью чего, к какому размеру и способ расчета
        trap_visual = cv2.drawContours(binary.copy(), [src_draw], -1, 150,
                                       thickness=3)  # src_draw - trap в другом формате
        cv2.imshow("trap_vis", trap_visual)

        cv2.imshow("perspective", perspective)

        hist = np.sum(perspective, axis=0)  # суммируем по оси 0
        mid = hist.shape[0] // 2  # ищем середину
        left = np.argmax(hist[:mid])  # индекс макс эл-та в массиве
        right = np.argmax(hist[mid:]) + mid  # отсчёт от центра

        cv2.line(perspective, (left, 0 ), (left, SIZE[1]), 50, 2)
        cv2.line(perspective, (right, 0), (right, SIZE[1]), 50, 2)
        cv2.line(perspective, ((left + right) // 2, 0), ((left + right) // 2, SIZE[1]), 110, 3)
        cv2.imshow("lines", perspective)

        # может быть перепутано лево и право, тогда надо на -1 домножить, проверить
        err = 0 - ((left + right) // 2 - SIZE[0] // 2)  # left+right/2 - центр дороги, size[0]/2 - центр картинки
        # print((left + right) // 2)
        # print((SIZE[0] // 2))
        # хотим получить угол для поворота, если влево - немного убрать, если вправо = немного добавить
        # a = 90 - err * KP  KP -пропорциональный регулятор, чем больше ошибка, тем больше изменений в сигнал
        # коэфф подбирается экспириментально, KD - дифференциальный коэффициент (скорость изм. ошибки)
        angle = int(90 + KP * err + KD * (err - last))  # разница между ошибкой, насколько изм., чем больше изм ошибки
        print(angle)
        # тем быстрее реагируем
        last = err

        if angle < 72:  # подобрать, чтоб не сожгло серво
            angle = 72
        elif angle > 108:
            angle = 108

        # control(pi, ESC, 1550, STEER, angle)
    else:
        print("end of video")
        break

    key = cv2.waitKey(10)

cv2.destroyAllWindows()

# сначало пишем на видео, потом предлагаем внедрить управление, что было, читать с камеры машинки
# подобрать угол камеры - ставим шнур в ноут и смотрим


# упрощение, тем кто работал с ардуино
# можно считать сумму под собой, если врезался в линию, то поверни - примитивный алгоритм
# улучшения:
# можно искать центр масс прерывистой полосы ( левой)
# можно видео академии высоких технологий
