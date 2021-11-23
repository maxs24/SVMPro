import numpy as np
import pandas as pd
import sklearn.svm as svm


def create_dataset(points):
    x = []
    y = []
    target = []
    line = generate_line()
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
        target.append(is_class(x[i], y[i], line))
    df_x = pd.DataFrame(data=x)
    df_y = pd.DataFrame(data=y)
    df_target = pd.DataFrame(data=target)
    data_frame = pd.concat([df_x, df_y], ignore_index=True, axis=1)
    data_frame = pd.concat([data_frame, df_target], ignore_index=True, axis=1)
    data_frame.columns = ['x', 'y', 'target']
    return data_frame


def generate_line():
    B = np.random.randint(1, 100)
    C = np.random.randint(-200 * B, 200 * B) * (-1)
    A = np.random.randint(int(0.5 * B), int(1.5 * B)) * (-1)
    return (A, B, C)


def is_class(x, y, line):
    value = line[0] * x + line[1] * y + line[2]
    if value > 0:
        return 1
    else:
        return 0


def random_points(size):
    x = np.random.randint(5, 595, size)
    y = np.random.randint(5, 395, size)
    points = []
    colors = []
    for j in range(size):
        points.append([x[j], y[j]])
        colors.append((0, 0, 0))
    return [points, colors]


def svm_fit(points, size):
    model = svm.SVC(kernel='linear')
    dataset = create_dataset(points)
    feauters = dataset[['x', 'y']]
    label = dataset['target']

    model.fit(feauters, label)

    return model

def predict_array(model, points, colors):
    x = []
    y = []
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
    df_x = pd.DataFrame(data=x)
    df_y = pd.DataFrame(data=y)
    data_frame = pd.concat([df_x, df_y], ignore_index=True, axis=1)
    data_frame.columns = ['x', 'y']
    W = np.asarray(svm.SVC.coef_)
    answers = model.predict(data_frame)
    for i in range(len(answers)):
        if answers[i] == 0:
            colors[i] = (255,0,0)
        else:
            colors[i] = (0, 255, 0)


#def return_line(model, points, size):









