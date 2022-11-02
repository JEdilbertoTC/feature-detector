import os

import cv2

orb = cv2.ORB_create()
sift = cv2.SIFT_create()
hog = cv2.HOGDescriptor()


def load_files(path='train'):
    images = []
    class_names = []
    files = os.listdir(path)
    for file in files:
        image = cv2.imread(f'{path}/{file}', 0)
        image = cv2.resize(image, (400, 400))
        image = cv2.GaussianBlur(image, (3, 3), 0)
        images.append(image)
        class_names.append(os.path.splitext(file)[0])

    return images, class_names


def find_descriptors(images):
    descriptors = []
    for image in images:
        # key_point, descriptor = orb.detectAndCompute(image, None)
        # key_point, descriptor = sift.detectAndCompute(image, None)
        descriptor = hog.compute(image)
        descriptors.append(descriptor)
    return descriptors


def find_good_matches(matches):
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
    return len(good_matches)


def find_image(image, descriptors, threshold=30):
    """
    Change threshold for any descriptor \n
    Suggestions: \n
    orb = 15 \n
    sift = 30 \n
    hog =  \n
    """
    # key_point, query_descriptors = orb.detectAndCompute(image, None)
    # key_point, query_descriptors = sift.detectAndCompute(image, None)
    query_descriptors = hog.compute(image)
    bf = cv2.BFMatcher()
    match_list = []
    value = -1

    for train_descriptors in descriptors:
        # matches = bf.knnMatch(train_descriptors, query_descriptors, k=2)
        matches = bf.knnMatch(train_descriptors[:500], query_descriptors[:500], k=2)
        good_matches = find_good_matches(matches)
        match_list.append(good_matches)

    print(match_list)
    if len(match_list) != 0:
        if max(match_list) >= threshold:
            value = match_list.index(max(match_list))

    return value


def run():
    images, class_names = load_files()

    descriptors = find_descriptors(images)

    query = cv2.imread('query/call of duty.jpg')

    query = cv2.resize(query, (400, 400))

    original = query.copy()
    query = cv2.GaussianBlur(query, (3, 3), 0)
    query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    # query = cv2.rotate(query, cv2.ROTATE_90_COUNTERCLOCKWISE)

    value = find_image(query, descriptors)

    if value != -1:
        cv2.putText(original, class_names[value], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 5)

    cv2.imshow('query', query)
    cv2.imshow('original', original)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # video = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = video.read()
    #
    #     original = frame.copy()
    #
    #     frame = cv2.GaussianBlur(frame, (5, 5), 0)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     frame = cv2.resize(frame, (400, 400))
    #
    #     value = find_image(frame, descriptors)
    #
    #     if value != -1:
    #         cv2.putText(original, class_names[value], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    #
    #     cv2.imshow('video', original)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # video.release()
    # cv2.destroyAllWindows()
