import face_recognition as fr
import cv2
import numpy as np

image1 = fr.load_image_file('img.jpg')
sabte_sorat = fr.face_encodings(image1)[0]

image2 = fr.load_image_file('img2.jpg')
sabte_sorat2 = fr.face_encodings(image2)[0]

imagetest = fr.load_image_file('images.jpg')
imagetest_encode = fr.face_encodings(imagetest)[0]

result = fr.compare_faces([sabte_sorat, sabte_sorat2], imagetest_encode)
print(result)

image2_show = cv2.cvtColor(imagetest, cv2.COLOR_RGB2BGR)
locate = fr.face_locations(imagetest)

for (y1, x2, y2, x1) in locate:
    cv2.rectangle(image2_show, (x1, y1), (x2, y2), (255, 0, 0), 6)

image2_show = cv2.resize(image2_show, (299, 199))
cv2.imshow('Static Image', image2_show)
cv2.waitKey(0)
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(rgb_frame)

    for (y1, x2, y2, x1) in face_locations:
        encodes = fr.face_encodings(rgb_frame, [(y1, x2, y2, x1)])
        if len(encodes) == 0:
            continue
        encode = encodes[0]
        result = fr.compare_faces([sabte_sorat, sabte_sorat2], encode)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 6)

        if result[0]:
            name = "Rozita"
        elif result[1]:
            name = "XXX"
        else:
            name = "***"
        cv2.putText(frame, name, (x1 + 6, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Live Video', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
