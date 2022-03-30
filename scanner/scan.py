from imutils.perspective import four_point_transform
import cv2

green = (0, 255, 0)

image = cv2.imread("img.jpeg")
# image = cv2.resize(image, (width, height))
orig_image = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert the image to gray scale
blur = cv2.GaussianBlur(gray, (5, 5), 0) # Add Gaussian blur
edged = cv2.Canny(blur, 75, 200) # Apply the Canny algorithm to find the edges

contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Show the image and all the contours
cv2.imshow("Image", image)
cv2.drawContours(image, contours, -1, green, 3)
cv2.imshow("All contours", image)
# Show the image and the edges
cv2.imshow('Original image:', image)
cv2.imshow('Edged:', edged)

for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
    if len(approx) == 4:
        doc_cnts = approx
        break
cv2.drawContours(orig_image, [doc_cnts], -1, green, 3)
cv2.imshow("Contours of the document", orig_image)
cv2.imwrite('outputs/contours.png', orig_image)
warped = four_point_transform(orig_image, doc_cnts.reshape(4, 2))
output = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


ret, output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU, dst=output)

cv2.imwrite('outputs/output.png', output)
cv2.imshow("Scanned", output)

cv2.waitKey(0)