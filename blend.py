# import os
# imgpath = 'datasets/Ship/JPEGImages_test_BBox_GT'
# txtsavepath = 'datasets/Ship'
#
#
# imgtxt = open('datasets/Ship/test.txt', 'w')
#
# for img in os.listdir(imgpath ):
#     name = img[0:-4] + '\n'
#     imgtxt.write(name)
#
# imgtxt.close()


from PIL import Image
f = open ('datasets/Ship/test.txt', "r")
file= f.readlines()
for line in file:
 line = line.replace('\n','')
 img1=Image.open('datasets/Ship/JPEGImages_test_BBox_GT/'+str(line)+'.jpg')
 img2=Image.open('runs/detect/predict_LD/'+str(line)+'.jpg')
# img2.show()
 im=Image.blend(img1,img2,alpha=0.6)
 im.save('runs/detect/compare_LD_cwd/'+str(line)+'.jpg')
f. close ()