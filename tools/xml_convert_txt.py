from random import sample


#40类的标签信息
# CLASSES = {
# 'applauding':0,
# 'blowing_bubbles':1,
# 'brushing_teeth':2,
# 'cleaning_the_floor':3,
# 'climbing':4,
# 'cooking':5,
# 'cutting_trees':6,
# 'cutting_vegetables':7,
# 'drinking':8,
# 'feeding_a_horse':9,
# 'fishing':10,
# 'fixing_a_bike':11,
# 'fixing_a_car':12,
# 'gardening':13,
# 'holding_an_umbrella':14,
# 'jumping':15,
# 'looking_through_a_microscope':16,
# 'looking_through_a_telescope':17,
# 'playing_guitar':18,
# 'playing_violin':19,
# 'pouring_liquid':20,
# 'pushing_a_cart':21,
# 'reading':22,
# 'phoning':23,
# 'riding_a_bike':24,
# 'riding_a_horse':25,
# 'rowing_a_boat':26,
# 'running':27,
# 'shooting_an_arrow':28,
# 'smoking':29,
# 'taking_photos':30,
# 'texting_message':31,
# 'throwing_frisby':32,
# 'using_a_computer':33,
# 'walking_the_dog':34,
# 'washing_dishes':35,
# 'watching_TV':36,
# 'waving_hands':37,
# 'writing_on_a_board':38,
# 'writing_on_a_book':39
# }

# a = [
# 'applauding',
# 'blowing_bubbles',
# 'brushing_teeth',
# 'cleaning_the_floor',
# 'climbing',
# 'cooking',
# 'cutting_trees',
# 'cutting_vegetables',
# 'drinking',
# 'feeding_a_horse',
# 'fishing',
# 'fixing_a_bike',
# 'fixing_a_car',
# 'gardening',
# 'holding_an_umbrella',
# 'jumping',
# 'looking_through_a_microscope',
# 'looking_through_a_telescope',
# 'playing_guitar',
# 'playing_violin',
# 'pouring_liquid',
# 'pushing_a_cart',
# 'reading',
# 'phoning',
# 'riding_a_bike',
# 'riding_a_horse',
# 'rowing_a_boat',
# 'running',
# 'shooting_an_arrow',
# 'smoking',
# 'taking_photos',
# 'texting_message',
# 'throwing_frisby',
# 'using_a_computer',
# 'walking_the_dog',
# 'washing_dishes',
# 'watching_TV',
# 'waving_hands',
# 'writing_on_a_board',
# 'writing_on_a_book'
# ]


#分18类的标签信息
# CLASSES = {
# 'applauding':0,
# 'blowing_bubbles':1,
# 'brushing_teeth':2,
# 'drinking':3,
# 'smoking':4,
# 'climbing':5,
# 'jumping':6,
# 'looking_through_a_microscope':7,
# 'reading':8,
# 'texting_message':9,
# 'using_a_computer':10,
# 'writing_on_a_book':11,
# 'playing_guitar':12,
# 'playing_violin':13,
# 'riding_a_bike':14,
# 'running':15,
# 'waving_hands':16,
# 'writing_on_a_board':17
# }

#self_waving_hands_label
# CLASSES = {
# 'waving_hands':0,
# 'using_a_computer':1,
# 'reading':2,
# 'drinking':3,
# 'smoking':4,
# 'running':5
# }

# Detection classfacation
CLASSES = {
'bottle':0,
'chair':1,
'person':2,
'potted_plant':3
}

def main():
    annotations = []


    # with open("../../Stanford40/ImageSplits/train.txt", "r") as f:
    #     train_file_list = f.readlines()
    # for i in range(len(train_file_list)):
    #     filename = train_file_list[i][:-9]
    #     gt_label = CLASSES[filename]
    #     gt_label = str(gt_label)
    #     filename_gt = train_file_list[i][:-1] + " " + gt_label
    #     annotations.append(filename_gt)
    #
    # with open("../../Stanford40/ImageSplits/test_gt.txt", "r") as ff:
    #     test_file_list = ff.readlines()
    # number = 0
    # current_gt = 0
    # for i in range(len(test_file_list)):
    #     gt_label = test_file_list[i][-2]
    #     if current_gt != gt_label:
    #         current_gt = gt_label
    #         number = 0
    #     number += 1
    #     if number > 80:
    #         continue
    #     annotations.append(test_file_list[i][:-1])
    #
    # with open("../../Stanford40/ImageSplits/train7200_gt.txt", "w+") as fff:
    #     for i in range(7200):
    #         fff.write(str(annotations[i])+"\n")


    # with open("../../Stanford40/ImageSplits/test_gt.txt", "r") as ff:
    #     test_file_list = ff.readlines()
    # number = 0
    # current_gt_end = 0
    # for i in range(len(test_file_list)):
    #     gt_label_end = test_file_list[i][-2]
    #     if current_gt_end != gt_label_end:
    #         current_gt_end = gt_label_end
    #         number = 0
    #     number += 1
    #     if number <= 80:
    #         continue
    #     annotations.append(test_file_list[i][:-1])
    #
    # with open("../../Stanford40/ImageSplits/test2332.txt", "w+") as fff:
    #     for i in range(len(annotations)):
    #         fff.write(str(annotations[i])+"\n")



    annotations_train=[]
    annotations_test=[]
    with open("../../my_datasets/Stanford40/txt_annotations/self_waving_hands_classfication_testlabel/self_waving_hands_classfication.txt", "r") as f:
        train_file_list = f.readlines()
    number = 0
    current_name = " "
    for i in range(len(train_file_list)):
        filename = train_file_list[i][:-9]
        gt_label = CLASSES[filename]
        gt_label = str(gt_label)
        filename_gt = train_file_list[i][:-1] + " " + gt_label
        annotations_train.append(filename_gt)
        # if current_name != filename:
        #     current_name = filename
        #     number = 0
        # number += 1
        # if number <= 180:
        #     annotations_train.append(filename_gt)
        # else:
        #     annotations_test.append(filename_gt)

    with open("../../my_datasets/Stanford40/txt_annotations/self_waving_hands_classfication_testlabel/self_waving_hands_classfication_testlabel.txt", "w+") as ff:
        for i in range(len(annotations_train)):
            ff.write(str(annotations_train[i])+"\n")

    # with open("../../Stanford40/txt_annotations/18class_test.txt", "w+") as fff:
    #     for i in range(len(annotations_test)):
    #         fff.write(str(annotations_test[i])+"\n")

if __name__ == '__main__':
    main()
