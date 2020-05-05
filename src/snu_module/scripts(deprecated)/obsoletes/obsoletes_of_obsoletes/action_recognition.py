import cv2
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn import svm
from PIL import Image


clf = joblib.load('/home/kyle/USR_SNU_MODULE/SNU_Integrated_v2/src/snu_module/scripts/trained_non_cor_10_100.pkl')


def stochastic_voting(pose_list, pose_probs):
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / float(np.sum(e_x))

    def normalize(x, a=0, b=1):
        x_min = np.min(x)
        x_max = np.max(x)
        return a + np.multiply((x-x_min), (b-a)) / float(x_max-x_min)

    # Posterior Probabilities
    pose_posterior = np.ones((1, 3))
    for prob_idx, prob in enumerate(pose_probs):
        # pose_posterior = np.multiply(pose_posterior, softmax(prob).reshape(1, 3))
        pose_posterior = np.multiply(pose_posterior, normalize(prob).reshape(1, 3))

    # Occurrence Array
    occurrence, _ = np.histogram(pose_list, bins=3, range=(0, 2))

    # Probabilistic Occurrence
    occurrence_prob = np.multiply(occurrence, pose_posterior)

    return occurrence_prob.argmax()


def svm_clf(color_img, trackers):
    H = color_img.shape[0]
    W = color_img.shape[1]
    for tracker_idx, tracker in enumerate(trackers):

        a = max(0, int(tracker.x[1]-(tracker.x[5]/2)))
        b = min(int(tracker.x[1]+(tracker.x[5]/2)), H-1)
        c = max(0, int(tracker.x[0]-(tracker.x[4]/2)))
        d = min(int(tracker.x[0]+(tracker.x[4]/2)), W-1)
        # print a, b, c, d

        if (a >= b) or (c >= d):
            tracker.update_action(0, None)
        else :
            crop_image = color_img[a:b, c:d]
            cr_rsz_img = cv2.resize(crop_image, dsize=(60, 60), interpolation=cv2.INTER_AREA)
            hog_feature = hog(cr_rsz_img, orientations=8, pixels_per_cell=(6, 6), visualize=False, multichannel=True)
            tracker.update_action(clf.predict([hog_feature]), clf.decision_function([hog_feature]))
            # tracker.pose_list.insert(0, clf.predict([hog_feature]))
            # tracker.pose_probs.insert(0, clf.decision_function([hog_feature]))

        # print tracker.pose
        trackers[tracker_idx] = tracker

        tracker.pose = stochastic_voting(tracker.pose_list, tracker.pose_probs)
        # print tracker.pose_list

    return trackers
