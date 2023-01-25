import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from stockfish import Stockfish

stockfish = Stockfish(r'C:/Users/15193/OneDrive/PYTHON/ChessDetect/SF/stockfish-11-win/stockfish-11-win/Windows/stockfish_20011801_x64.exe')

# FLANN PARAMS

# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=200)

# flann = cv.FlannBasedMatcher(index_params, search_params)

# GET IMAGES

imc = cv.imread('ChessDetect/test/test12.jpg')
im = cv.imread('ChessDetect/test/test12.jpg', cv.IMREAD_GRAYSCALE)

# GET TURN

turn = input("turn (w/b): ")

# blank = cv.imread('ChessDetect/templates/boardcorner/blank.jpg', cv.IMREAD_GRAYSCALE)

# START SIFT

# sift = cv.SIFT_create()

# blank_pts, blank_desc = sift.detectAndCompute(blank, None)
# board_pts, board_desc = sift.detectAndCompute(im, None)

# FEATURE MATCHING

# matches = flann.knnMatch(blank_desc, board_desc, k=2)

# masked_blank_pts = []
# masked_board_pts = []

# lowe_coeff = 0.8

# for i, (m,n) in enumerate(matches):
#     if m.distance < lowe_coeff * n.distance:
#         masked_board_pts.append(board_pts[m.trainIdx].pt)
#         masked_blank_pts.append(blank_pts[m.queryIdx].pt)

# masked_blank_pts = np.int32(masked_blank_pts)
# masked_board_pts = np.int32(masked_board_pts)

# F, mask = cv.findFundamentalMat(masked_blank_pts, masked_board_pts, cv.FM_LMEDS)

# masked_blank_pts = masked_blank_pts[mask.ravel()==1]
# masked_board_pts = masked_board_pts[mask.ravel()==1]

# CREATE BOARD OUTLINE

# leftmost = min(masked_board_pts[:, 0])
# rightmost = max(masked_board_pts[:, 0])

# uppermost = min(masked_board_pts[:, 1])
# lowermost = max(masked_board_pts[:, 1])

# bound_safety = 10

# GET BOARD PARAMS

# board_crop = im[uppermost - bound_safety:lowermost + bound_safety, leftmost - bound_safety:rightmost + bound_safety]
square_sl = int(im.shape[0] / 8)

# GET PIECE LOCATIONS

dataset = 'ChessDetect/templates/128h/'

board_data = np.zeros((8, 8), dtype=np.int32)
key_arr = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']

icount = 0

for item in os.listdir(dataset):

    piece = cv.imread(dataset + item, cv.IMREAD_GRAYSCALE)
    piece_res = cv.resize(piece, (int(square_sl), int(square_sl)))

    tm_result = cv.matchTemplate(im, piece_res, cv.TM_CCOEFF_NORMED)
    
    threshold = 0

    if icount < 6:
        threshold = 0.85
    else:
        threshold = 0.7

    locations = np.where(tm_result >= threshold)
    locations = list(zip(*locations[::-1]))

    sqs = []
    for piece_loc in locations:
        file = piece_loc[0] * 8 / (im.shape[0])
        file = round(file)
        #file = chr(ord('a') + file - 1)

        rank = (piece_loc[1] * 8 / (im.shape[0]))
        rank = round(rank)

        pos = [file, rank]

        sqs.append(pos)
    
    sqs_clear = []

    for sq in sqs:
        if sq not in sqs_clear:
            sqs_clear.append(sq)

    # print(item + ": ")
    # print(sqs_clear)

    for sq in sqs_clear:
        r = sq[1]
        f = sq[0]
        board_data[r][f] = ord(key_arr[icount])

    icount = icount + 1

# GET FEN CODE
fen = ""

for r in range (8):
    zcount = 0
    for f in range (8):
        if board_data[r][f] == 0:
            zcount = zcount + 1
            if f == 7:
                fen = fen + str(zcount)
        else:
            if zcount > 0:
                fen = fen + str(zcount)
                zcount = 0
            fen = fen + chr(board_data[r][f])
    if r != 7: fen = fen + "/"

fen = fen + " " + turn + " - - 0 1"

print(fen)

# GET BEST MOVE

# print(stockfish.is_fen_valid(fen))

if stockfish.is_fen_valid(fen):
    stockfish.set_fen_position(fen)
    best_move = stockfish.get_best_move()

start_f, start_r, end_f, end_r = best_move[0], best_move[1], best_move[2], best_move[3]

start_pt = [round((ord(start_f) - ord('a')) * im.shape[0] / 8) + int(square_sl)/2, round((8 - int(start_r)) * im.shape[0] / 8) + int(square_sl)/2]
end_pt = [round((ord(end_f) - ord('a')) * im.shape[0] / 8) + int(square_sl)/2, round((8 - int(end_r)) * im.shape[0] / 8) + int(square_sl)/2]

start_pt = np.int32(start_pt)
end_pt = np.int32(end_pt)

# DISPLAY

imc = cv.line(imc, start_pt, end_pt, (0, 255, 0), 2)
imc = cv.circle(imc, end_pt, radius=5, color=(0, 255, 0), thickness=3)

# out = im

# for pt in masked_board_pts:
#     out = cv.circle(out, (int(pt[0]), int(pt[1])), radius=5, color=(0, 255, 0), thickness=3)

plt.imshow(cv.cvtColor(imc, cv.COLOR_BGR2RGB))
plt.show()

# MUST SPECIFY WHICH COLOUR IS BEING PLAYED AS AND WHO'S TURN IT IS

##### TO DO #####
# implement play for black
# create templates based on initial set up
# live play